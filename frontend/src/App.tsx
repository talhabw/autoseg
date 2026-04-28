import { useEffect, useCallback } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Header } from './components/Layout/Header';
import { Sidebar } from './components/Layout/Sidebar';
import { ImageCanvas } from './components/Canvas/ImageCanvas';
import { CreateProjectModal } from './components/Modals/CreateProjectModal';
import { OpenProjectModal } from './components/Modals/OpenProjectModal';
import { ExportModal } from './components/Modals/ExportModal';
import { SettingsModal } from './components/Modals/SettingsModal';
import { LoadingOverlay } from './components/LoadingOverlay';
import { PerformancePropagationOverlay } from './components/PerformancePropagationOverlay';
import { Toaster } from "@/components/ui/sonner";
import { useProjectStore } from './stores/projectStore';
import { useAnnotationStore } from './stores/annotationStore';
import { useUIStore } from './stores/uiStore';
import { useNotifications } from './hooks/useNotifications';
import * as api from './api/client';

const queryClient = new QueryClient();

// Helper to compute bounding box IoU for client-side duplicate detection
function bboxIoU(box1: number[], box2: number[]): number {
  const [x1_1, y1_1, x2_1, y2_1] = box1;
  const [x1_2, y1_2, x2_2, y2_2] = box2;
  
  const xi1 = Math.max(x1_1, x1_2);
  const yi1 = Math.max(y1_1, y1_2);
  const xi2 = Math.min(x2_1, x2_2);
  const yi2 = Math.min(y2_1, y2_2);
  
  const interWidth = Math.max(0, xi2 - xi1);
  const interHeight = Math.max(0, yi2 - yi1);
  const interArea = interWidth * interHeight;
  
  const area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
  const area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
  const unionArea = area1 + area2 - interArea;
  
  return unionArea > 0 ? interArea / unionArea : 0;
}

// Module-level state - truly synchronous, survives React re-renders
let _propagationLock = false;
let _lastPropagationTime = 0;
let _propagationRequestId = 0;
let _pendingAutoNext = false; // Flag to trigger auto-next after annotations load
const _sessionFailedImageIds = new Set<number>(); // Track failed images in current session (skip mode)
let _performancePropagationActive = false; // Performance mode loop active

const NAVIGATION_LOAD_DEBOUNCE_MS = 90;
const LAST_IMAGE_INDEX_SAVE_DEBOUNCE_MS = 180;

// Export function to stop auto-tracking (used by UI components)
// eslint-disable-next-line react-refresh/only-export-components
export function stopAutoTracking() {
  _pendingAutoNext = false;
  _performancePropagationActive = false;
  _sessionFailedImageIds.clear(); // Clear failed images on stop
  // Increment request ID to cancel any in-progress propagation
  _propagationRequestId++;
  console.log(`stopAutoTracking: Incremented _propagationRequestId to ${_propagationRequestId}, cleared failed images`);
}

// Performance propagation: tight loop without UI refreshes per frame
// Only updates the progress overlay, not ImageCanvas or annotation store
async function runPerformancePropagation() {
  if (_performancePropagationActive) {
    console.log('[PERF] Already running, ignoring');
    return;
  }

  _performancePropagationActive = true;
  _propagationRequestId++;
  const myRequestId = _propagationRequestId;

  const { images, currentImageIndex: startIdx, project } = useProjectStore.getState();
  const { propagationLoaded, loadPropagation, setStatusMessage } = useUIStore.getState();
  // Note: Can't use hooks here since this is a module-level function

  if (!project) {
    _performancePropagationActive = false;
    return;
  }

  // Ensure models are loaded
  if (!propagationLoaded) {
    setStatusMessage('Loading tracking models...');
    await loadPropagation();
  }

  const totalImages = images.length - 1 - startIdx; // Images remaining to propagate
  if (totalImages <= 0) {
    useUIStore.getState().addToast('Already at last image', 'info');
    _performancePropagationActive = false;
    return;
  }

  // Initialize progress overlay
  useUIStore.getState().setPerformanceProgress({
    current: 0,
    total: totalImages,
    successCount: 0,
    failedCount: 0,
    skippedCount: 0,
    startTime: Date.now(),
    failedImageIndex: null,
    failedLabels: [],
  });

  let currentIdx = startIdx;

  while (currentIdx < images.length - 1 && _performancePropagationActive) {
    // Check if request was cancelled
    if (myRequestId !== _propagationRequestId) {
      console.log(`[PERF] Aborted: request superseded (${myRequestId} vs ${_propagationRequestId})`);
      break;
    }

    const sourceImageId = images[currentIdx].id;
    const targetImageId = images[currentIdx + 1].id;
    const imageNum = currentIdx - startIdx + 1;

    console.log(`[PERF] Processing image ${imageNum}/${totalImages}: ${sourceImageId} -> ${targetImageId}`);

    // Get source annotations directly from API (don't touch the store)
    let sourceAnnotations: Awaited<ReturnType<typeof api.listAnnotations>>;
    try {
      sourceAnnotations = await api.listAnnotations(sourceImageId);
    } catch (err) {
      console.error(`[PERF] Failed to fetch annotations for image ${sourceImageId}:`, err);
      useUIStore.getState().updatePerformanceProgress({ failedCount: (useUIStore.getState().performanceProgress?.failedCount ?? 0) + 1 });
      currentIdx++;
      continue;
    }

    if (sourceAnnotations.length === 0) {
      console.log(`[PERF] No annotations on image ${sourceImageId}, skipping`);
      useUIStore.getState().updatePerformanceProgress({
        current: imageNum,
        skippedCount: (useUIStore.getState().performanceProgress?.skippedCount ?? 0) + 1,
      });
      currentIdx++;
      continue;
    }

    // Propagate all annotations
    const propagationResults: Array<{
      label_id: number;
      bbox: [number, number, number, number];
      mask_rle: object;
      polygon: number[];
    }> = [];
    const failedLabels = new Set<number>();
    let duplicateSkipCount = 0;

    const { sizeMinRatio, sizeMaxRatio, stopOnSizeMismatch, topK, useBBoxHint, bboxHintScale, pruneThinArtifacts, propagationMode, iouVerify, iouThreshold } = useUIStore.getState();

    for (const ann of sourceAnnotations) {
      if (!ann.bbox) continue;
      if (myRequestId !== _propagationRequestId) break;

      try {
        const useAdvancedApi = propagationMode !== 'peak' || iouVerify;
        let result;

        if (useAdvancedApi) {
          result = await api.propagateAdvanced(
            sourceImageId, targetImageId, ann.id,
            {
              mode: propagationMode, iouVerify, iouThreshold,
              useCachedMasks: true, useBBoxHint, bboxHintScale, pruneThinArtifacts, sizeMinRatio, sizeMaxRatio,
              stopOnSizeMismatch, topK, skipDuplicateThreshold: 0.9,
            }
          );
        } else {
          result = await api.propagate(
            sourceImageId, targetImageId, ann.id,
            sizeMinRatio, sizeMaxRatio, stopOnSizeMismatch,
            0.9, topK, useBBoxHint, bboxHintScale, pruneThinArtifacts
          );
        }

        if (result.duplicate_skipped) {
          duplicateSkipCount++;
          continue;
        }

        // Client-side batch duplicate check
        const BBOX_IOU_THRESHOLD = 0.85;
        let isDuplicate = false;
        for (const existing of propagationResults) {
          if (bboxIoU(result.bbox, existing.bbox) >= BBOX_IOU_THRESHOLD) {
            isDuplicate = true;
            break;
          }
        }
        if (isDuplicate) {
          duplicateSkipCount++;
          continue;
        }

        propagationResults.push({
          label_id: ann.label_id,
          bbox: result.bbox,
          mask_rle: result.mask_rle,
          polygon: result.polygon,
        });
      } catch (err) {
        console.error(`[PERF] Failed to propagate annotation ${ann.id}:`, err);
        failedLabels.add(ann.label_id);
      }
    }

    // Try fallback for failed labels
    const stillFailedLabels: { labelId: number; labelName: string }[] = [];
    if (failedLabels.size > 0 && project) {
      for (const labelId of failedLabels) {
        let fallbackSuccess = false;
        const triedImageIds: number[] = [..._sessionFailedImageIds];
        const MAX_FALLBACK_ATTEMPTS = 2;

        for (let attempt = 0; attempt < MAX_FALLBACK_ATTEMPTS && !fallbackSuccess; attempt++) {
          try {
            const fallbackResult = await api.findFallbackReference(
              labelId, currentIdx + 1, project.id,
              triedImageIds.length > 0 ? triedImageIds : undefined
            );

            if (!fallbackResult.found || !fallbackResult.annotation) break;
            triedImageIds.push(fallbackResult.annotation.image_id);

            const result = await api.propagateAdvanced(
              fallbackResult.annotation.image_id, targetImageId, fallbackResult.annotation.id,
              {
                mode: propagationMode, iouVerify, iouThreshold,
                useCachedMasks: true, useBBoxHint, bboxHintScale, pruneThinArtifacts, sizeMinRatio, sizeMaxRatio,
                stopOnSizeMismatch, topK, skipDuplicateThreshold: 0.9,
              }
            );

            if (result.duplicate_skipped) {
              duplicateSkipCount++;
              fallbackSuccess = true;
              continue;
            }

            propagationResults.push({
              label_id: labelId,
              bbox: result.bbox,
              mask_rle: result.mask_rle,
              polygon: result.polygon,
            });
            fallbackSuccess = true;
          } catch (err) {
            console.error(`[PERF] Fallback attempt ${attempt + 1} failed for label ${labelId}:`, err);
          }
        }

        if (!fallbackSuccess) {
          const labels = useAnnotationStore.getState().labels;
          const label = labels.find(l => l.id === labelId);
          stillFailedLabels.push({ labelId, labelName: label?.name || `Label ${labelId}` });
        }
      }
    }

    // Save successful results via API
    for (const result of propagationResults) {
      try {
        await api.createAnnotation({
          image_id: targetImageId,
          label_id: result.label_id,
          bbox: result.bbox,
          mask_rle: result.mask_rle,
          polygon: result.polygon,
          source: 'tracked',
          status: 'pending',
        });
      } catch (err) {
        console.error(`[PERF] Failed to create annotation:`, err);
      }
    }

    // Update progress
    const progress = useUIStore.getState().performanceProgress;
    const newSuccessCount = (progress?.successCount ?? 0) + (propagationResults.length > 0 ? 1 : 0);
    const newFailedCount = (progress?.failedCount ?? 0) + (stillFailedLabels.length > 0 ? 1 : 0);

    // Handle failures based on propagation failure mode
    if (stillFailedLabels.length > 0) {
      _sessionFailedImageIds.add(targetImageId);
      const { propagationFailureMode } = useUIStore.getState();

      if (propagationFailureMode === 'stop') {
        // STOP: Pause the loop, show the failed image to user
        console.log(`[PERF] Stopping at image ${currentIdx + 1} - failed labels: ${stillFailedLabels.map(l => l.labelName).join(', ')}`);

        // Navigate to the failed image so user can see and fix it
        useProjectStore.getState().setCurrentImageIndex(currentIdx + 1);

        useUIStore.getState().updatePerformanceProgress({
          current: imageNum,
          successCount: newSuccessCount,
          failedCount: newFailedCount,
          failedImageIndex: currentIdx + 1,
          failedLabels: stillFailedLabels.map(l => l.labelName),
        });

        // Wait for user to resume or stop
        await waitForResumeOrStop(myRequestId);

        if (!_performancePropagationActive || myRequestId !== _propagationRequestId) {
          console.log(`[PERF] Stopped after pause`);
          break;
        }

        // User resumed - reload annotations for the current target image
        // (user may have fixed annotations on it)
        console.log(`[PERF] Resuming from image ${currentIdx + 1}`);
        // Move forward (user fixed this image, continue from here)
        currentIdx++;
        continue;
      }
      // Skip mode: just continue
    }

    useUIStore.getState().updatePerformanceProgress({
      current: imageNum,
      successCount: newSuccessCount,
      failedCount: newFailedCount,
      skippedCount: (progress?.skippedCount ?? 0) + duplicateSkipCount,
    });

    currentIdx++;
  }

  // Done - navigate to where we ended up
  const finalIdx = Math.min(currentIdx, images.length - 1);
  useProjectStore.getState().setCurrentImageIndex(finalIdx);

  // Show summary toast
  const finalProgress = useUIStore.getState().performanceProgress;
  if (finalProgress && _performancePropagationActive) {
    const elapsed = ((Date.now() - finalProgress.startTime) / 1000).toFixed(1);
    useUIStore.getState().addToast(
      `Performance propagation complete: ${finalProgress.successCount} succeeded, ${finalProgress.failedCount} failed in ${elapsed}s`,
      finalProgress.failedCount > 0 ? 'warning' : 'success',
      5000
    );
  }

  // Cleanup
  _performancePropagationActive = false;
  _propagationLock = false;
  useUIStore.getState().setIsPropagating(false);
  useUIStore.getState().setPerformanceProgress(null);
}

// Helper: wait until user clicks Resume or Stop in the overlay
function waitForResumeOrStop(requestId: number): Promise<void> {
  return new Promise((resolve) => {
    const check = () => {
      const progress = useUIStore.getState().performanceProgress;
      // Resolve when: no progress (stopped), no failed image (resumed), or request superseded
      if (!progress || progress.failedImageIndex === null || requestId !== _propagationRequestId || !_performancePropagationActive) {
        resolve();
        return;
      }
      setTimeout(check, 200);
    };
    check();
  });
}

function AppContent() {
  const { images, currentImageIndex, project, tryOpenLastProject } = useProjectStore();
  const {
    loadAnnotations,
    loadLabels,
    clearAnnotations,
  } = useAnnotationStore();
  const {
    setIsPropagating,
    setStatusMessage,
    checkModelStatus,
  } = useUIStore();

  const { requestPermission, notifyPropagationFailure } = useNotifications();

  const currentImage = images[currentImageIndex];

  // On app load: try to reconnect to last project and check model status
  useEffect(() => {
    const init = async () => {
      setStatusMessage('Initializing...');

      // Check if models are already loaded on backend
      await checkModelStatus();

      // Try to open last project
      const opened = await tryOpenLastProject();
      if (opened) {
        setStatusMessage('Reopened last project');
      } else {
        setStatusMessage('Ready - Open or create a project');
      }
    };

    init();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Mount-only: reads fresh state from stores

  // Load annotations when image changes
  useEffect(() => {
    if (!currentImage) {
      clearAnnotations();
      return;
    }

    clearAnnotations();

    let cancelled = false;
    const abortController = new AbortController();
    const loadDelay = _pendingAutoNext ? 0 : NAVIGATION_LOAD_DEBOUNCE_MS;

    const timeoutId = window.setTimeout(() => {
      loadAnnotations(currentImage.id, abortController.signal)
        .then(() => {
          if (cancelled || abortController.signal.aborted) return;

          // Check if we should continue auto-propagation
          if (_pendingAutoNext) {
            _pendingAutoNext = false;
            const { autoNext, trackModeEnabled } = useUIStore.getState();
            const { currentImageIndex: idx, images: imgs } = useProjectStore.getState();

            if (autoNext && trackModeEnabled && idx < imgs.length - 1 && !_propagationLock) {
              const { performanceMode } = useUIStore.getState();
              if (performanceMode && !_performancePropagationActive) {
                // Performance mode: run tight loop without UI updates
                runPerformancePropagation();
              } else if (!performanceMode) {
                // Normal mode: small delay to let React render
                setTimeout(() => handlePropagateAndNext(), 50);
              }
              // If performance propagation is already active, do nothing (it manages its own loop)
            }
          }
        })
        .catch((err) => {
          if (abortController.signal.aborted) return;
          console.error('Failed to load annotations:', err);
        });
    }, loadDelay);

    return () => {
      cancelled = true;
      abortController.abort();
      window.clearTimeout(timeoutId);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentImage?.id]); // Only re-run on image change; callbacks read fresh store state

  // Persist last visited image index after navigation settles.
  useEffect(() => {
    if (!project || !currentImage) return;

    const timeoutId = window.setTimeout(() => {
      api.setSetting('last_image_index', String(currentImageIndex)).catch(() => {});
    }, LAST_IMAGE_INDEX_SAVE_DEBOUNCE_MS);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [project?.id, currentImage?.id, currentImageIndex]);

  // Load labels when project changes
  useEffect(() => {
    if (project) {
      loadLabels();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project?.id]); // Only re-run on project change

  // Handle segmentation
  const handleSegment = useCallback(async () => {
    const { selectedAnnotationId: selectedAnn, annotations, refinePoints: points } = useAnnotationStore.getState();
    const { images: currentImages, currentImageIndex: currentIdx } = useProjectStore.getState();
    const currentImg = currentImages[currentIdx];

    if (!selectedAnn || !currentImg) return;

    const annotation = annotations.find((a) => a.id === selectedAnn);
    if (!annotation?.bbox) return;

    const { samLoaded: isSamLoaded, loadSAM: loadSamModel, setStatusMessage: setStatus } = useUIStore.getState();

    if (!isSamLoaded) {
      setStatus('Loading SAM model...');
      await loadSamModel();
    }

    setStatus('Running segmentation...');
    try {
      const posPoints = points
        .filter((p) => p.type === 'positive')
        .map((p) => [p.x, p.y] as [number, number]);
      const negPoints = points
        .filter((p) => p.type === 'negative')
        .map((p) => [p.x, p.y] as [number, number]);

      const result = await api.segment(
        currentImg.id,
        annotation.bbox,
        posPoints.length > 0 ? posPoints : undefined,
        negPoints.length > 0 ? negPoints : undefined
      );

      // Update annotation with mask and polygon
      await useAnnotationStore.getState().updateAnnotation(selectedAnn, {
        bbox: result.bbox,
        mask_rle: result.mask_rle,
        polygon: result.polygon,
      });

      // Don't clear points after refinement - let them accumulate for iterative refinement
      // User can press Escape to clear manually, or points clear when selecting different annotation
      useUIStore.getState().addToast(`Segmentation complete (score: ${(result.score * 100).toFixed(0)}%)`, 'success');
      useUIStore.getState().setMode('refine');
    } catch (err) {
      useUIStore.getState().addToast('Segmentation failed', 'error');
      console.error('Segmentation error:', err);
    }
  }, []); // No dependencies - reads fresh state from stores

  // Handle propagation to next frame
  const handlePropagateAndNext = useCallback(async () => {
    const timestamp = Date.now();
    const logPrefix = `[PROP ${timestamp}]`;

    console.log(`${logPrefix} ========== handlePropagateAndNext called ==========`);
    console.log(`${logPrefix} _propagationLock=${_propagationLock}, _lastPropagationTime=${_lastPropagationTime}, _propagationRequestId=${_propagationRequestId}`);

    // Module-level lock check - truly synchronous, prevents race conditions
    if (_propagationLock) {
      console.log(`${logPrefix} ❌ BLOCKED: Lock is held, ignoring key press`);
      return;
    }

    // Debouncing: Check time since last propagation
    const now = Date.now();
    if (now - _lastPropagationTime < 150) {
      console.log(`${logPrefix} ❌ BLOCKED: Debounce (${now - _lastPropagationTime}ms since last)`);
      return;
    }

    // Set lock immediately (synchronous) BEFORE any async operation or state read
    _propagationLock = true;
    _lastPropagationTime = now;
    _propagationRequestId++;
    const currentRequestId = _propagationRequestId;
    
    // Set isPropagating immediately so UI shows stop button right away
    setIsPropagating(true);

    console.log(`${logPrefix} ✅ Lock acquired, requestId=${currentRequestId}`);

    // Request notification permission early (non-blocking)
    // This allows us to notify the user if propagation fails
    requestPermission().catch(() => {
      console.log(`${logPrefix} Notification permission not granted (non-critical)`);
    });

    // Read fresh state from stores to avoid closure issues
    const { images: currentImages, currentImageIndex: currentIdx, nextImage: goNext, project } = useProjectStore.getState();
    const currentImg = currentImages[currentIdx];

    console.log(`${logPrefix} State: currentIdx=${currentIdx}, currentImg.id=${currentImg?.id}, imageCount=${currentImages.length}`);

    if (!currentImg || currentIdx >= currentImages.length - 1) {
      console.log(`${logPrefix} ⏭️ At last image or no image, just navigating without propagation`);
      _propagationLock = false;
      setIsPropagating(false);
      goNext();
      return;
    }

    // Capture the target image BEFORE any async operations
    const targetImageRecord = currentImages[currentIdx + 1];
    const targetImageId = targetImageRecord.id;

    console.log(`${logPrefix} Target: targetIdx=${currentIdx + 1}, targetImageId=${targetImageId}`);

    // Capture annotations from CURRENT image (copy to avoid mutation issues)
    // CRITICAL: Filter by image_id to prevent race condition where loadAnnotations
    // has already updated the store with annotations from another image
    let allAnnotations = useAnnotationStore.getState().annotations;
    let sourceAnnotations = allAnnotations.filter(ann => ann.image_id === currentImg.id);

    console.log(`${logPrefix} Source annotations: ${sourceAnnotations.length} for image ${currentImg.id} (store has ${allAnnotations.length} total)`);

    // If no annotations found, wait briefly for loadAnnotations to complete (retry up to 500ms)
    if (sourceAnnotations.length === 0) {
      console.log(`${logPrefix} ⏳ No annotations yet, waiting for load...`);

      // Retry up to 5 times with 100ms delays
      for (let retry = 0; retry < 5 && sourceAnnotations.length === 0; retry++) {
        await new Promise(resolve => setTimeout(resolve, 100));
        allAnnotations = useAnnotationStore.getState().annotations;
        sourceAnnotations = allAnnotations.filter(ann => ann.image_id === currentImg.id);
        console.log(`${logPrefix} Retry ${retry + 1}: ${sourceAnnotations.length} annotations for image ${currentImg.id}`);
      }
    }

    if (sourceAnnotations.length === 0) {
      // Still no annotations after waiting - truly no annotations on this image
      console.log(`${logPrefix} ⏸️ No annotations for source image ${currentImg.id} after waiting`);
      
      // If auto-tracking, skip this image and continue to next
      const { autoNext: shouldAutoNext, trackModeEnabled: isTracking } = useUIStore.getState();
      if (shouldAutoNext && isTracking && currentIdx < currentImages.length - 1) {
        console.log(`${logPrefix} 🔁 Auto-tracking: skipping unlabeled image, moving to next`);
        useUIStore.getState().addToast('Skipping unlabeled image...', 'info', 1000);
        _pendingAutoNext = true;
        _propagationLock = false;
        setIsPropagating(false);
        goNext();
        return;
      }
      
      useUIStore.getState().addToast('No annotations to propagate. Add annotations first.', 'info');
      _propagationLock = false;
      setIsPropagating(false);
      // Don't call goNext() - stay on current image
      return;
    }

    // Track which labels failed propagation from previous image
    const failedLabels = new Set<number>();
    let fallbackCount = 0;

    try {
      const { propagationLoaded: propLoaded, loadPropagation: loadProp, setStatusMessage: setStatus } = useUIStore.getState();

      if (!propLoaded) {
        console.log(`${logPrefix} Loading propagation models...`);
        setStatus('Loading tracking models...');
        await loadProp();
        console.log(`${logPrefix} Propagation models loaded`);
      }

      // Check if request was superseded
      if (currentRequestId !== _propagationRequestId) {
        console.log(`${logPrefix} ❌ ABORTED: Request superseded after model loading (${currentRequestId} vs ${_propagationRequestId})`);
        return;
      }

      setStatus(`Tracking ${sourceAnnotations.length} annotations...`);

      // Collect all propagation results BEFORE creating any annotations
      const propagationResults: Array<{
        label_id: number;
        bbox: [number, number, number, number];
        mask_rle: object;
        polygon: number[];
      }> = [];
      let duplicateSkipCount = 0;
      const notFoundLabels: { labelId: number; labelName: string }[] = [];  // Labels where object wasn't found in target

      for (let i = 0; i < sourceAnnotations.length; i++) {
        const ann = sourceAnnotations[i];
        if (!ann.bbox) continue;

        // Check if request was superseded
        if (currentRequestId !== _propagationRequestId) {
          console.log(`${logPrefix} ❌ ABORTED: Request superseded mid-loop at ann ${i} (${currentRequestId} vs ${_propagationRequestId})`);
          return;
        }

        try {
          console.log(`${logPrefix} Propagating annotation ${i + 1}/${sourceAnnotations.length} (id=${ann.id})`);

          // Get propagation settings from store
          const { sizeMinRatio, sizeMaxRatio, stopOnSizeMismatch, topK, useBBoxHint, bboxHintScale, pruneThinArtifacts, propagationMode, iouVerify, iouThreshold } = useUIStore.getState();
          
          // Use advanced propagation API when mode is not 'peak' or IoU verification is enabled
          const useAdvancedApi = propagationMode !== 'peak' || iouVerify;
          
          let result;
          if (useAdvancedApi) {
            result = await api.propagateAdvanced(
              currentImg.id,
              targetImageId,
              ann.id,
              {
                mode: propagationMode,
                iouVerify,
                iouThreshold,
                useCachedMasks: true,
                useBBoxHint,
                bboxHintScale,
                pruneThinArtifacts,
                sizeMinRatio,
                sizeMaxRatio,
                stopOnSizeMismatch,
                topK,
                skipDuplicateThreshold: 0.9,  // Skip if 90%+ overlap with existing
              }
            );
          } else {
            result = await api.propagate(
              currentImg.id,
              targetImageId,
              ann.id,
              sizeMinRatio,
              sizeMaxRatio,
              stopOnSizeMismatch,
              0.9,  // skipDuplicateThreshold - skip if 90%+ overlap with existing
              topK,
              useBBoxHint,
              bboxHintScale,
              pruneThinArtifacts
            );
          }

          // Check if this was a duplicate (detected by backend against existing DB annotations)
          if (result.duplicate_skipped) {
            // Check if it's because the object matched a different class (object not found in target)
            if (result.conflicting_label_name) {
              // Get the label name for the source annotation
              const sourceLabel = useAnnotationStore.getState().labels.find(l => l.id === ann.label_id);
              const sourceLabelName = sourceLabel?.name || `Label ${ann.label_id}`;
              console.log(`${logPrefix} Object not found: "${sourceLabelName}" matched existing "${result.conflicting_label_name}" (IoU=${result.duplicate_iou?.toFixed(3)})`);
              notFoundLabels.push({ labelId: ann.label_id, labelName: sourceLabelName });
            } else {
              console.log(`${logPrefix} Duplicate detected for ann ${ann.id} (IoU=${result.duplicate_iou?.toFixed(3)}), skipping`);
            }
            duplicateSkipCount++;
            continue;  // Don't add to propagationResults
          }

          // Client-side duplicate check: compare against results already in this batch
          // This catches duplicates that haven't been saved to DB yet
          const BBOX_IOU_THRESHOLD = 0.85;  // Use bbox IoU as proxy for mask IoU
          let isBatchDuplicate = false;
          for (const existing of propagationResults) {
            const iou = bboxIoU(result.bbox, existing.bbox);
            if (iou >= BBOX_IOU_THRESHOLD) {
              console.log(`${logPrefix} Batch duplicate detected for ann ${ann.id} (bbox IoU=${iou.toFixed(3)}), skipping`);
              isBatchDuplicate = true;
              break;
            }
          }
          if (isBatchDuplicate) {
            duplicateSkipCount++;
            continue;
          }

          if (result.fallback_used) {
            console.warn(`${logPrefix} Fallback used for ann ${ann.id} (ratio=${result.area_ratio?.toFixed(2)}, conf=${result.confidence?.toFixed(2)})`);
          } else {
            console.log(`${logPrefix} Propagation success for ann ${ann.id}: conf=${result.confidence?.toFixed(2)}, ratio=${result.area_ratio?.toFixed(2)}`);
          }

          propagationResults.push({
            label_id: ann.label_id,
            bbox: result.bbox,
            mask_rle: result.mask_rle,
            polygon: result.polygon,
          });
        } catch (err: unknown) {
          console.error(`${logPrefix} Propagation failed for annotation:`, ann.id, err);
          // Track this label as failed for fallback attempt
          failedLabels.add(ann.label_id);
        }
      }

      // Try fallback for failed labels with multiple attempts (if we have a project)
      // MAX_FALLBACK_ATTEMPTS = 2 means: try up to 2 different earlier images per failed label
      const MAX_FALLBACK_ATTEMPTS = 2;
      const stillFailedLabels: { labelId: number; labelName: string }[] = [];

      if (failedLabels.size > 0 && project) {
        console.log(`${logPrefix} 🔄 Attempting fallback for ${failedLabels.size} failed labels (up to ${MAX_FALLBACK_ATTEMPTS} attempts each)...`);
        setStatus(`Finding fallback references for ${failedLabels.size} labels...`);

        for (const labelId of failedLabels) {
          let fallbackSuccess = false;
          const triedImageIds: number[] = [..._sessionFailedImageIds]; // Start with session-failed images excluded

          // Try up to MAX_FALLBACK_ATTEMPTS different reference images
          for (let attempt = 0; attempt < MAX_FALLBACK_ATTEMPTS && !fallbackSuccess; attempt++) {
            try {
              // Find a fallback reference, excluding already-tried images AND session-failed images
              const fallbackResult = await api.findFallbackReference(
                labelId,
                currentIdx + 1, // beforeImageIndex - the target image index
                project.id,
                triedImageIds.length > 0 ? triedImageIds : undefined
              );

              if (!fallbackResult.found || !fallbackResult.annotation) {
                console.log(`${logPrefix} No fallback found for label ${labelId} (attempt ${attempt + 1}/${MAX_FALLBACK_ATTEMPTS})`);
                break; // No more fallbacks available
              }

              // Track this image so we don't try it again
              triedImageIds.push(fallbackResult.annotation.image_id);
              console.log(`${logPrefix} Found fallback for label ${labelId} at image index ${fallbackResult.image_index} (attempt ${attempt + 1})`);

              // Get settings again
              const { sizeMinRatio, sizeMaxRatio, stopOnSizeMismatch, topK, useBBoxHint, bboxHintScale, pruneThinArtifacts, propagationMode, iouVerify, iouThreshold } = useUIStore.getState();

              // Propagate from the fallback reference
              const result = await api.propagateAdvanced(
                fallbackResult.annotation.image_id,
                targetImageId,
                fallbackResult.annotation.id,
                {
                  mode: propagationMode,
                  iouVerify,
                  iouThreshold,
                  useCachedMasks: true,
                  useBBoxHint,
                  bboxHintScale,
                  pruneThinArtifacts,
                  sizeMinRatio,
                  sizeMaxRatio,
                  stopOnSizeMismatch,
                  topK,
                  skipDuplicateThreshold: 0.9,
                }
              );

              if (result.duplicate_skipped) {
                if (result.conflicting_label_name) {
                  const sourceLabel = useAnnotationStore.getState().labels.find(l => l.id === labelId);
                  const sourceLabelName = sourceLabel?.name || `Label ${labelId}`;
                  console.log(`${logPrefix} Fallback object not found: "${sourceLabelName}" matched existing "${result.conflicting_label_name}" (attempt ${attempt + 1})`);
                  // Try next fallback
                  continue;
                } else {
                  console.log(`${logPrefix} Fallback duplicate for label ${labelId}, skipping`);
                }
                duplicateSkipCount++;
                fallbackSuccess = true; // Consider duplicate as "handled"
                continue;
              }

              // Client-side batch duplicate check
              const BBOX_IOU_THRESHOLD = 0.85;
              let isBatchDuplicate = false;
              for (const existing of propagationResults) {
                const iou = bboxIoU(result.bbox, existing.bbox);
                if (iou >= BBOX_IOU_THRESHOLD) {
                  console.log(`${logPrefix} Fallback batch duplicate for label ${labelId} (bbox IoU=${iou.toFixed(3)}), skipping`);
                  isBatchDuplicate = true;
                  break;
                }
              }
              if (isBatchDuplicate) {
                duplicateSkipCount++;
                fallbackSuccess = true;
                continue;
              }

              console.log(`${logPrefix} ✅ Fallback propagation success for label ${labelId} (attempt ${attempt + 1})`);
              propagationResults.push({
                label_id: labelId,
                bbox: result.bbox,
                mask_rle: result.mask_rle,
                polygon: result.polygon,
              });
              fallbackCount++;
              fallbackSuccess = true;
            } catch (err: unknown) {
              console.error(`${logPrefix} Fallback attempt ${attempt + 1} failed for label ${labelId}:`, err);
              // Try next fallback
            }
          }

          // If all fallback attempts failed, track this label
          if (!fallbackSuccess) {
            const sourceLabel = useAnnotationStore.getState().labels.find(l => l.id === labelId);
            const labelName = sourceLabel?.name || `Label ${labelId}`;
            stillFailedLabels.push({ labelId, labelName });
          }
        }
      }

      // Handle failed labels based on propagationFailureMode setting
      if (stillFailedLabels.length > 0) {
        const failedNames = stillFailedLabels.map(l => l.labelName);
        const { propagationFailureMode } = useUIStore.getState();
        
        console.log(`${logPrefix} ⚠️ ${stillFailedLabels.length} labels failed all fallback attempts: ${failedNames.join(', ')} (mode: ${propagationFailureMode})`);
        
        // Mark this image as failed in session (for skip mode reference exclusion)
        _sessionFailedImageIds.add(targetImageId);
        
        if (propagationFailureMode === 'stop') {
          // STOP MODE: Stop propagation, notify user, navigate to failed image
          console.log(`${logPrefix} 🛑 STOPPING: propagationFailureMode='stop'`);
          setStatus(`Propagation stopped - failed to track: ${failedNames.join(', ')}`);
          
          // Send browser notification
          notifyPropagationFailure(failedNames);
          
          // Show toast with details
          const { toast } = await import('sonner');
          toast.error('Propagation Stopped', {
            description: `Could not track: ${failedNames.slice(0, 3).join(', ')}${failedNames.length > 3 ? ` and ${failedNames.length - 3} more` : ''}. Manual annotation required.`,
            duration: 10000,
          });

          // Create annotations for labels that DID succeed before stopping
          if (propagationResults.length > 0) {
            console.log(`${logPrefix} Creating ${propagationResults.length} successful annotations before stopping...`);
            for (const result of propagationResults) {
              try {
                await api.createAnnotation({
                  image_id: targetImageId,
                  label_id: result.label_id,
                  bbox: result.bbox,
                  mask_rle: result.mask_rle,
                  polygon: result.polygon,
                  source: 'tracked',
                  status: 'pending',
                });
              } catch (err) {
                console.error(`${logPrefix} Failed to create annotation:`, err);
              }
            }
          }

          // Navigate to the failed image so user can manually annotate
          console.log(`${logPrefix} Navigating to failed image ${targetImageId} for manual annotation`);
          // Find the index of the target image and navigate to it
          const targetIdx = currentImages.findIndex(img => img.id === targetImageId);
          if (targetIdx >= 0) {
            useProjectStore.getState().setCurrentImageIndex(targetIdx);
          }

          setIsPropagating(false);
          _propagationLock = false;
          return; // STOP - don't continue to next image
        } else {
          // SKIP MODE: Log warning, save successful results, continue to next image
          console.log(`${logPrefix} ⏭️ SKIPPING: propagationFailureMode='skip', continuing to next image`);
          setStatus(`Skipped image - failed to track: ${failedNames.join(', ')}`);
          
          // Show a brief info toast (non-blocking)
          const { toast } = await import('sonner');
          toast.warning('Skipped Image', {
            description: `Could not track: ${failedNames.slice(0, 2).join(', ')}${failedNames.length > 2 ? ` (+${failedNames.length - 2})` : ''}`,
            duration: 3000,
          });
          
          // Still save the successful propagations for this image
          // (fall through to annotation creation below)
        }
      }

      // Check if request was superseded before creating annotations
      if (currentRequestId !== _propagationRequestId) {
        console.log(`${logPrefix} ❌ ABORTED: Request superseded before annotation creation (${currentRequestId} vs ${_propagationRequestId})`);
        return;
      }

      console.log(`${logPrefix} Creating ${propagationResults.length} annotations on image ${targetImageId}...`);

      // Create all annotations atomically for the target image
      // Use direct API calls to avoid polluting current image's annotation state
      for (let i = 0; i < propagationResults.length; i++) {
        const result = propagationResults[i];
        try {
          console.log(`${logPrefix} Creating annotation ${i + 1}/${propagationResults.length} on image ${targetImageId}`);
          await api.createAnnotation({
            image_id: targetImageId,
            label_id: result.label_id,
            bbox: result.bbox,
            mask_rle: result.mask_rle,
            polygon: result.polygon,
            source: 'tracked',
            status: 'pending',
          });
        } catch (err) {
          console.error(`${logPrefix} Failed to create tracked annotation:`, err);
        }
      }

      // Final check before navigation  
      if (currentRequestId !== _propagationRequestId) {
        console.log(`${logPrefix} ❌ ABORTED: Request superseded before navigation (${currentRequestId} vs ${_propagationRequestId})`);
        return;
      }

      // Build summary message
      const totalFailed = sourceAnnotations.length - propagationResults.length - duplicateSkipCount;
      let message = `Tracked ${propagationResults.length}/${sourceAnnotations.length}`;
      if (fallbackCount > 0) {
        message += ` (${fallbackCount} via fallback)`;
      }
      if (totalFailed > 0) {
        message += ` (${totalFailed} failed)`;
      }
      if (duplicateSkipCount > 0) {
        message += ` (${duplicateSkipCount} duplicate${duplicateSkipCount > 1 ? 's' : ''} skipped)`;
      }
      useUIStore.getState().addToast(message, propagationResults.length > 0 ? 'success' : 'warning');

      // Navigate to next image - this will trigger loadAnnotations for the new image
      const prevIdx = useProjectStore.getState().currentImageIndex;
      console.log(`${logPrefix} 🚀 Navigating: currentIdx BEFORE nextImage() = ${prevIdx}`);

      // Set auto-next flag before navigation if enabled
      const { autoNext: shouldAutoNext } = useUIStore.getState();
      if (shouldAutoNext && prevIdx + 1 < currentImages.length) {
        console.log(`${logPrefix} 🔁 Auto-next enabled, setting pending flag`);
        _pendingAutoNext = true;
      }

      useProjectStore.getState().nextImage();

      const newIdx = useProjectStore.getState().currentImageIndex;
      console.log(`${logPrefix} ✅ Navigation complete: currentIdx AFTER nextImage() = ${newIdx}`);

      // If there were labels whose objects weren't found, auto-select the first one and switch to bbox mode
      if (notFoundLabels.length > 0) {
        const firstMissing = notFoundLabels[0];
        console.log(`${logPrefix} 🎯 Auto-selecting missing label "${firstMissing.labelName}" and switching to bbox mode`);
        
        // Set selected label and switch to bbox drawing mode
        useAnnotationStore.getState().selectLabel(firstMissing.labelId);
        useUIStore.getState().setMode('draw');
        
        // Show helpful toast
        const missingNames = notFoundLabels.map(l => l.labelName).join(', ');
        useUIStore.getState().addToast(
          `Not found: ${missingNames}. Draw bbox to add.`,
          'info',
          4000
        );
      }
    } catch (err) {
      // Clear pending flag on error to prevent unwanted auto-propagation
      _pendingAutoNext = false;
      throw err;
    } finally {
      console.log(`${logPrefix} 🔓 Releasing lock, isPropagating=false`);
      setIsPropagating(false);
      _propagationLock = false;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // No dependencies - reads fresh state from stores

  // Approve selected annotation
  const handleApprove = useCallback(async () => {
    const selectedAnn = useAnnotationStore.getState().selectedAnnotationId;
    if (!selectedAnn) return;
    await useAnnotationStore.getState().updateAnnotation(selectedAnn, {
      status: 'approved',
    });
    useUIStore.getState().addToast('Annotation approved', 'success', 1500);
  }, []);

  // Reject selected annotation
  const handleReject = useCallback(async () => {
    const selectedAnn = useAnnotationStore.getState().selectedAnnotationId;
    if (!selectedAnn) return;
    await useAnnotationStore.getState().updateAnnotation(selectedAnn, {
      status: 'rejected',
    });
    useUIStore.getState().addToast('Annotation rejected', 'info', 1500);
  }, []);

  // Jump to next pending image
  const handleNextPending = useCallback(async () => {
    try {
      const result = await api.getImagesWithStatus('pending');
      if (result.image_indices.length === 0) {
        useUIStore.getState().addToast('No pending annotations found', 'info');
        return;
      }

      // Find next pending image after current index
      const currentIdx = useProjectStore.getState().currentImageIndex;
      const nextIdx = result.image_indices.find(idx => idx > currentIdx);
      if (nextIdx !== undefined) {
        useProjectStore.getState().setCurrentImageIndex(nextIdx);
        useUIStore.getState().addToast(`Jumped to image ${nextIdx + 1} (pending)`, 'success', 2000);
      } else {
        // Wrap around to first pending
        useProjectStore.getState().setCurrentImageIndex(result.image_indices[0]);
        useUIStore.getState().addToast(`Wrapped to image ${result.image_indices[0] + 1} (pending)`, 'success', 2000);
      }
    } catch (err) {
      console.error('Failed to get pending images:', err);
    }
  }, []);

  // Jump to next unlabeled image (no annotations at all)
  const handleNextUnlabeled = useCallback(async () => {
    try {
      const { project } = useProjectStore.getState();
      if (!project) {
        useUIStore.getState().addToast('No project open', 'warning');
        return;
      }

      const result = await api.findImagesMissingAnnotations(project.id);
      if (result.image_indices.length === 0) {
        useUIStore.getState().addToast('All images have annotations!', 'success');
        return;
      }

      // Find next unlabeled image after current index
      const currentIdx = useProjectStore.getState().currentImageIndex;
      const nextIdx = result.image_indices.find(idx => idx > currentIdx);
      if (nextIdx !== undefined) {
        useProjectStore.getState().setCurrentImageIndex(nextIdx);
        useUIStore.getState().addToast(`Jumped to image ${nextIdx + 1} (unlabeled) - ${result.total_missing} total`, 'info', 2000);
      } else {
        // Wrap around to first unlabeled
        useProjectStore.getState().setCurrentImageIndex(result.image_indices[0]);
        useUIStore.getState().addToast(`Wrapped to image ${result.image_indices[0] + 1} (unlabeled) - ${result.total_missing} total`, 'info', 2000);
      }
    } catch (err) {
      console.error('Failed to get unlabeled images:', err);
    }
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      // Block ALL keys while propagating (use module-level lock for truly synchronous check)
      if (_propagationLock) {
        console.log('Key blocked - propagation in progress');
        e.preventDefault();
        return;
      }

      // During performance propagation, only allow Escape (to stop)
      if (_performancePropagationActive) {
        if (e.key.toLowerCase() !== 'escape') {
          e.preventDefault();
          return;
        }
      }

      // Read fresh state from stores
      const { trackModeEnabled: trackEnabled, reviewModeEnabled: reviewEnabled, mode: currentMode, propagationLoaded: propLoaded, loadPropagation: loadProp, setMode: setModeAction, setTrackMode: setTrack, setReviewMode: setReview } = useUIStore.getState();
      const { selectedAnnotationId: selectedAnn, refinePoints: points, clearRefinePoints: clearPoints } = useAnnotationStore.getState();
      const { nextImage: goNext, prevImage: goPrev } = useProjectStore.getState();

      switch (e.key.toLowerCase()) {
        case 'a':
        case 'arrowleft':
          goPrev();
          break;
        case 'd':
        case 'arrowright':
          if (trackEnabled) {
            const { performanceMode: perfMode, autoNext: isAutoNext } = useUIStore.getState();
            if (perfMode && isAutoNext && !_performancePropagationActive) {
              // Performance mode with auto-next: run tight loop
              runPerformancePropagation();
            } else {
              // Normal mode or single-step: standard propagation
              handlePropagateAndNext();
            }
          } else {
            goNext();
          }
          break;
        case 'v':
          setModeAction('view');
          break;
        case 'b':
          setModeAction('draw');
          break;
        case 'r':
          setModeAction('refine');
          break;
        case 't':
          if (e.shiftKey) {
            // Shift+T: Toggle auto-tracking (only when track mode is on)
            if (trackEnabled) {
              const { autoNext: currentAutoNext, setAutoNext: setAuto } = useUIStore.getState();
              if (currentAutoNext) {
                // Stop auto-tracking - also cancels in-progress propagation
                stopAutoTracking();
                setAuto(false);
                useUIStore.getState().addToast('Auto-tracking stopped', 'info', 1500);
              } else {
                setAuto(true);
                useUIStore.getState().addToast('Auto-tracking enabled', 'success', 1500);
                // In performance mode, immediately start the tight propagation loop
                const { performanceMode: perfMode2 } = useUIStore.getState();
                if (perfMode2 && !_performancePropagationActive) {
                  runPerformancePropagation();
                }
              }
            } else {
              useUIStore.getState().addToast('Enable track mode first (T)', 'warning', 1500);
            }
          } else {
            // T: Toggle track mode on/off
            if (!trackEnabled) {
              if (!propLoaded) {
                loadProp().then(() => setTrack(true));
              } else {
                setTrack(true);
              }
            } else {
              // Turn off track mode (also disables auto-next and cancels in-progress)
              stopAutoTracking();
              useUIStore.getState().setAutoNext(false);
              setTrack(false);
            }
          }
          break;
        case 'q':
          setReview(!reviewEnabled);
          break;
        case 'y':
        case '1':
          // Approve (Y for Yes, 1 for quick access)
          if (reviewEnabled && selectedAnn) {
            handleApprove();
          }
          break;
        case 'n':
        case '2':
          // Reject (N for No, 2 for quick access)
          if (reviewEnabled && selectedAnn) {
            handleReject();
          }
          break;
        case ']':
          // Jump to next pending image
          handleNextPending();
          break;
        case '[':
          // Jump to next unlabeled image (no annotations)
          handleNextUnlabeled();
          break;
        case 's':
          if (currentMode === 'refine' || selectedAnn) {
            handleSegment();
          }
          break;
        case 'enter':
          if (currentMode === 'refine' && points.length > 0) {
            handleSegment();
          }
          break;
        case 'escape':
          // Stop auto-tracking if in progress, otherwise clear points
          if (useUIStore.getState().autoNext) {
            stopAutoTracking();
            useUIStore.getState().setAutoNext(false);
            useUIStore.getState().addToast('Auto-tracking cancelled', 'info', 1500);
          }
          clearPoints();
          break;
        case 'delete':
        case 'backspace':
          if (selectedAnn) {
            useAnnotationStore.getState().deleteAnnotation(selectedAnn);
          }
          break;
        case 'c':
          // Copy bbox info to console for debug script
          if (selectedAnn) {
            const { annotations } = useAnnotationStore.getState();
            const { images: imgs, currentImageIndex: idx } = useProjectStore.getState();
            const annotation = annotations.find(a => a.id === selectedAnn);
            const currentImg = imgs[idx];
            if (annotation?.bbox && currentImg) {
              const [x1, y1, x2, y2] = annotation.bbox;
              const debugCmd = `python scripts/debug_propagate.py "${currentImg.path}" "<target_image>" ${x1} ${y1} ${x2} ${y2}`;
              console.log('=== DEBUG INFO ===');
              console.log(`Image: ${currentImg.path}`);
              console.log(`Annotation ID: ${annotation.id}`);
              console.log(`BBox: [${x1}, ${y1}, ${x2}, ${y2}]`);
              console.log(`Debug command: ${debugCmd}`);
              console.log('==================');
              useUIStore.getState().addToast(`BBox logged: [${x1.toFixed(0)}, ${y1.toFixed(0)}, ${x2.toFixed(0)}, ${y2.toFixed(0)}]`, 'info', 3000);
            }
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty deps - all callbacks are now stable and read fresh state

  return (
    <div className="h-screen flex flex-col">
      <Header />
      <div className="flex-1 min-h-0 flex overflow-hidden relative">
        <ImageCanvas />
        <Sidebar />
      </div>





      {/* Modals */}
      <CreateProjectModal />
      <OpenProjectModal />
      <ExportModal />
      <SettingsModal />

      {/* Loading overlay for model operations */}
      <LoadingOverlay />

      {/* Performance propagation overlay */}
      <PerformancePropagationOverlay />

      {/* Toast Notifications */}
      <Toaster />
    </div >
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}

export default App;

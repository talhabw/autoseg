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
import { Toaster } from "@/components/ui/sonner";
import { useProjectStore } from './stores/projectStore';
import { useAnnotationStore } from './stores/annotationStore';
import { useUIStore } from './stores/uiStore';
import * as api from './api/client';

const queryClient = new QueryClient();

// Module-level state - truly synchronous, survives React re-renders
let _propagationLock = false;
let _lastPropagationTime = 0;
let _propagationRequestId = 0;
let _pendingAutoNext = false; // Flag to trigger auto-next after annotations load

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
  }, []);

  // Load annotations when image changes
  useEffect(() => {
    if (currentImage) {
      loadAnnotations(currentImage.id).then(() => {
        // Check if we should continue auto-propagation
        if (_pendingAutoNext) {
          _pendingAutoNext = false;
          const { autoNext, trackModeEnabled } = useUIStore.getState();
          const { currentImageIndex: idx, images: imgs } = useProjectStore.getState();
          
          if (autoNext && trackModeEnabled && idx < imgs.length - 1 && !_propagationLock) {
            // Small delay to let React render, but much shorter than 500ms
            setTimeout(() => handlePropagateAndNext(), 50);
          }
        }
      });
    } else {
      clearAnnotations();
    }
  }, [currentImage?.id]);

  // Load labels when project changes
  useEffect(() => {
    if (project) {
      loadLabels();
    }
  }, [project?.id]);

  // Handle segmentation
  const handleSegment = useCallback(async () => {
    const { selectedAnnotationId: selectedAnn, annotations, refinePoints: points, clearRefinePoints: clearPoints } = useAnnotationStore.getState();
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

      clearPoints();
      clearPoints();
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

    console.log(`${logPrefix} ✅ Lock acquired, requestId=${currentRequestId}`);

    // Read fresh state from stores to avoid closure issues
    const { images: currentImages, currentImageIndex: currentIdx, nextImage: goNext, project } = useProjectStore.getState();
    const currentImg = currentImages[currentIdx];

    console.log(`${logPrefix} State: currentIdx=${currentIdx}, currentImg.id=${currentImg?.id}, imageCount=${currentImages.length}`);

    if (!currentImg || currentIdx >= currentImages.length - 1) {
      console.log(`${logPrefix} ⏭️ At last image or no image, just navigating without propagation`);
      _propagationLock = false;
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
      console.log(`${logPrefix} ⏸️ No annotations for source image ${currentImg.id} after waiting, staying on current image`);
      useUIStore.getState().addToast('No annotations to propagate. Add annotations first.', 'info');
      _propagationLock = false;
      // Don't call goNext() - stay on current image
      return;
    }

    setIsPropagating(true);
    console.log(`${logPrefix} isPropagating=true`);

    // Track which labels failed propagation from previous image
    const failedLabels = new Set<number>();
    let successCount = 0;
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
          const { sizeMinRatio, sizeMaxRatio, stopOnSizeMismatch, topK, propagationMode, iouVerify, iouThreshold } = useUIStore.getState();
          
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
                sizeMinRatio,
                sizeMaxRatio,
                stopOnSizeMismatch,
                topK,
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
              topK
            );
          }

          // Check if this was a duplicate
          if (result.duplicate_skipped) {
            console.log(`${logPrefix} Duplicate detected for ann ${ann.id} (IoU=${result.duplicate_iou?.toFixed(3)}), skipping`);
            duplicateSkipCount++;
            continue;  // Don't add to propagationResults
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
          successCount++;
        } catch (err: any) {
          console.error(`${logPrefix} Propagation failed for annotation:`, ann.id, err);
          // Track this label as failed for fallback attempt
          failedLabels.add(ann.label_id);
        }
      }

      // Try fallback for failed labels (if we have a project)
      if (failedLabels.size > 0 && project) {
        console.log(`${logPrefix} 🔄 Attempting fallback for ${failedLabels.size} failed labels...`);
        setStatus(`Finding fallback references for ${failedLabels.size} labels...`);

        for (const labelId of failedLabels) {
          try {
            // Find a fallback reference annotation (approved annotation from earlier images)
            const fallbackResult = await api.findFallbackReference(
              labelId,
              currentIdx + 1, // beforeImageIndex - the target image index
              project.id
            );

            if (!fallbackResult.found || !fallbackResult.annotation) {
              console.log(`${logPrefix} No fallback found for label ${labelId}`);
              useUIStore.getState().addToast(`No reference found for label ${labelId}`, 'warning');
              continue;
            }

            console.log(`${logPrefix} Found fallback for label ${labelId} at image index ${fallbackResult.image_index}`);

            // Get settings again
            const { sizeMinRatio, sizeMaxRatio, stopOnSizeMismatch, topK, propagationMode, iouVerify, iouThreshold } = useUIStore.getState();

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
                sizeMinRatio,
                sizeMaxRatio,
                stopOnSizeMismatch,
                topK,
              }
            );

            if (result.duplicate_skipped) {
              console.log(`${logPrefix} Fallback duplicate for label ${labelId}, skipping`);
              duplicateSkipCount++;
              continue;
            }

            console.log(`${logPrefix} Fallback propagation success for label ${labelId}`);
            propagationResults.push({
              label_id: labelId,
              bbox: result.bbox,
              mask_rle: result.mask_rle,
              polygon: result.polygon,
            });
            fallbackCount++;
          } catch (err: any) {
            console.error(`${logPrefix} Fallback propagation failed for label ${labelId}:`, err);
            const errorMessage = err.response?.data?.detail || 'Fallback propagation failed';
            useUIStore.getState().addToast(`Fallback failed for label ${labelId}: ${errorMessage}`, 'error');
          }
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
      let message = `Tracked ${propagationResults.length}/${sourceAnnotations.length}`;
      if (fallbackCount > 0) {
        message += ` (${fallbackCount} via fallback)`;
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
      if (shouldAutoNext && prevIdx + 1 < currentImages.length - 1) {
        console.log(`${logPrefix} 🔁 Auto-next enabled, setting pending flag`);
        _pendingAutoNext = true;
      }

      useProjectStore.getState().nextImage();

      const newIdx = useProjectStore.getState().currentImageIndex;
      console.log(`${logPrefix} ✅ Navigation complete: currentIdx AFTER nextImage() = ${newIdx}`);
    } finally {
      console.log(`${logPrefix} 🔓 Releasing lock, isPropagating=false`);
      setIsPropagating(false);
      _propagationLock = false;
    }
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
            handlePropagateAndNext();
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
          // When enabling track mode, auto-load models if not loaded
          if (!trackEnabled && !propLoaded) {
            loadProp().then(() => setTrack(true));
          } else {
            setTrack(!trackEnabled);
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

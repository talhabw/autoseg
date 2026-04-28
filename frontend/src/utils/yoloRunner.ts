import * as api from '../api/client';
import { useAnnotationStore } from '../stores/annotationStore';
import { useProjectStore } from '../stores/projectStore';
import { useUIStore } from '../stores/uiStore';

let autoYoloRequestId = 0;

interface RunYoloOptions {
  showToast?: boolean;
  statusPrefix?: string;
}

export async function runYoloOnCurrentImage(options: RunYoloOptions = {}): Promise<boolean> {
  const currentImage = getCurrentImage();
  if (!currentImage) {
    useUIStore.getState().addToast('No image selected', 'warning');
    return false;
  }

  return runYoloForImage(currentImage.id, options);
}

export async function runYoloForImage(
  imageId: number,
  options: RunYoloOptions = {}
): Promise<boolean> {
  const { addToast, setStatusMessage, setIsRunningYolo } = useUIStore.getState();
  const yoloOptions = getYoloOptions();
  if (!yoloOptions) return false;

  const showToast = options.showToast ?? true;
  setIsRunningYolo(true);
  setStatusMessage(options.statusPrefix ?? 'Running YOLO on current image...');

  try {
    const summary = await api.runYoloOnImage(imageId, yoloOptions);

    await useAnnotationStore.getState().loadLabels();
    useAnnotationStore.getState().invalidateAnnotations(imageId);

    const currentImage = getCurrentImage();
    if (currentImage?.id === imageId) {
      await useAnnotationStore.getState().loadAnnotations(imageId);
    }

    setStatusMessage(`YOLO created ${summary.created}/${summary.detections} annotations`);
    if (showToast) {
      addToast(
        `YOLO created ${summary.created}/${summary.detections}${summary.skipped_duplicates > 0 ? ` (${summary.skipped_duplicates} duplicates skipped)` : ''}`,
        summary.created > 0 ? 'success' : 'info'
      );
    }
    return true;
  } catch (err) {
    console.error('YOLO annotation failed:', err);
    setStatusMessage('YOLO annotation failed');
    if (showToast) addToast('YOLO annotation failed', 'error');
    return false;
  } finally {
    setIsRunningYolo(false);
  }
}

export async function startAutoYolo(): Promise<void> {
  const projectState = useProjectStore.getState();
  if (!projectState.images[projectState.currentImageIndex]) {
    useUIStore.getState().addToast('No image selected', 'warning');
    return;
  }

  if (!getYoloOptions()) return;

  autoYoloRequestId += 1;
  const requestId = autoYoloRequestId;
  let imageIndex = projectState.currentImageIndex;

  useUIStore.getState().setAutoYoloEnabled(true);
  useUIStore.getState().addToast('Auto YOLO started. Press Esc to stop.', 'success', 2000);

  while (requestId === autoYoloRequestId) {
    const { images, setCurrentImageIndex } = useProjectStore.getState();
    if (imageIndex >= images.length) break;

    const image = images[imageIndex];
    setCurrentImageIndex(imageIndex);

    const ok = await runYoloForImage(image.id, {
      showToast: false,
      statusPrefix: `Auto YOLO ${imageIndex + 1}/${images.length}`,
    });

    if (!ok || requestId !== autoYoloRequestId) break;
    imageIndex += 1;
  }

  if (requestId === autoYoloRequestId) {
    useUIStore.getState().setAutoYoloEnabled(false);
    useUIStore.getState().setStatusMessage('Auto YOLO complete');
    useUIStore.getState().addToast('Auto YOLO complete', 'success', 3000);
  }
}

export function stopAutoYolo(): void {
  autoYoloRequestId += 1;
  useUIStore.getState().setAutoYoloEnabled(false);
  useUIStore.getState().setStatusMessage('Auto YOLO stopping after current image...');
  useUIStore.getState().addToast('Auto YOLO stopped', 'info', 1500);
}

function getYoloOptions(): api.YoloAnnotateOptions | null {
  const {
    yoloModelPath,
    yoloConfidence,
    yoloIou,
    yoloClassFilter,
    yoloUseSam,
    yoloUseYoloMasks,
    yoloMaxDetections,
    yoloDevice,
    addToast,
  } = useUIStore.getState();

  if (!yoloModelPath.trim()) {
    addToast('Set a YOLO model path in Settings first', 'warning');
    return null;
  }

  return {
    modelPath: yoloModelPath.trim(),
    confidence: yoloConfidence,
    iou: yoloIou,
    maxDetections: yoloMaxDetections,
    device: yoloDevice,
    classFilter: parseClassFilter(yoloClassFilter),
    useSam: yoloUseSam,
    useYoloMasks: yoloUseYoloMasks,
    status: 'pending',
    duplicateThreshold: 0.85,
    replaceExistingYolo: true,
  };
}

function getCurrentImage() {
  const { images, currentImageIndex } = useProjectStore.getState();
  return images[currentImageIndex] ?? null;
}

function parseClassFilter(value: string): string[] | null {
  const items = value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
  return items.length > 0 ? items : null;
}

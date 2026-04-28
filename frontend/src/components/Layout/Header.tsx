
import { useState } from 'react';
import { useUIStore } from '../../stores/uiStore';
import { useProjectStore } from '../../stores/projectStore';
import { useAnnotationStore } from '../../stores/annotationStore';
import { stopAutoTracking } from '../../App';
import * as api from '../../api/client';


import type { InteractionMode } from '../../types';
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Hand,
  Pencil,
  MousePointer2,
  Sparkles,
  Activity,
  Square,
  CheckCircle2,
  Settings,
  FilePlus,
  FolderOpen,
  Download,
  FastForward,
  Search,
} from 'lucide-react';

export function Header() {
  const {
    mode,
    setMode,
    trackModeEnabled,
    setTrackMode,
    reviewModeEnabled,
    setReviewMode,
    samLoaded,
    propagationLoaded,
    isLoadingModel,
    isPropagating,
    loadSAM,
    loadPropagation,
    setShowCreateProjectModal,
    setShowOpenProjectModal,
    setShowExportModal,
    setShowSettingsModal,
    setStatusMessage,
    addToast,
    autoNext,
    setAutoNext,
    yoloModelPath,
    yoloConfidence,
    yoloIou,
    yoloClassFilter,
    yoloUseSam,
    yoloUseYoloMasks,
    yoloMaxDetections,
    yoloDevice,
  } = useUIStore();

  const { project, images, currentImageIndex } = useProjectStore();
  const currentImage = images[currentImageIndex];
  const [isRunningYolo, setIsRunningYolo] = useState(false);

  // Trigger segmentation via keyboard event simulation
  const triggerSegmentation = () => {
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 's', bubbles: true }));
  };

  const runYoloOnCurrentImage = async () => {
    if (!currentImage) return;
    if (!yoloModelPath.trim()) {
      addToast('Set a YOLO model path in Settings first', 'warning');
      return;
    }

    setIsRunningYolo(true);
    setStatusMessage('Running YOLO on current image...');
    try {
      const summary = await api.runYoloOnImage(currentImage.id, {
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
      });

      await useAnnotationStore.getState().loadLabels();
      await useAnnotationStore.getState().loadAnnotations(currentImage.id);
      setStatusMessage(`YOLO created ${summary.created}/${summary.detections} annotations`);
      addToast(
        `YOLO created ${summary.created}/${summary.detections}${summary.skipped_duplicates > 0 ? ` (${summary.skipped_duplicates} duplicates skipped)` : ''}`,
        summary.created > 0 ? 'success' : 'info'
      );
    } catch (err) {
      console.error('YOLO annotation failed:', err);
      setStatusMessage('YOLO annotation failed');
      addToast('YOLO annotation failed', 'error');
    } finally {
      setIsRunningYolo(false);
    }
  };

  const tools: { mode: InteractionMode; label: string; shortcut: string; icon: React.ComponentType<{ className?: string }> }[] = [
    { mode: 'view', label: 'View', shortcut: 'V', icon: Hand },
    { mode: 'draw', label: 'Draw', shortcut: 'B', icon: Pencil },
    { mode: 'refine', label: 'Refine', shortcut: 'R', icon: MousePointer2 },
  ];

  return (
    <header className="h-14 bg-background border-b flex items-center px-4 gap-4 shadow-sm z-30 relative justify-between">
      {/* Group 1: File Actions (Simplified) */}
      <div className="flex items-center gap-1">
        <Button variant="ghost" size="icon" onClick={() => setShowCreateProjectModal(true)} title="New Project">
          <FilePlus className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" onClick={() => setShowOpenProjectModal(true)} title="Open Project">
          <FolderOpen className="h-4 w-4" />
        </Button>
        {project && (
          <Button variant="ghost" size="icon" onClick={() => setShowExportModal(true)} title="Export">
            <Download className="h-4 w-4" />
          </Button>
        )}

        {project && (
          <>
            <Separator orientation="vertical" className="h-6 mx-1" />
            <span className="text-xs text-muted-foreground font-mono truncate max-w-[150px]" title={project.name}>
              {project.name}
            </span>
          </>
        )}
      </div>

      {/* Group 2: Tools & Actions */}
      <div className="flex items-center gap-2">
        {/* Tools Group */}
        <div className="flex items-center bg-muted/50 rounded-lg p-1 gap-1 border">
          {tools.map((tool) => (
            <Button
              key={tool.mode}
              variant={mode === tool.mode ? "secondary" : "ghost"}
              size="sm"
              className={`h-8 px-2 gap-1 ${mode === tool.mode ? 'bg-background shadow-sm text-foreground' : 'text-muted-foreground'}`}
              onClick={() => setMode(tool.mode)}
              title={`${tool.label} (${tool.shortcut})`}
            >
              <tool.icon className="h-4 w-4" />
              <span className="text-[10px] font-mono opacity-60">{tool.shortcut}</span>
            </Button>
          ))}
        </div>

        <Separator orientation="vertical" className="h-6" />

        {/* Action: Segment */}
        <Button
          variant="default"
          size="sm"
          className="h-8 gap-2 bg-blue-600 hover:bg-blue-700 text-white"
          onClick={async () => {
            if (!samLoaded) {
              setStatusMessage('Loading SAM...');
              await loadSAM();
            }
            triggerSegmentation();
          }}
          disabled={isLoadingModel}
          title="Run Segmentation (S)"
        >
          {isLoadingModel ? (
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
          ) : (
            <Sparkles className="h-4 w-4" />
          )}
          <span className="hidden sm:inline">Segment</span>
          <span className="text-[10px] font-mono opacity-60 hidden sm:inline">(S)</span>
        </Button>

        {project && (
          <Button
            variant="outline"
            size="sm"
            className="h-8 gap-2 text-muted-foreground"
            onClick={runYoloOnCurrentImage}
            disabled={isRunningYolo || !currentImage}
            title="Run YOLO detection on current image"
          >
            {isRunningYolo ? (
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-muted-foreground/30 border-t-muted-foreground" />
            ) : (
              <Search className="h-4 w-4" />
            )}
            <span className="hidden sm:inline">YOLO</span>
          </Button>
        )}

        {/* Action: Track */}
        <Button
          variant={trackModeEnabled ? (isPropagating ? "default" : "secondary") : "outline"}
          size="sm"
          className={`h-8 gap-2 transition-all ${trackModeEnabled
            ? 'bg-emerald-600/20 text-emerald-500 border-emerald-600/50 hover:bg-emerald-600/30'
            : 'text-muted-foreground'}`}
          onClick={() => {
            if (!propagationLoaded && !trackModeEnabled) {
              loadPropagation().then(() => setTrackMode(true));
            } else {
              setTrackMode(!trackModeEnabled);
            }
          }}
          disabled={isLoadingModel || isPropagating}
          title="Toggle Track Mode (T)"
        >
          <Activity className={`h-4 w-4 ${isPropagating ? 'animate-pulse' : ''}`} />
          <span className="hidden sm:inline">{isPropagating ? 'Tracking' : 'Track'}</span>
          <span className="text-[10px] font-mono opacity-60 hidden sm:inline">(T)</span>
        </Button>

        {/* Action: Auto-next (only visible when track mode is enabled) */}
        {trackModeEnabled && (
          <Button
            variant={autoNext ? "secondary" : "ghost"}
            size="sm"
            className={`h-8 gap-2 transition-all ${autoNext
              ? 'bg-cyan-500/20 text-cyan-500 border-cyan-500/50 hover:bg-cyan-500/30'
              : 'text-muted-foreground'} ${isPropagating && autoNext ? 'animate-pulse' : ''}`}
            onClick={() => {
              if (autoNext) {
                // Stop auto-tracking - this now cancels in-progress propagation too
                stopAutoTracking();
                setAutoNext(false);
              } else {
                setAutoNext(true);
              }
            }}
            title={isPropagating && autoNext ? "Click to stop auto-tracking" : "Auto-advance after tracking (Shift+T)"}
          >
            {isPropagating && autoNext ? (
              <Square className="h-4 w-4" />
            ) : (
              <FastForward className="h-4 w-4" />
            )}
            <span className="hidden sm:inline">{isPropagating && autoNext ? 'Stop' : 'Auto'}</span>
            <span className="text-[10px] font-mono opacity-60 hidden sm:inline">(⇧T)</span>
          </Button>
        )}

        {/* Action: Review */}
        <Button
          variant={reviewModeEnabled ? "secondary" : "ghost"}
          size="sm"
          className={`h-8 gap-2 ${reviewModeEnabled ? 'bg-amber-500/20 text-amber-500 hover:bg-amber-500/30' : 'text-muted-foreground'}`}
          onClick={() => setReviewMode(!reviewModeEnabled)}
          title="Toggle Review Mode (Q)"
        >
          <CheckCircle2 className="h-4 w-4" />
          <span className="hidden sm:inline">Review</span>
          <span className="text-[10px] font-mono opacity-60 hidden sm:inline">(Q)</span>
        </Button>
      </div>

      {/* Group 3: Settings */}
      <div className="flex items-center gap-2">
        {/* Dev Reset (Hidden unless strictly needed, keeping generic dev check or just minimized) */}


        <Button variant="ghost" size="icon" onClick={() => setShowSettingsModal(true)} title="Settings">
          <Settings className="h-4 w-4" />
        </Button>
      </div>
    </header>
  );
}

function parseClassFilter(value: string): string[] | null {
  const items = value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
  return items.length > 0 ? items : null;
}

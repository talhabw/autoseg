
import { useUIStore } from '../../stores/uiStore';
import { useProjectStore } from '../../stores/projectStore';


import type { InteractionMode } from '../../types';
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Hand,
  Pencil,
  MousePointer2,
  Sparkles,
  Activity,
  CheckCircle2,
  Settings,
  FilePlus,
  FolderOpen,
  Download
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
  } = useUIStore();

  const { project } = useProjectStore();

  // Trigger segmentation via keyboard event simulation
  const triggerSegmentation = () => {
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 's', bubbles: true }));
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

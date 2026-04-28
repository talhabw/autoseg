import { useState, useEffect } from 'react';
import { useProjectStore } from '../../stores/projectStore';
import { useAnnotationStore } from '../../stores/annotationStore';
import { useUIStore } from '../../stores/uiStore';
import * as api from '../../api/client';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { FolderBrowser } from "../FolderBrowser/FolderBrowser";
import { FolderOpen, Wand2 } from "lucide-react";

export function CreateProjectModal() {
  const {
    showCreateProjectModal,
    setShowCreateProjectModal,
    setStatusMessage,
    addToast,
    yoloModelPath,
    setYoloModelPath,
    yoloConfidence,
    setYoloConfidence,
    yoloIou,
    yoloClassFilter,
    setYoloClassFilter,
    yoloUseSam,
    setYoloUseSam,
    yoloUseYoloMasks,
    yoloMaxDetections,
    yoloDevice,
  } = useUIStore();
  const { createProject } = useProjectStore();

  const [name, setName] = useState('');
  const [imageDir, setImageDir] = useState('');
  const [projectDir, setProjectDir] = useState('');
  const [projectDirManuallyEdited, setProjectDirManuallyEdited] = useState(false);
  const [runYoloPreprocess, setRunYoloPreprocess] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Auto-derive project name and folder from image directory
  useEffect(() => {
    if (!imageDir.trim()) return;

    // Clean the path
    const cleanPath = imageDir.replace(/\/+$/, ''); // Remove trailing slashes

    // Auto-suggest project name from folder name
    if (!name.trim()) {
      const folderName = cleanPath.split('/').pop() || '';
      if (folderName && folderName !== 'images') {
        setName(folderName.replace(/_/g, ' ').replace(/-/g, ' '));
      }
    }

    // Auto-suggest project directory if not manually edited
    if (!projectDirManuallyEdited) {
      // If path ends with /images, use sibling folder
      if (cleanPath.endsWith('/images')) {
        const basePath = cleanPath.slice(0, -7); // Remove '/images'
        setProjectDir(`${basePath}/autoseg_project`);
      } else {
        // Otherwise, create autoseg_project as sibling
        const parentPath = cleanPath.split('/').slice(0, -1).join('/');
        const folderName = cleanPath.split('/').pop() || 'project';
        setProjectDir(`${parentPath}/${folderName}_autoseg`);
      }
    }
  }, [imageDir, projectDirManuallyEdited, name]);

  // Reset form when modal closes
  useEffect(() => {
    if (!showCreateProjectModal) {
      setName('');
      setImageDir('');
      setProjectDir('');
      setProjectDirManuallyEdited(false);
      setRunYoloPreprocess(false);
      setError('');
    }
  }, [showCreateProjectModal]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!imageDir.trim() || !projectDir.trim()) {
      setError('Image directory and project directory are required');
      return;
    }

    // Auto-generate name if not provided
    const finalName = name.trim() || 'Untitled Project';

    if (runYoloPreprocess && !yoloModelPath.trim()) {
      setError('YOLO model path is required when preprocessing is enabled');
      return;
    }

    setIsLoading(true);
    try {
      await createProject(projectDir, imageDir, finalName);
      setStatusMessage(`Created project: ${finalName}`);
      setShowCreateProjectModal(false);

      if (runYoloPreprocess) {
        const options: api.YoloAnnotateOptions = {
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
          replaceExistingYolo: false,
        };
        void runYoloPreprocessForProject(finalName, options);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create project');
    } finally {
      setIsLoading(false);
    }
  };

  const runYoloPreprocessForProject = async (
    projectName: string,
    options: api.YoloAnnotateOptions
  ) => {
    setStatusMessage('Running YOLO preprocessing...');
    addToast('Project created. YOLO preprocessing is running in the background.', 'info', 4000);

    try {
      const summary = await api.runYoloOnProject(options);
      await useAnnotationStore.getState().loadLabels();

      const currentImage = useProjectStore.getState().currentImage;
      if (currentImage) {
        await useAnnotationStore.getState().loadAnnotations(currentImage.id);
      }

      setStatusMessage(`YOLO preprocessing complete: ${summary.created}/${summary.detections} annotations`);
      addToast(
        `${projectName}: YOLO created ${summary.created}/${summary.detections} annotations`,
        summary.created > 0 ? 'success' : 'info',
        6000
      );
    } catch (err) {
      console.error('YOLO preprocessing failed:', err);
      setStatusMessage('YOLO preprocessing failed');
      addToast('YOLO preprocessing failed', 'error', 8000);
    }
  };

  return (
    <Dialog open={showCreateProjectModal} onOpenChange={setShowCreateProjectModal}>
      <DialogContent className="sm:max-w-[620px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FolderOpen className="w-5 h-5" />
            Create New Project
          </DialogTitle>
          <DialogDescription>
            Point to your images folder and we'll set up the project automatically.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="imageDir">Image Directory</Label>
            <FolderBrowser
              value={imageDir}
              onChange={setImageDir}
              placeholder="/path/to/your/images"
            />
            <p className="text-xs text-muted-foreground">
              Folder containing images to annotate (jpg, png, etc.)
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="projectDir">Project Directory</Label>
              {projectDirManuallyEdited && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-6 text-xs gap-1"
                  onClick={() => {
                    setProjectDirManuallyEdited(false);
                    // Trigger re-derivation
                    setImageDir(imageDir + ' ');
                    setTimeout(() => setImageDir(imageDir.trim()), 0);
                  }}
                >
                  <Wand2 className="w-3 h-3" />
                  Auto
                </Button>
              )}
            </div>
            <Input
              id="projectDir"
              placeholder="/path/to/project"
              value={projectDir}
              onChange={(e) => {
                setProjectDir(e.target.value);
                setProjectDirManuallyEdited(true);
              }}
            />
            <p className="text-xs text-muted-foreground">
              Where annotations and project data will be saved
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="name">Project Name (optional)</Label>
            <Input
              id="name"
              placeholder="My Annotation Project"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <div className="space-y-4 rounded-lg border p-4 bg-muted/10">
            <div className="flex items-center gap-3">
              <Switch
                id="run-yolo-preprocess"
                checked={runYoloPreprocess}
                onCheckedChange={setRunYoloPreprocess}
              />
              <div className="space-y-1">
                <Label htmlFor="run-yolo-preprocess">Run YOLO preprocessing</Label>
                <p className="text-xs text-muted-foreground">
                  Detect objects with a YOLO model and optionally refine boxes with SAM3 before you start reviewing.
                </p>
              </div>
            </div>

            {runYoloPreprocess && (
              <div className="space-y-4 pt-2 border-t border-border/50">
                <div className="space-y-2">
                  <Label htmlFor="yoloModelPath">YOLO Model Path</Label>
                  <Input
                    id="yoloModelPath"
                    placeholder="/path/to/yolo11.pt"
                    value={yoloModelPath}
                    onChange={(e) => setYoloModelPath(e.target.value)}
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label>Confidence</Label>
                    <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded text-muted-foreground">
                      {(yoloConfidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <Slider
                    min={1}
                    max={100}
                    step={1}
                    value={[yoloConfidence * 100]}
                    onValueChange={(vals) => setYoloConfidence(vals[0] / 100)}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="yoloClasses">Classes (optional)</Label>
                  <Input
                    id="yoloClasses"
                    placeholder="torpedo_hole,0"
                    value={yoloClassFilter}
                    onChange={(e) => setYoloClassFilter(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Comma-separated YOLO class names or ids. Empty means all classes.
                  </p>
                </div>

                <div className="flex items-center justify-between rounded-md border p-3">
                  <div className="space-y-1">
                    <Label htmlFor="yoloUseSamCreate">Refine with SAM3</Label>
                    <p className="text-xs text-muted-foreground">
                      Recommended for YOLO bbox models.
                    </p>
                  </div>
                  <Switch
                    id="yoloUseSamCreate"
                    checked={yoloUseSam}
                    onCheckedChange={setYoloUseSam}
                  />
                </div>
              </div>
            )}
          </div>

          {error && (
            <div className="p-3 bg-red-500/10 border border-red-500/50 rounded-lg text-sm text-red-600 dark:text-red-400">
              {error}
            </div>
          )}

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => setShowCreateProjectModal(false)}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={isLoading || !imageDir.trim() || !projectDir.trim()}
            >
              {isLoading ? 'Creating...' : 'Create Project'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function parseClassFilter(value: string): string[] | null {
  const items = value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
  return items.length > 0 ? items : null;
}

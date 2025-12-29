import { useState, useEffect } from 'react';
import { useProjectStore } from '../../stores/projectStore';
import { useUIStore } from '../../stores/uiStore';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { FolderBrowser } from "../FolderBrowser/FolderBrowser";
import { FolderOpen } from "lucide-react";

export function OpenProjectModal() {
  const { showOpenProjectModal, setShowOpenProjectModal, setStatusMessage } = useUIStore();
  const { openProject } = useProjectStore();

  const [projectDir, setProjectDir] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Reset form when modal closes
  useEffect(() => {
    if (!showOpenProjectModal) {
      setProjectDir('');
      setError('');
    }
  }, [showOpenProjectModal]);

  // Try to auto-detect project folder from images folder path
  const getProjectDir = (path: string): string => {
    const cleanPath = path.replace(/\/+$/, '');

    // If it ends with /images, look for sibling autoseg_project
    if (cleanPath.endsWith('/images')) {
      const basePath = cleanPath.slice(0, -7);
      return `${basePath}/autoseg_project`;
    }

    // If it ends with _autoseg, use as is
    if (cleanPath.endsWith('_autoseg') || cleanPath.endsWith('autoseg_project')) {
      return cleanPath;
    }

    return cleanPath;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!projectDir.trim()) {
      setError('Project directory is required');
      return;
    }

    const resolvedDir = getProjectDir(projectDir.trim());

    setIsLoading(true);
    try {
      await openProject(resolvedDir);
      setStatusMessage(`Opened project`);
      setShowOpenProjectModal(false);
    } catch (err) {
      // If failed with auto-resolved path, try original path
      if (resolvedDir !== projectDir.trim()) {
        try {
          await openProject(projectDir.trim());
          setStatusMessage(`Opened project`);
          setShowOpenProjectModal(false);
          return;
        } catch {
          // Fall through to show first error
        }
      }
      setError(err instanceof Error ? err.message : 'Failed to open project');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={showOpenProjectModal} onOpenChange={setShowOpenProjectModal}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FolderOpen className="w-5 h-5" />
            Open Project
          </DialogTitle>
          <DialogDescription>
            Enter the path to an existing AutoSeg project folder.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="projectDir">Project Directory</Label>
            <FolderBrowser
              value={projectDir}
              onChange={setProjectDir}
              placeholder="/path/to/project or /path/to/images"
            />
            <p className="text-xs text-muted-foreground">
              The folder containing autoseg.db (or your images folder if project is alongside)
            </p>
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
              onClick={() => setShowOpenProjectModal(false)}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={isLoading || !projectDir.trim()}
            >
              {isLoading ? 'Opening...' : 'Open Project'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { FolderBrowser } from "../FolderBrowser/FolderBrowser";
import { FolderOpen, Wand2 } from "lucide-react";

export function CreateProjectModal() {
  const { showCreateProjectModal, setShowCreateProjectModal, setStatusMessage } = useUIStore();
  const { createProject } = useProjectStore();

  const [name, setName] = useState('');
  const [imageDir, setImageDir] = useState('');
  const [projectDir, setProjectDir] = useState('');
  const [projectDirManuallyEdited, setProjectDirManuallyEdited] = useState(false);
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

    setIsLoading(true);
    try {
      await createProject(projectDir, imageDir, finalName);
      setStatusMessage(`Created project: ${finalName}`);
      setShowCreateProjectModal(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create project');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={showCreateProjectModal} onOpenChange={setShowCreateProjectModal}>
      <DialogContent className="sm:max-w-[550px]">
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

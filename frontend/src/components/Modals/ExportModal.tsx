import { useState } from 'react';
import { useProjectStore } from '../../stores/projectStore';
import { useUIStore } from '../../stores/uiStore';
import { exportYolo, validateProject, type ValidateResponse } from '../../api/client';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";

// Actually, I didn't install Checkbox. I'll use a standard input with Label for now or install it. 
// Ideally I should install it, but to keep it simple and fast I'll use native input wrapped in Label as before but styled better, OR add checkbox installation.
// Let's stick to standard HTML checkbox properly styled or install standard `checkbox` component. 
// I'll assume standard HTML checkbox for now to avoid breaking flow, but styled.
// Wait, I can install check box. `npx shadcn@latest add checkbox`
// I'll auto-run that in parallel or sequence.

export function ExportModal() {
  const { showExportModal, setShowExportModal, setStatusMessage } = useUIStore();
  const { project } = useProjectStore();

  const [outputDir, setOutputDir] = useState('');
  const [trainSplit, setTrainSplit] = useState('0.8');
  const [seed, setSeed] = useState('42');
  const [approvedOnly, setApprovedOnly] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [error, setError] = useState('');
  const [validation, setValidation] = useState<ValidateResponse | null>(null);
  const [result, setResult] = useState<{
    train_images: number;
    val_images: number;
    total_annotations: number;
    warnings: string[];
    is_valid: boolean;
  } | null>(null);

  if (!project) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setResult(null);

    if (!outputDir.trim()) {
      setError('Output directory is required');
      return;
    }

    const split = parseFloat(trainSplit);
    if (isNaN(split) || split <= 0 || split >= 1) {
      setError('Train split must be between 0 and 1');
      return;
    }

    setIsLoading(true);
    try {
      const res = await exportYolo({
        output_dir: outputDir,
        train_split: split,
        seed: parseInt(seed) || 42,
        approved_only: approvedOnly,
      });
      setResult(res);
      setStatusMessage(`Exported to ${outputDir}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={showExportModal} onOpenChange={setShowExportModal}>
      <DialogContent className="sm:max-w-[600px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Export YOLO-seg</DialogTitle>
        </DialogHeader>

        {result ? (
          <div className="space-y-6">
            <div className={`p-4 rounded-lg border ${result.is_valid ? 'bg-green-500/10 border-green-500/20 text-green-600' : 'bg-yellow-500/10 border-yellow-500/20 text-yellow-600'}`}>
              <h3 className="font-semibold flex items-center gap-2">
                {result.is_valid ? '✓ Export Successful!' : '⚠ Export Complete with Warnings'}
              </h3>
            </div>

            <div className="grid grid-cols-3 gap-4 text-center">
              <Card className="p-4">
                <div className="text-2xl font-bold">{result.train_images}</div>
                <div className="text-xs text-muted-foreground">Train Images</div>
              </Card>
              <Card className="p-4">
                <div className="text-2xl font-bold">{result.val_images}</div>
                <div className="text-xs text-muted-foreground">Val Images</div>
              </Card>
              <Card className="p-4">
                <div className="text-2xl font-bold">{result.total_annotations}</div>
                <div className="text-xs text-muted-foreground">Annotations</div>
              </Card>
            </div>

            {result.warnings.length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-semibold mb-2">Warnings:</h4>
                <ul className="text-sm text-muted-foreground space-y-1 list-disc pl-4">
                  {result.warnings.slice(0, 5).map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                  {result.warnings.length > 5 && (
                    <li>... and {result.warnings.length - 5} more</li>
                  )}
                </ul>
              </div>
            )}

            <div className="flex justify-end pt-2">
              <Button
                onClick={() => {
                  setResult(null);
                  setShowExportModal(false);
                }}
              >
                Close
              </Button>
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="outputDir">Output Directory (absolute path)</Label>
              <Input
                id="outputDir"
                placeholder={`${project.root_dir}/export_yolo`}
                value={outputDir}
                onChange={(e) => setOutputDir(e.target.value)}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="trainSplit">Train Split</Label>
                <Input
                  id="trainSplit"
                  value={trainSplit}
                  onChange={(e) => setTrainSplit(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="seed">Random Seed</Label>
                <Input
                  id="seed"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                />
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="approvedOnly"
                checked={approvedOnly}
                onChange={(e) => setApprovedOnly(e.target.checked)}
                className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
              />
              <Label htmlFor="approvedOnly" className="cursor-pointer">
                Export approved annotations only
              </Label>
            </div>

            {error && (
              <div className="p-3 bg-red-500/10 border border-red-500/50 rounded-lg text-sm text-red-600 dark:text-red-400">
                {error}
              </div>
            )}

            {/* Validation Results */}
            {validation && (
              <div className={`p-3 rounded-lg border ${validation.is_valid ? 'bg-green-500/10 border-green-500/20' : 'bg-yellow-500/10 border-yellow-500/20'}`}>
                <div className="flex items-center gap-2 mb-2">
                  {validation.is_valid ? (
                    <span className="text-green-600 font-medium">✓ Validation Passed</span>
                  ) : (
                    <span className="text-yellow-600 font-medium">⚠ Validation Issues Found</span>
                  )}
                </div>
                <div className="text-xs text-muted-foreground space-y-1">
                  <div>{validation.total_images} images, {validation.total_annotations} annotations</div>
                  {validation.error_count > 0 && (
                    <div className="text-red-500">{validation.error_count} errors</div>
                  )}
                  {validation.warning_count > 0 && (
                    <div className="text-yellow-500">{validation.warning_count} warnings</div>
                  )}
                  {validation.errors.length > 0 && (
                    <ul className="mt-2 text-red-500 list-disc pl-4">
                      {validation.errors.slice(0, 3).map((e, i) => (
                        <li key={i}>{e.message} (ann #{e.annotation_id})</li>
                      ))}
                      {validation.errors.length > 3 && (
                        <li>... and {validation.errors.length - 3} more errors</li>
                      )}
                    </ul>
                  )}
                </div>
              </div>
            )}

            <div className="flex gap-2 justify-end pt-2">
              <Button
                type="button"
                variant="secondary"
                onClick={async () => {
                  setIsValidating(true);
                  setError('');
                  try {
                    const res = await validateProject();
                    setValidation(res);
                  } catch (err) {
                    setError(err instanceof Error ? err.message : 'Validation failed');
                  } finally {
                    setIsValidating(false);
                  }
                }}
                disabled={isValidating || isLoading}
              >
                {isValidating ? 'Validating...' : 'Validate'}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => setShowExportModal(false)}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={isLoading}
              >
                {isLoading ? 'Exporting...' : 'Export'}
              </Button>
            </div>
          </form>
        )}
      </DialogContent>
    </Dialog>
  );
}

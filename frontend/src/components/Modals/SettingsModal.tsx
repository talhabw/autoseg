import { useUIStore, type EmbedModel } from '../../stores/uiStore';
import { useProjectStore } from '../../stores/projectStore';
import { useAnnotationStore } from '../../stores/annotationStore';
import { useEffect, useState } from 'react';
import * as api from '../../api/client';
import type { PropagationMode } from '../../types';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { ListOrdered, RefreshCw, Trash2 } from 'lucide-react';

export function SettingsModal() {
  const {
    showSettingsModal,
    setShowSettingsModal,
    sizeMinRatio,
    setSizeMinRatio,
    sizeMaxRatio,
    setSizeMaxRatio,
    maskOpacity,
    setMaskOpacity,
    embedModel,
    setEmbedModel,
    stopOnSizeMismatch,
    setStopOnSizeMismatch,
    topK,
    setTopK,
    useBBoxHint,
    setUseBBoxHint,
    bboxHintScale,
    setBboxHintScale,
    pruneThinArtifacts,
    setPruneThinArtifacts,
    propagationMode,
    setPropagationMode,
    iouVerify,
    setIouVerify,
    iouThreshold,
    setIouThreshold,
    trackingDuplicateThreshold,
    setTrackingDuplicateThreshold,
    propagationFailureMode,
    setPropagationFailureMode,
    samMaskThreshold,
    setSamMaskThreshold,
    samMinRegionArea,
    setSamMinRegionArea,
    samKeepLargestRegion,
    setSamKeepLargestRegion,
    samLoaded,
    performanceMode,
    setPerformanceMode,
    yoloModelPath,
    setYoloModelPath,
    yoloConfidence,
    setYoloConfidence,
    yoloIou,
    setYoloIou,
    yoloClassFilter,
    setYoloClassFilter,
    yoloUseSam,
    setYoloUseSam,
    yoloUseYoloMasks,
    setYoloUseYoloMasks,
    yoloMaxDetections,
    setYoloMaxDetections,
    yoloDevice,
    setYoloDevice,
    addToast,
  } = useUIStore();

  const { resyncImages, project, images } = useProjectStore();
  const { labels, loadLabels } = useAnnotationStore();

  const [availableModels, setAvailableModels] = useState<{ id: string; name: string; available: boolean }[]>([]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [bulkDeleteAfter, setBulkDeleteAfter] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);
  const [showLabelMap, setShowLabelMap] = useState(false);

  useEffect(() => {
    if (showSettingsModal) {
      api.getAvailableEmbedModels().then((data) => {
        setAvailableModels(data.models);
      }).catch(console.error);
    }
  }, [showSettingsModal]);

  useEffect(() => {
    if (showSettingsModal && project) {
      void loadLabels();
    }
  }, [showSettingsModal, project, loadLabels]);

  const handleRefreshImages = async () => {
    setIsRefreshing(true);
    try {
      const result = await resyncImages();
      if (result.added > 0 || result.removed > 0) {
        addToast(`Updated image list: +${result.added} new, -${result.removed} removed`, 'success');
      } else {
        addToast('Image list is up to date', 'info');
      }
    } catch (err) {
      addToast('Failed to refresh images', 'error');
      console.error('Failed to refresh images:', err);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleBulkDelete = async () => {
    const afterIndex = parseInt(bulkDeleteAfter, 10);
    if (isNaN(afterIndex) || afterIndex < 0) {
      addToast('Please enter a valid image number', 'error');
      return;
    }
    if (!project) return;

    const imagesToAffect = images.length - afterIndex - 1;
    if (imagesToAffect <= 0) {
      addToast('No images after that index', 'info');
      return;
    }

    const confirmed = window.confirm(
      `This will delete ALL annotations from images after #${afterIndex + 1} (${imagesToAffect} images). This cannot be undone. Continue?`
    );
    if (!confirmed) return;

    setIsDeleting(true);
    try {
      const result = await api.deleteAnnotationsAfterIndex(project.id, afterIndex);
      addToast(`Deleted ${result.count} annotations`, 'success');
      setBulkDeleteAfter('');
    } catch (err) {
      addToast('Failed to delete annotations', 'error');
      console.error('Bulk delete failed:', err);
    } finally {
      setIsDeleting(false);
    }
  };

  // However, since we're using the store to toggle visibility, we can just pass `open={showSettingsModal}` and `onOpenChange={setShowSettingsModal}`
  // if (!showSettingsModal) return null; // Logic handled by Dialog open prop

  return (
    <>
    <Dialog open={showSettingsModal} onOpenChange={setShowSettingsModal}>
      <DialogContent className="sm:max-w-[550px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
        </DialogHeader>

        <div className="space-y-8 py-4">

          {/* Project Settings */}
          {project && (
            <div className="space-y-4">
              <h3 className="text-sm font-medium leading-none flex items-center gap-2 text-muted-foreground">
                Project
                <Separator className="flex-1" />
              </h3>

              <div className="space-y-3 pl-2">
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <Label>Refresh Image List</Label>
                    <p className="text-xs text-muted-foreground">
                      Scan the image directory for new or removed files.
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleRefreshImages}
                    disabled={isRefreshing}
                  >
                    <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
                    {isRefreshing ? 'Scanning...' : 'Refresh'}
                  </Button>
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-border/50">
                  <div className="space-y-1">
                    <Label>Export Label IDs</Label>
                    <p className="text-xs text-muted-foreground">
                      View the class IDs that YOLO/COCO exports will write for this project.
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowLabelMap(true)}
                    disabled={labels.length === 0}
                  >
                    <ListOrdered className="h-4 w-4 mr-2" />
                    View IDs
                  </Button>
                </div>

                <div className="space-y-2 pt-2 border-t border-border/50">
                  <div className="space-y-1">
                    <Label>Delete Annotations After Image</Label>
                    <p className="text-xs text-muted-foreground">
                      Remove all annotations from images after a specific index. Enter the last image number to keep (1-based).
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Input
                      type="number"
                      min={1}
                      max={images.length}
                      placeholder={`1-${images.length}`}
                      value={bulkDeleteAfter}
                      onChange={(e) => setBulkDeleteAfter(e.target.value)}
                      className="w-28"
                    />
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={handleBulkDelete}
                      disabled={isDeleting || !bulkDeleteAfter}
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      {isDeleting ? 'Deleting...' : 'Delete'}
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Display Settings */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium leading-none flex items-center gap-2 text-muted-foreground">
              Display
              <Separator className="flex-1" />
            </h3>

            <div className="space-y-3 pl-2">
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label>Mask Opacity</Label>
                  <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded text-muted-foreground">
                    {(maskOpacity * 100).toFixed(0)}%
                  </span>
                </div>
                <Slider
                  min={0}
                  max={100}
                  step={1}
                  value={[maskOpacity * 100]}
                  onValueChange={(vals) => setMaskOpacity(vals[0] / 100)}
                />
              </div>
            </div>
          </div>

          {/* Model Settings */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium leading-none flex items-center gap-2 text-muted-foreground">
              Models
              <Separator className="flex-1" />
            </h3>

            <div className="space-y-3 pl-2">
              <div className="space-y-2">
                <Label>Embedding Model</Label>
                <Select
                  value={embedModel}
                  onValueChange={(val) => setEmbedModel(val as EmbedModel)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableModels.length > 0 ? (
                      availableModels.map((model) => (
                        <SelectItem
                          key={model.id}
                          value={model.id}
                          disabled={!model.available}
                          className={!model.available ? "opacity-50" : ""}
                        >
                          {model.available ? "" : "[MISSING] "} {model.name}
                        </SelectItem>
                      ))
                    ) : (
                      // Fallback if API fails or loading
                      <>
                        <SelectItem value="vitb16">DINOv3 ViT-B/16</SelectItem>
                        <SelectItem value="vitl16">DINOv3 ViT-L/16</SelectItem>
                        <SelectItem value="vith16">DINOv3 ViT-H/16</SelectItem>
                      </>
                    )}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Larger models provide better segmentation accuracy but require more GPU memory and inference time.
                </p>
              </div>

              <div className="space-y-4 pt-4 border-t border-border/50">
                <div className="space-y-2">
                  <Label>YOLO Model Path</Label>
                  <Input
                    placeholder="/path/to/yolo11.pt"
                    value={yoloModelPath}
                    onChange={(e) => setYoloModelPath(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Used for create-time preprocessing and the YOLO button on the current image. Supports bbox and segmentation YOLO models.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label>YOLO Confidence</Label>
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
                    <div className="flex justify-between items-center">
                      <Label>YOLO IoU</Label>
                      <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded text-muted-foreground">
                        {(yoloIou * 100).toFixed(0)}%
                      </span>
                    </div>
                    <Slider
                      min={1}
                      max={100}
                      step={1}
                      value={[yoloIou * 100]}
                      onValueChange={(vals) => setYoloIou(vals[0] / 100)}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>YOLO Classes</Label>
                    <Input
                      placeholder="optional: hole,0,torpedo"
                      value={yoloClassFilter}
                      onChange={(e) => setYoloClassFilter(e.target.value)}
                    />
                    <p className="text-xs text-muted-foreground">
                      Comma-separated class names or ids. Leave empty to use all detected classes.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label>Max Detections</Label>
                    <Input
                      type="number"
                      min={1}
                      max={1000}
                      value={yoloMaxDetections}
                      onChange={(e) => setYoloMaxDetections(Number(e.target.value) || 1)}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>YOLO Device</Label>
                  <Select value={yoloDevice} onValueChange={setYoloDevice}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select device" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="cuda">CUDA</SelectItem>
                      <SelectItem value="cpu">CPU</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center space-x-4 rounded-lg border p-4">
                  <Switch
                    id="yolo-use-sam"
                    checked={yoloUseSam}
                    onCheckedChange={setYoloUseSam}
                  />
                  <div className="flex-1 space-y-1">
                    <Label htmlFor="yolo-use-sam">Refine YOLO Boxes With SAM</Label>
                    <p className="text-xs text-muted-foreground">
                      Recommended for bbox models. YOLO creates boxes, then SAM3 turns them into masks.
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-4 rounded-lg border p-4">
                  <Switch
                    id="yolo-use-masks"
                    checked={yoloUseYoloMasks}
                    onCheckedChange={setYoloUseYoloMasks}
                  />
                  <div className="flex-1 space-y-1">
                    <Label htmlFor="yolo-use-masks">Use YOLO Seg Masks When Available</Label>
                    <p className="text-xs text-muted-foreground">
                      For YOLO segmentation models, masks are used when SAM is off or if SAM refinement fails.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* SAM Settings */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium leading-none flex items-center gap-2 text-muted-foreground">
              Segmentation (SAM)
              <Separator className="flex-1" />
            </h3>

            <div className="space-y-6 pl-2">
              {!samLoaded && (
                <p className="text-xs text-amber-500">
                  Load the SAM model first to adjust these settings.
                </p>
              )}

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label>Mask Threshold</Label>
                  <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded text-muted-foreground">
                    {samMaskThreshold.toFixed(2)}
                  </span>
                </div>
                <Slider
                  min={-200}
                  max={200}
                  step={5}
                  value={[samMaskThreshold * 100]}
                  onValueChange={(vals) => setSamMaskThreshold(vals[0] / 100)}
                  disabled={!samLoaded}
                />
                <p className="text-xs text-muted-foreground">
                  Controls mask boundary sensitivity. Lower values = larger masks, higher values = smaller/tighter masks.
                  Range: -2.0 to 2.0. Default: 0.0
                </p>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label>Min Region Size</Label>
                  <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded text-muted-foreground">
                    {samMinRegionArea} px
                  </span>
                </div>
                <Slider
                  min={0}
                  max={2000}
                  step={50}
                  value={[samMinRegionArea]}
                  onValueChange={(vals) => setSamMinRegionArea(vals[0])}
                  disabled={!samLoaded}
                />
                <p className="text-xs text-muted-foreground">
                  Remove disconnected regions smaller than this size. Helps eliminate noise and small artifacts.
                  Set to 0 to keep all regions. Default: 100
                </p>
              </div>

              <div className="flex items-center space-x-4 rounded-lg border p-4">
                <Switch
                  id="keep-largest-region"
                  checked={samKeepLargestRegion}
                  onCheckedChange={setSamKeepLargestRegion}
                  disabled={!samLoaded}
                />
                <div className="flex-1 space-y-1">
                  <Label htmlFor="keep-largest-region">Keep Only Largest Region</Label>
                  <p className="text-xs text-muted-foreground">
                    When enabled, ONLY the largest connected region is kept - all other disconnected parts are removed.
                    When disabled, all regions larger than Min Region Size are kept.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Propagation Settings */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium leading-none flex items-center gap-2 text-muted-foreground">
              Propagation / Tracking
              <Separator className="flex-1" />
            </h3>

            <div className="space-y-6 pl-2">
              <div className="space-y-2">
                <Label>Propagation Mode</Label>
                <Select
                  value={propagationMode}
                  onValueChange={(val) => setPropagationMode(val as PropagationMode)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto (Recommended)</SelectItem>
                    <SelectItem value="peak">Peak-based</SelectItem>
                    <SelectItem value="dense">Dense correspondence</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Auto tries peak-based first, falls back to dense if needed. Dense uses legacy-style patch correspondence.
                </p>
              </div>

              <div className="flex items-center space-x-4 rounded-lg border p-4">
                <Switch
                  id="iou-verify"
                  checked={iouVerify}
                  onCheckedChange={setIouVerify}
                />
                <div className="flex-1 space-y-1">
                  <Label htmlFor="iou-verify">IoU Verification</Label>
                  <p className="text-xs text-muted-foreground">
                    Verify propagation results against dense prediction. Rejects results below threshold.
                  </p>
                </div>
              </div>

              {iouVerify && (
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label>IoU Threshold</Label>
                    <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded text-muted-foreground">
                      {(iouThreshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  <Slider
                    min={0}
                    max={100}
                    step={5}
                    value={[iouThreshold * 100]}
                    onValueChange={(vals) => setIouThreshold(vals[0] / 100)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Minimum IoU between SAM result and dense prediction to accept.
                  </p>
                </div>
              )}

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label>Duplicate Threshold</Label>
                  <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded text-muted-foreground">
                    {(trackingDuplicateThreshold * 100).toFixed(0)}%
                  </span>
                </div>
                <Slider
                  min={30}
                  max={100}
                  step={5}
                  value={[trackingDuplicateThreshold * 100]}
                  onValueChange={(vals) => setTrackingDuplicateThreshold(vals[0] / 100)}
                />
                <p className="text-xs text-muted-foreground">
                  Skip a tracked result when it overlaps an existing same-label annotation by this much. Lower values remove more duplicates; higher values are more permissive.
                </p>
              </div>

              <div className="space-y-2">
                <Label>On Propagation Failure</Label>
                <Select
                  value={propagationFailureMode}
                  onValueChange={(val) => setPropagationFailureMode(val as 'stop' | 'skip')}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select behavior" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="stop">Stop and notify</SelectItem>
                    <SelectItem value="skip">Skip and continue</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Stop: Stops tracking and shows the failed image. Skip: Continues to next image, excluding failed ones from future references.
                </p>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label>Peak Candidates (Top-K)</Label>
                  <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded text-muted-foreground">
                    {topK}
                  </span>
                </div>
                <Slider
                  min={1}
                  max={10}
                  step={1}
                  value={[topK]}
                  onValueChange={(vals) => setTopK(vals[0])}
                />
                <p className="text-xs text-muted-foreground">
                  Number of peak locations to try when matching. Higher = more thorough but slower.
                </p>
              </div>

              <div className="flex items-center space-x-4 rounded-lg border p-4">
                <Switch
                  id="use-bbox-hint"
                  checked={useBBoxHint}
                  onCheckedChange={setUseBBoxHint}
                />
                <div className="flex-1 space-y-1">
                  <Label htmlFor="use-bbox-hint">Use BBox Hints For Tracking</Label>
                  <p className="text-xs text-muted-foreground">
                    Constrains tracked SAM prompts with a translated bbox hint. Turn this off to use free point-only prompting instead.
                  </p>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label>BBox Hint Scale</Label>
                  <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded text-muted-foreground">
                    {useBBoxHint ? `${bboxHintScale.toFixed(2)}x` : 'Off'}
                  </span>
                </div>
                <Slider
                  min={50}
                  max={300}
                  step={5}
                  value={[bboxHintScale * 100]}
                  disabled={!useBBoxHint}
                  onValueChange={(vals) => setBboxHintScale(vals[0] / 100)}
                />
                <p className="text-xs text-muted-foreground">
                  Controls how much the SAM tracking hint box is padded when bbox hints are enabled. Lower = tighter and more restrictive. Higher = looser and more rotation-tolerant.
                </p>
              </div>

              <div className="flex items-center space-x-4 rounded-lg border p-4">
                <Switch
                  id="prune-thin-artifacts"
                  checked={pruneThinArtifacts}
                  onCheckedChange={setPruneThinArtifacts}
                />
                <div className="flex-1 space-y-1">
                  <Label htmlFor="prune-thin-artifacts">Prune Thin Tracking Artifacts</Label>
                  <p className="text-xs text-muted-foreground">
                    Removes small thin branches that SAM sometimes adds to tracked masks before the result is shown and saved.
                  </p>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label>Min Object Size</Label>
                  <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded text-muted-foreground">
                    {sizeMinRatio.toFixed(2)}x
                  </span>
                </div>
                <Slider
                  min={10}
                  max={100}
                  step={1}
                  value={[sizeMinRatio * 100]}
                  onValueChange={(vals) => setSizeMinRatio(vals[0] / 100)}
                />
                <p className="text-xs text-muted-foreground">
                  Minimum tracked object size relative to initial frame.
                </p>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label>Max Object Size</Label>
                  <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded text-muted-foreground">
                    {sizeMaxRatio.toFixed(2)}x
                  </span>
                </div>
                <Slider
                  min={100}
                  max={300}
                  step={1}
                  value={[sizeMaxRatio * 100]}
                  onValueChange={(vals) => setSizeMaxRatio(vals[0] / 100)}
                />
                <p className="text-xs text-muted-foreground">
                  Maximum tracked object size relative to initial frame.
                </p>
              </div>

              <div className="flex items-center space-x-4 rounded-lg border p-4">
                <Switch
                  id="stop-tracking"
                  checked={stopOnSizeMismatch}
                  onCheckedChange={setStopOnSizeMismatch}
                />
                <div className="flex-1 space-y-1">
                  <Label htmlFor="stop-tracking">Stop tracking on size mismatch</Label>
                  <p className="text-xs text-muted-foreground">
                    If enabled, tracking halts if the object size deviates beyond the min/max thresholds. Safer but less robust to occlusion.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Performance Settings */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium leading-none flex items-center gap-2 text-muted-foreground">
              Performance
              <Separator className="flex-1" />
            </h3>

            <div className="space-y-6 pl-2">
              <div className="flex items-center space-x-4 rounded-lg border p-4">
                <Switch
                  id="performance-mode"
                  checked={performanceMode}
                  onCheckedChange={setPerformanceMode}
                />
                <div className="flex-1 space-y-1">
                  <Label htmlFor="performance-mode">Performance Propagation Mode</Label>
                  <p className="text-xs text-muted-foreground">
                    When enabled, auto-propagation skips UI rendering for each frame and runs a tight API-only loop. 
                    The canvas only refreshes when propagation fails and requires your input. 
                    Dramatically faster for large datasets.
                  </p>
                </div>
              </div>
            </div>
          </div>

        </div>

        <div className="flex justify-end pt-2">
          <Button onClick={() => setShowSettingsModal(false)}>
            Done
          </Button>
        </div>
      </DialogContent>
    </Dialog>
    <Dialog open={showLabelMap} onOpenChange={setShowLabelMap}>
      <DialogContent className="sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle>Export Label IDs</DialogTitle>
          <DialogDescription>
            Export IDs are assigned from the project label list sorted by name, matching the current export code.
          </DialogDescription>
        </DialogHeader>

        {labels.length > 0 ? (
          <div className="max-h-[60vh] overflow-y-auto rounded-md border">
            <div className="grid grid-cols-[80px_80px_90px_1fr] gap-3 border-b bg-muted/40 px-3 py-2 text-xs font-medium text-muted-foreground">
              <span>YOLO ID</span>
              <span>COCO ID</span>
              <span>Label ID</span>
              <span>Name</span>
            </div>
            {labels.map((label, index) => (
              <div
                key={label.id}
                className="grid grid-cols-[80px_80px_90px_1fr] items-center gap-3 border-b px-3 py-2 text-sm last:border-b-0"
              >
                <span className="font-mono text-muted-foreground">{index}</span>
                <span className="font-mono text-muted-foreground">{index + 1}</span>
                <span className="font-mono text-muted-foreground">{label.id}</span>
                <span className="flex min-w-0 items-center gap-2">
                  <span
                    className="h-3 w-3 shrink-0 rounded-full border"
                    style={{ backgroundColor: label.color }}
                  />
                  <span className="truncate">{label.name}</span>
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="rounded-md border p-4 text-sm text-muted-foreground">
            No labels have been created for this project yet.
          </p>
        )}

        <p className="text-xs text-muted-foreground">
          YOLO segmentation/detection label files use the zero-based YOLO ID. COCO export category IDs use the one-based COCO ID.
        </p>
      </DialogContent>
    </Dialog>
    </>
  );
}

import { useUIStore, type EmbedModel } from '../../stores/uiStore';
import { useEffect, useState } from 'react';
import * as api from '../../api/client';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
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
  } = useUIStore();

  const [availableModels, setAvailableModels] = useState<{ id: string; name: string; available: boolean }[]>([]);

  useEffect(() => {
    if (showSettingsModal) {
      api.getAvailableEmbedModels().then((data) => {
        setAvailableModels(data.models);
      }).catch(console.error);
    }
  }, [showSettingsModal]);

  // However, since we're using the store to toggle visibility, we can just pass `open={showSettingsModal}` and `onOpenChange={setShowSettingsModal}`
  // if (!showSettingsModal) return null; // Logic handled by Dialog open prop

  return (
    <Dialog open={showSettingsModal} onOpenChange={setShowSettingsModal}>
      <DialogContent className="sm:max-w-[550px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
        </DialogHeader>

        <div className="space-y-8 py-4">

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

        </div>

        <div className="flex justify-end pt-2">
          <Button onClick={() => setShowSettingsModal(false)}>
            Done
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

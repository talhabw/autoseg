import { useAnnotationStore } from '../../stores/annotationStore';
import { useProjectStore } from '../../stores/projectStore';
import { useUIStore, type ReviewFilter } from '../../stores/uiStore';
import * as api from '../../api/client';
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { Check, X, Trash2, Tag } from "lucide-react";

export function AnnotationList() {
  const {
    annotations,
    labels,
    selectedAnnotationId,
    selectAnnotation,
    updateAnnotation,
    deleteAnnotation
  } = useAnnotationStore();

  const { currentImageIndex, setCurrentImageIndex } = useProjectStore();
  const { reviewFilter, setReviewFilter, reviewModeEnabled } = useUIStore();

  const getLabel = (labelId: number) => labels.find((l) => l.id === labelId);

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'approved':
        return <Badge variant="outline" className="bg-green-500/15 text-green-600 border-green-500/20 hover:bg-green-500/25">✓</Badge>;
      case 'pending':
        return <Badge variant="outline" className="bg-yellow-500/15 text-yellow-600 border-yellow-500/20 hover:bg-yellow-500/25">?</Badge>;
      case 'rejected':
        return <Badge variant="outline" className="bg-red-500/15 text-red-600 border-red-500/20 hover:bg-red-500/25">✕</Badge>;
      default:
        return null;
    }
  };

  const getSourceIcon = (source: string) => {
    switch (source) {
      case 'manual':
        return '✏️';
      case 'propagated':
      case 'tracked':
        return '🔄';
      default:
        return '';
    }
  };

  // Go to next image with pending annotations
  const goToNextPending = async () => {
    try {
      const result = await api.getImagesWithStatus('pending');
      if (result.image_indices.length === 0) {
        useUIStore.getState().addToast('No pending annotations found', 'info');
        return;
      }

      // Find next pending image after current index
      const nextIndex = result.image_indices.find(idx => idx > currentImageIndex);
      if (nextIndex !== undefined) {
        setCurrentImageIndex(nextIndex);
        useUIStore.getState().addToast(`Jumped to image ${nextIndex + 1} (pending)`, 'success', 2000);
      } else {
        // Wrap around to first pending
        setCurrentImageIndex(result.image_indices[0]);
        useUIStore.getState().addToast(`Wrapped to image ${result.image_indices[0] + 1} (pending)`, 'success', 2000);
      }
    } catch (err) {
      console.error('Failed to get pending images:', err);
    }
  };

  // Filter annotations based on review filter
  const filteredAnnotations = annotations.filter((ann) => {
    if (reviewFilter === 'all') return true;
    return ann.status === reviewFilter;
  });

  // Count by status
  const statusCounts = {
    all: annotations.length,
    pending: annotations.filter(a => a.status === 'pending').length,
    approved: annotations.filter(a => a.status === 'approved').length,
    rejected: annotations.filter(a => a.status === 'rejected').length,
  };

  const filterButtons: { filter: ReviewFilter; label: string; activeClass: string }[] = [
    { filter: 'all', label: 'All', activeClass: 'bg-primary text-primary-foreground' },
    { filter: 'pending', label: '?', activeClass: 'bg-yellow-500 text-white hover:bg-yellow-600' },
    { filter: 'approved', label: '✓', activeClass: 'bg-green-500 text-white hover:bg-green-600' },
    { filter: 'rejected', label: '✕', activeClass: 'bg-red-500 text-white hover:bg-red-600' },
  ];

  if (annotations.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-muted-foreground text-sm">
          No annotations yet
        </p>
        <p className="text-muted-foreground text-xs mt-1">
          Draw a bounding box to create one
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Filter buttons - Always show if we have annotations */}
      {statusCounts.all > 0 && (
        <div className="space-y-4 mb-6">
          <div className="flex gap-1.5 p-1 bg-muted rounded-lg">
            {filterButtons.map((btn) => (
              <button
                key={btn.filter}
                className={cn(
                  "flex-1 px-2 py-1.5 rounded-md text-xs font-medium transition-all",
                  reviewFilter === btn.filter
                    ? btn.activeClass + " shadow-sm"
                    : "text-muted-foreground hover:text-foreground hover:bg-background/50"
                )}
                onClick={() => setReviewFilter(btn.filter)}
                title={`Show ${btn.filter} annotations`}
              >
                {btn.label} <span className="opacity-75 text-[10px] ml-0.5">({statusCounts[btn.filter]})</span>
              </button>
            ))}
          </div>

          {/* Actions for Pending Review - Logic: Show if batch actions available */}
          {reviewModeEnabled && statusCounts.pending > 0 && (
            <div className="space-y-2 p-3 bg-muted/30 rounded-lg border border-dashed animate-in fade-in">
              <Button
                variant="secondary"
                size="sm"
                className="w-full bg-yellow-500/10 text-yellow-600 hover:bg-yellow-500/20 border border-yellow-500/20 shadow-none"
                onClick={goToNextPending}
                title="Go to next image with pending annotations"
              >
                → Next Pending Image
              </Button>

              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  className="flex-1 bg-green-500/10 text-green-600 hover:bg-green-500/20 border-green-500/20 h-8"
                  onClick={async () => {
                    const pending = annotations.filter(a => a.status === 'pending');
                    for (const ann of pending) {
                      await updateAnnotation(ann.id, { status: 'approved' });
                    }
                    useUIStore.getState().addToast(`Approved ${pending.length} annotations`, 'success');
                  }}
                  title="Approve all pending annotations on this image"
                >
                  <Check className="w-3 h-3 mr-1" /> All
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="flex-1 bg-red-500/10 text-red-600 hover:bg-red-500/20 border-red-500/20 h-8"
                  onClick={async () => {
                    const pending = annotations.filter(a => a.status === 'pending');
                    for (const ann of pending) {
                      await updateAnnotation(ann.id, { status: 'rejected' });
                    }
                    useUIStore.getState().addToast(`Rejected ${pending.length} annotations`, 'info');
                  }}
                  title="Reject all pending annotations on this image"
                >
                  <X className="w-3 h-3 mr-1" /> All
                </Button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Filtered annotation list */}
      {filteredAnnotations.length === 0 ? (
        <div className="text-center py-12 opacity-50">
          <p className="text-muted-foreground text-sm">
            No {reviewFilter === 'all' ? '' : reviewFilter} annotations
          </p>
        </div>
      ) : (
        <div className="space-y-2.5">
          {filteredAnnotations.map((ann, index) => {
            const label = getLabel(ann.label_id);
            const isSelected = ann.id === selectedAnnotationId;

            return (
              <div
                key={ann.id}
                className={cn(
                  "p-3.5 rounded-xl cursor-pointer transition-all duration-200 border",
                  isSelected
                    ? "bg-primary/5 border-primary shadow-[0_0_0_1px_hsl(var(--primary))]"
                    : "bg-card hover:border-primary/50 hover:bg-accent/50 border-transparent"
                )}
                onClick={() => selectAnnotation(ann.id)}
              >
                <div className="flex items-center gap-3">
                  {/* Color dot */}
                  <span
                    className={cn(
                      "w-2.5 h-2.5 rounded-full flex-shrink-0 shadow-sm",
                      isSelected && "ring-2 ring-primary ring-offset-2"
                    )}
                    style={{ backgroundColor: label?.color || '#888' }}
                  />

                  {/* Label name */}
                  <span className={cn(
                    "flex-1 text-sm font-medium truncate",
                    isSelected ? "text-foreground" : "text-foreground"
                  )}>
                    {label?.name || 'Unknown'} <span className="text-muted-foreground font-normal text-xs">#{index + 1}</span>
                  </span>

                  {/* Badges */}
                  <div className="flex items-center gap-1.5">
                    {/* Source icon */}
                    {ann.source !== 'manual' && (
                      <span className="text-xs opacity-70" title={`Source: ${ann.source}`}>
                        {getSourceIcon(ann.source)}
                      </span>
                    )}

                    {/* Status */}
                    {getStatusBadge(ann.status)}
                  </div>
                </div>

                {/* Actions (show when selected) */}
                {isSelected && (
                  <div className="mt-3 pt-3 border-t border-border/50 animate-in slide-in-from-top-2 duration-200 space-y-3">
                    {/* Label change dropdown */}
                    <div className="flex items-center gap-2">
                      <Tag className="w-4 h-4 text-muted-foreground" />
                      <Select
                        value={String(ann.label_id)}
                        onValueChange={(value) => {
                          updateAnnotation(ann.id, { label_id: Number(value) });
                        }}
                      >
                        <SelectTrigger className="flex-1 h-8 text-xs">
                          <SelectValue placeholder="Select label" />
                        </SelectTrigger>
                        <SelectContent>
                          {labels.map((lbl) => (
                            <SelectItem key={lbl.id} value={String(lbl.id)}>
                              <div className="flex items-center gap-2">
                                <span
                                  className="w-3 h-3 rounded-full flex-shrink-0"
                                  style={{ backgroundColor: lbl.color }}
                                />
                                {lbl.name}
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Action buttons */}
                    <div className="flex gap-2">
                      {/* Show Approve/Reject only if in Review Mode OR item is pending, to avoid clutter in manual mode */}
                      {(reviewModeEnabled || ann.status === 'pending') && (
                        <>
                          <Button
                            size="sm"
                            variant="outline"
                            className="flex-1 bg-green-500/10 text-green-600 hover:bg-green-500/20 border-green-500/20 h-7 text-xs font-normal"
                            onClick={(e) => {
                              e.stopPropagation();
                              updateAnnotation(ann.id, { status: 'approved' });
                            }}
                          >
                            ✓ Approve
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            className="flex-1 bg-red-500/10 text-red-600 hover:bg-red-500/20 border-red-500/20 h-7 text-xs font-normal"
                            onClick={(e) => {
                              e.stopPropagation();
                              updateAnnotation(ann.id, { status: 'rejected' });
                            }}
                          >
                            ✕ Reject
                          </Button>
                        </>
                      )}
                      <Button
                        size="icon"
                        variant="ghost"
                        className="w-8 h-8 ml-auto text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteAnnotation(ann.id);
                        }}
                        title="Delete"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                )}

                {/* Confidence (if tracked) */}
                {ann.confidence !== null && (
                  <div className="mt-2.5 flex items-center gap-2">
                    <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${ann.confidence > 0.8 ? 'bg-green-500' : ann.confidence > 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                        style={{ width: `${ann.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-[10px] tabular-nums text-muted-foreground opacity-80">
                      {(ann.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

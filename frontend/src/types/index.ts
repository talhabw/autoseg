// Types for AutoSeg

export interface Project {
  id: number;
  name: string;
  root_dir: string;
  image_count: number;
}

export interface ImageRecord {
  id: number;
  project_id: number;
  path: string;
  width: number;
  height: number;
  order_index: number;
}

export interface Label {
  id: number;
  project_id: number;
  name: string;
  color: string;
}

export interface Annotation {
  id: number;
  image_id: number;
  label_id: number;
  bbox: [number, number, number, number] | null;  // [x1, y1, x2, y2]
  polygon: number[] | null;
  mask_rle: object | null;
  source: 'manual' | 'propagated' | 'tracked';
  confidence: number | null;
  status: 'approved' | 'pending' | 'rejected';
}

export type InteractionMode = 'view' | 'draw' | 'refine';

export interface SegmentResult {
  mask_rle: object;
  polygon: number[];
  score: number;
  bbox: [number, number, number, number];
}

export interface PropagateResult {
  bbox: [number, number, number, number];
  mask_rle: object;
  polygon: number[];
  confidence: number;
  fallback_used?: boolean;  // True if size-mismatch fallback was used
  area_ratio?: number; // Ratio of new area to old area
  duplicate_skipped?: boolean;  // True if propagation was skipped due to duplicate
  duplicate_iou?: number;  // IoU with the overlapping annotation
  conflicting_label_name?: string | null;  // If skipped due to different class at same location
}

// Point for refinement
export interface RefinePoint {
  x: number;
  y: number;
  type: 'positive' | 'negative';
}

// Bounding box being drawn
export interface DrawingBbox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

// ==================== Advanced ML Types ====================

// Instance found by find-all-instances
export interface FoundInstance {
  bbox: [number, number, number, number];
  mask_rle: object;
  polygon: number[];
  confidence: number;
  method: string;
  area_ratio?: number;
}

// Response from find-all-instances endpoint
export interface FindAllInstancesResult {
  instances: FoundInstance[];
  count: number;
}

// Advanced propagation result with method tracking
export interface PropagateAdvancedResult {
  bbox: [number, number, number, number];
  mask_rle: object;
  polygon: number[];
  confidence: number;
  fallback_used: boolean;
  area_ratio: number;
  method: 'peak' | 'dense' | 'iou_match';
  iou_score: number | null;
  duplicate_skipped?: boolean;
  duplicate_iou?: number;
  conflicting_label_name?: string | null;  // If skipped due to different class at same location
}

// Propagation mode for advanced propagation
export type PropagationMode = 'peak' | 'dense' | 'auto';

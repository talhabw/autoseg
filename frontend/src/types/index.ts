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

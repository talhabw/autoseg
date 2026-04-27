import axios from 'axios';
import type { 
  Project, 
  ImageRecord, 
  Label, 
  Annotation, 
  SegmentResult, 
  PropagateResult,
  FindAllInstancesResult,
  PropagateAdvancedResult,
  PropagationMode,
} from '../types';

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// ==================== Projects ====================

export async function createProject(projectDir: string, imageDir: string, name: string): Promise<Project> {
  const response = await api.post<Project>('/projects', {
    project_dir: projectDir,
    image_dir: imageDir,
    name,
  });
  return response.data;
}

export async function openProject(projectDir: string): Promise<Project> {
  const response = await api.post<Project>('/projects/open', {
    project_dir: projectDir,
  });
  return response.data;
}

export async function getCurrentProject(): Promise<Project | null> {
  const response = await api.get<Project | null>('/projects/current');
  return response.data;
}

export async function closeProject(): Promise<void> {
  await api.post('/projects/close');
}

export async function getSetting(key: string): Promise<string | null> {
  const response = await api.get<{ key: string; value: string | null }>(`/projects/settings/${key}`);
  return response.data.value;
}

export async function setSetting(key: string, value: string): Promise<void> {
  await api.put(`/projects/settings/${key}`, null, { params: { value } });
}

export interface ResyncImagesResult {
  added: number;
  removed: number;
  unchanged: number;
  total: number;
}

export async function resyncImages(): Promise<ResyncImagesResult> {
  const response = await api.post<ResyncImagesResult>('/projects/resync-images');
  return response.data;
}

// ==================== Images ====================

export async function listImages(): Promise<ImageRecord[]> {
  const response = await api.get<ImageRecord[]>('/images');
  return response.data;
}

export async function getImage(imageId: number): Promise<ImageRecord> {
  const response = await api.get<ImageRecord>(`/images/${imageId}`);
  return response.data;
}

export async function getImageByIndex(orderIndex: number): Promise<ImageRecord> {
  const response = await api.get<ImageRecord>(`/images/by-index/${orderIndex}`);
  return response.data;
}

export function getImageUrl(imageId: number, cacheBuster?: string | number): string {
  const url = `/api/images/${imageId}/file`;
  return cacheBuster ? `${url}?v=${cacheBuster}` : url;
}

export function getThumbnailUrl(imageId: number, size = 200, cacheBuster?: string | number): string {
  let url = `/api/images/${imageId}/thumbnail?size=${size}`;
  if (cacheBuster) url += `&v=${cacheBuster}`;
  return url;
}

export function getOptimizedImageUrl(
  imageId: number,
  maxWidth = 2048,
  quality = 85,
  cacheBuster?: string | number
): string {
  let url = `/api/images/${imageId}/optimized?max_width=${maxWidth}&quality=${quality}`;
  if (cacheBuster) url += `&v=${cacheBuster}`;
  return url;
}

export async function getImagesWithStatus(status: string): Promise<{
  status: string;
  image_indices: number[];
  count: number;
}> {
  const response = await api.get(`/images/with-status/${status}`);
  return response.data;
}

// ==================== Labels ====================

export async function listLabels(): Promise<Label[]> {
  const response = await api.get<Label[]>('/labels');
  return response.data;
}

export async function createLabel(name: string, color?: string): Promise<Label> {
  const response = await api.post<Label>('/labels', { name, color });
  return response.data;
}

export async function updateLabel(
  labelId: number,
  data: { name?: string; color?: string }
): Promise<Label> {
  const response = await api.patch<Label>(`/labels/${labelId}`, data);
  return response.data;
}

// ==================== Annotations ====================

export async function listAnnotations(imageId: number, signal?: AbortSignal): Promise<Annotation[]> {
  const response = await api.get<Annotation[]>('/annotations', {
    params: { image_id: imageId },
    signal,
  });
  return response.data;
}

export async function createAnnotation(data: {
  image_id: number;
  label_id: number;
  bbox: [number, number, number, number];
  source?: string;
  status?: string;
  mask_rle?: object;
  polygon?: number[];
}): Promise<Annotation> {
  const response = await api.post<Annotation>('/annotations', data);
  return response.data;
}

export async function updateAnnotation(
  annotationId: number,
  data: Partial<{
    label_id: number;
    bbox: [number, number, number, number];
    polygon: number[];
    mask_rle: object;
    status: string;
  }>
): Promise<Annotation> {
  const response = await api.put<Annotation>(`/annotations/${annotationId}`, data);
  return response.data;
}

export async function deleteAnnotation(annotationId: number): Promise<void> {
  await api.delete(`/annotations/${annotationId}`);
}

export async function deleteAllAnnotations(projectId: number): Promise<{ count: number }> {
  const response = await api.delete<{ status: string; count: number }>(`/annotations/all/${projectId}`);
  return { count: response.data.count };
}

export async function deleteAnnotationsAfterIndex(
  projectId: number,
  afterIndex: number
): Promise<{ count: number }> {
  const response = await api.delete<{ status: string; count: number; after_index: number }>(
    `/annotations/after-index/${projectId}`,
    { params: { after_index: afterIndex } }
  );
  return { count: response.data.count };
}

export interface FallbackReferenceResult {
  found: boolean;
  annotation: Annotation | null;
  image_index: number | null;
}

export async function findFallbackReference(
  labelId: number,
  beforeImageIndex: number,
  projectId: number,
  excludeImageIds?: number[]
): Promise<FallbackReferenceResult> {
  const response = await api.get<FallbackReferenceResult>(`/annotations/fallback/${labelId}`, {
    params: {
      before_image_index: beforeImageIndex,
      project_id: projectId,
      exclude_image_ids: excludeImageIds?.join(','),
    },
  });
  return response.data;
}

export interface MissingAnnotationsResult {
  image_indices: number[];
  total_missing: number;
}

export async function findImagesMissingAnnotations(
  projectId: number,
  labelId?: number
): Promise<MissingAnnotationsResult> {
  const response = await api.get<MissingAnnotationsResult>(`/annotations/missing/${projectId}`, {
    params: labelId !== undefined ? { label_id: labelId } : {},
  });
  return response.data;
}

// ==================== ML ====================

export async function loadSAM(device = 'cuda'): Promise<void> {
  await api.post('/ml/sam/load', { device });
}

export async function getSAMStatus(): Promise<{ loaded: boolean }> {
  const response = await api.get<{ loaded: boolean }>('/ml/sam/status');
  return response.data;
}

export interface SAMSettings {
  mask_threshold: number;
  multimask_output: boolean;
  stability_score_offset: number;
  min_region_area: number;
  keep_largest_region: boolean;
}

export async function getSAMSettings(): Promise<SAMSettings> {
  const response = await api.get<SAMSettings>('/ml/sam/settings');
  return response.data;
}

export async function updateSAMSettings(settings: Partial<SAMSettings>): Promise<SAMSettings> {
  const response = await api.patch<SAMSettings>('/ml/sam/settings', settings);
  return response.data;
}

export async function unloadAllModels(): Promise<{ models: string[] }> {
  const response = await api.post<{ status: string; models: string[] }>('/ml/unload');
  return { models: response.data.models };
}

export async function unloadEmbedModel(): Promise<{ models: string[] }> {
  const response = await api.post<{ status: string; models: string[] }>('/ml/unload/embed');
  return { models: response.data.models };
}

export async function getAvailableEmbedModels(): Promise<{ models: { id: string; name: string; available: boolean; download_url?: string; weights_file: string }[] }> {
  const response = await api.get('/ml/embed/models');
  return response.data;
}

export async function segment(
  imageId: number,
  bbox: [number, number, number, number],
  posPoints?: [number, number][],
  negPoints?: [number, number][]
): Promise<SegmentResult> {
  const response = await api.post<SegmentResult>('/ml/segment', {
    image_id: imageId,
    bbox,
    pos_points: posPoints,
    neg_points: negPoints,
  });
  return response.data;
}

export async function loadPropagation(device = 'cuda', embedModel = 'vith16'): Promise<void> {
  await api.post('/ml/propagate/load', { device, embed_model: embedModel });
}

export async function getPropagationStatus(): Promise<{
  loaded: boolean;
  sam_loaded: boolean;
  embed_loaded: boolean;
}> {
  const response = await api.get('/ml/propagate/status');
  return response.data;
}

export async function propagate(
  sourceImageId: number,
  targetImageId: number,
  sourceAnnotationId: number,
  sizeMinRatio: number = 0.8,
  sizeMaxRatio: number = 1.2,
  stopOnSizeMismatch: boolean = true,
  skipDuplicateThreshold: number = 0.9,
  topK: number = 5,
  bboxHintScale: number = 1.15,
  pruneThinArtifacts: boolean = true
): Promise<PropagateResult> {
  const response = await api.post<PropagateResult>('/ml/propagate', {
    source_image_id: sourceImageId,
    target_image_id: targetImageId,
    source_annotation_id: sourceAnnotationId,
    bbox_hint_scale: bboxHintScale,
    prune_thin_artifacts: pruneThinArtifacts,
    size_min_ratio: sizeMinRatio,
    size_max_ratio: sizeMaxRatio,
    stop_on_size_mismatch: stopOnSizeMismatch,
    skip_duplicate_threshold: skipDuplicateThreshold,
    top_k: topK,
  });
  return response.data;
}

// ==================== Advanced ML Features ====================

/**
 * Find all instances of a class in the target image.
 * Uses a reference annotation to define what the class looks like.
 */
export async function findAllInstances(
  referenceImageId: number,
  referenceAnnotationId: number,
  targetImageId: number,
  options: {
    minSimilarity?: number;
    maxInstances?: number;
    sizeTolerance?: number;
    useCachedMasks?: boolean;
  } = {}
): Promise<FindAllInstancesResult> {
  const response = await api.post<FindAllInstancesResult>('/ml/find-instances', {
    reference_image_id: referenceImageId,
    reference_annotation_id: referenceAnnotationId,
    target_image_id: targetImageId,
    min_similarity: options.minSimilarity ?? 0.6,
    max_instances: options.maxInstances ?? 20,
    size_tolerance: options.sizeTolerance ?? 0.5,
    use_cached_masks: options.useCachedMasks ?? true,
  });
  return response.data;
}

/**
 * Advanced propagation with mode selection and IoU verification.
 * 
 * Modes:
 * - "peak": Peak-based propagation (default)
 * - "dense": Dense feature correspondence (legacy DINO style)  
 * - "auto": Try peak first, fall back to dense if needed
 */
export async function propagateAdvanced(
  sourceImageId: number,
  targetImageId: number,
  sourceAnnotationId: number,
  options: {
    mode?: PropagationMode;
    iouVerify?: boolean;
    iouThreshold?: number;
    useCachedMasks?: boolean;
    bboxHintScale?: number;
    pruneThinArtifacts?: boolean;
    sizeMinRatio?: number;
    sizeMaxRatio?: number;
    stopOnSizeMismatch?: boolean;
    topK?: number;
    skipDuplicateThreshold?: number;
  } = {}
): Promise<PropagateAdvancedResult> {
  const response = await api.post<PropagateAdvancedResult>('/ml/propagate/advanced', {
    source_image_id: sourceImageId,
    target_image_id: targetImageId,
    source_annotation_id: sourceAnnotationId,
    mode: options.mode ?? 'auto',
    iou_verify: options.iouVerify ?? true,
    iou_threshold: options.iouThreshold ?? 0.3,
    use_cached_masks: options.useCachedMasks ?? true,
    bbox_hint_scale: options.bboxHintScale ?? 1.15,
    prune_thin_artifacts: options.pruneThinArtifacts ?? true,
    size_min_ratio: options.sizeMinRatio ?? 0.8,
    size_max_ratio: options.sizeMaxRatio ?? 1.2,
    stop_on_size_mismatch: options.stopOnSizeMismatch ?? true,
    top_k: options.topK ?? 5,
    skip_duplicate_threshold: options.skipDuplicateThreshold ?? 0.9,
  });
  return response.data;
}

// ==================== Export ====================

export interface ValidationWarning {
  annotation_id: number;
  severity: string;
  code: string;
  message: string;
}

export interface ValidateResponse {
  total_images: number;
  total_annotations: number;
  error_count: number;
  warning_count: number;
  is_valid: boolean;
  errors: ValidationWarning[];
  warnings: ValidationWarning[];
}

export async function validateProject(): Promise<ValidateResponse> {
  const response = await api.get<ValidateResponse>('/export/validate');
  return response.data;
}

export async function exportYolo(data: {
  output_dir: string;
  train_split?: number;
  seed?: number;
  approved_only?: boolean;
  include_negative?: boolean;
  labels_only?: boolean;
  labels_colocate?: boolean;
}): Promise<{
  train_images: number;
  val_images: number;
  total_annotations: number;
  warnings: string[];
  is_valid: boolean;
  validation_errors: string[];
}> {
  const response = await api.post('/export/yolo', data);
  return response.data;
}

export async function exportBbox(data: {
  output_dir: string;
  format: 'yolo-detect' | 'coco';
  train_split?: number;
  seed?: number;
  approved_only?: boolean;
  include_segmentation?: boolean;
  include_negative?: boolean;
  labels_only?: boolean;
  labels_colocate?: boolean;
}): Promise<{
  format: string;
  train_images: number;
  val_images: number;
  total_annotations: number;
  warnings: string[];
}> {
  const response = await api.post('/export/bbox', data);
  return response.data;
}

// ==================== Files ====================

export interface DirectoryEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

export interface DirectoryListing {
  path: string;
  parent: string | null;
  entries: DirectoryEntry[];
}

export async function listDirectory(path: string, dirsOnly = true): Promise<DirectoryListing> {
  const response = await api.get<DirectoryListing>('/files/list', {
    params: { path, dirs_only: dirsOnly },
  });
  return response.data;
}

export async function getHomeDirectory(): Promise<{ path: string }> {
  const response = await api.get<{ path: string }>('/files/home');
  return response.data;
}

export default api;

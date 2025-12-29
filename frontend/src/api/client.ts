import axios from 'axios';
import type { 
  Project, 
  ImageRecord, 
  Label, 
  Annotation, 
  SegmentResult, 
  PropagateResult 
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

// ==================== Annotations ====================

export async function listAnnotations(imageId: number): Promise<Annotation[]> {
  const response = await api.get<Annotation[]>('/annotations', {
    params: { image_id: imageId },
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

// ==================== ML ====================

export async function loadSAM(device = 'cuda'): Promise<void> {
  await api.post('/ml/sam/load', { device });
}

export async function getSAMStatus(): Promise<{ loaded: boolean }> {
  const response = await api.get<{ loaded: boolean }>('/ml/sam/status');
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
  topK: number = 5
): Promise<PropagateResult> {
  const response = await api.post<PropagateResult>('/ml/propagate', {
    source_image_id: sourceImageId,
    target_image_id: targetImageId,
    source_annotation_id: sourceAnnotationId,
    size_min_ratio: sizeMinRatio,
    size_max_ratio: sizeMaxRatio,
    stop_on_size_mismatch: stopOnSizeMismatch,
    skip_duplicate_threshold: skipDuplicateThreshold,
    top_k: topK,
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


import { create } from 'zustand';
import type { Annotation, Label, RefinePoint } from '../types';
import * as api from '../api/client';
import { useUIStore } from './uiStore';

const MAX_ANNOTATION_CACHE_ENTRIES = 40;
const annotationCache = new Map<number, Annotation[]>();
const annotationInflight = new Map<number, Promise<Annotation[]>>();

interface AnnotationState {
  // State
  annotations: Annotation[];
  labels: Label[];
  selectedAnnotationId: number | null;
  selectedLabelId: number | null;
  refinePoints: RefinePoint[];
  isLoading: boolean;

  // Actions
  loadAnnotations: (imageId: number, signal?: AbortSignal) => Promise<void>;
  prefetchAnnotations: (imageIds: number[]) => void;
  invalidateAnnotations: (imageId?: number) => void;
  loadLabels: () => Promise<void>;
  createAnnotation: (data: {
    image_id: number;
    label_id: number;
    bbox: [number, number, number, number];
    source?: string;
    status?: string;
    mask_rle?: object;
    polygon?: number[];
  }) => Promise<Annotation>;
  updateAnnotation: (
    annotationId: number,
    data: Partial<{
      label_id: number;
      bbox: [number, number, number, number];
      polygon: number[];
      mask_rle: object;
      status: string;
    }>
  ) => Promise<void>;
  deleteAnnotation: (annotationId: number) => Promise<void>;
  selectAnnotation: (annotationId: number | null) => void;
  selectLabel: (labelId: number | null) => void;
  createLabel: (name: string, color?: string) => Promise<Label>;
  updateLabel: (labelId: number, data: { name?: string; color?: string }) => Promise<void>;
  
  // Refine points
  addRefinePoint: (point: RefinePoint) => void;
  clearRefinePoints: () => void;
  
  // Clear state
  clearAnnotations: () => void;
}

export const useAnnotationStore = create<AnnotationState>((set, get) => ({
  // Initial state
  annotations: [],
  labels: [],
  selectedAnnotationId: null,
  selectedLabelId: null,
  refinePoints: [],
  isLoading: false,

  // Actions
  loadAnnotations: async (imageId, signal) => {
    set({ isLoading: true });
    try {
      const cachedAnnotations = annotationCache.get(imageId);
      if (cachedAnnotations) {
        if (!signal?.aborted) {
          set({ annotations: cachedAnnotations, selectedAnnotationId: null, refinePoints: [], isLoading: false });
        }
        return;
      }

      const annotations = await getAnnotationsForImage(imageId, signal);
      if (signal?.aborted) return;
      set({ annotations, selectedAnnotationId: null, refinePoints: [] });
    } catch (err) {
      if (signal?.aborted) return;
      throw err;
    } finally {
      if (!signal?.aborted) {
        set({ isLoading: false });
      }
    }
  },

  prefetchAnnotations: (imageIds) => {
    for (const imageId of imageIds) {
      if (annotationCache.has(imageId) || annotationInflight.has(imageId)) continue;
      void getAnnotationsForImage(imageId).catch(() => {
        // Prefetch failures should not affect the active workflow.
      });
    }
  },

  invalidateAnnotations: (imageId) => {
    if (imageId === undefined) {
      annotationCache.clear();
      annotationInflight.clear();
      return;
    }
    annotationCache.delete(imageId);
    annotationInflight.delete(imageId);
  },

  loadLabels: async () => {
    try {
      const labels = await api.listLabels();
      set({ labels });
      // Auto-select first label if none selected
      if (labels.length > 0 && get().selectedLabelId === null) {
        set({ selectedLabelId: labels[0].id });
      }
    } catch (err) {
      console.error('Failed to load labels:', err);
    }
  },

  createAnnotation: async (data) => {
    const annotation = await api.createAnnotation(data);
    addAnnotationToCache(annotation);
    set((state) => ({
      annotations: [...state.annotations, annotation],
      selectedAnnotationId: annotation.id,
    }));
    return annotation;
  },

  updateAnnotation: async (annotationId, data) => {
    const updated = await api.updateAnnotation(annotationId, data);
    updateAnnotationInCache(updated);
    set((state) => ({
      annotations: state.annotations.map((a) =>
        a.id === annotationId ? updated : a
      ),
    }));
  },

  deleteAnnotation: async (annotationId) => {
    const existingImageId = get().annotations.find((a) => a.id === annotationId)?.image_id;
    await api.deleteAnnotation(annotationId);
    removeAnnotationFromCache(annotationId, existingImageId);
    set((state) => ({
      annotations: state.annotations.filter((a) => a.id !== annotationId),
      selectedAnnotationId:
        state.selectedAnnotationId === annotationId
          ? null
          : state.selectedAnnotationId,
    }));
  },

  selectAnnotation: (annotationId) => {
    set({ selectedAnnotationId: annotationId, refinePoints: [] });
    // Switch to refine mode when an annotation is selected
    if (annotationId !== null) {
      useUIStore.getState().setMode('refine');
    }
  },

  selectLabel: (labelId) => {
    set({ selectedLabelId: labelId });
  },

  createLabel: async (name, color) => {
    const label = await api.createLabel(name, color);
    set((state) => ({
      labels: [...state.labels, label],
      selectedLabelId: label.id,
    }));
    return label;
  },

  updateLabel: async (labelId, data) => {
    const updated = await api.updateLabel(labelId, data);
    set((state) => ({
      labels: state.labels.map((l) => (l.id === labelId ? updated : l)),
    }));
  },

  addRefinePoint: (point) => {
    set((state) => ({
      refinePoints: [...state.refinePoints, point],
    }));
  },

  clearRefinePoints: () => {
    set({ refinePoints: [] });
  },

  clearAnnotations: () => {
    set({ annotations: [], selectedAnnotationId: null, refinePoints: [], isLoading: false });
  },
}));

async function getAnnotationsForImage(imageId: number, signal?: AbortSignal): Promise<Annotation[]> {
  const cachedAnnotations = annotationCache.get(imageId);
  if (cachedAnnotations) return cachedAnnotations;

  const existingRequest = annotationInflight.get(imageId);
  if (existingRequest) return existingRequest;

  const request = api.listAnnotations(imageId, signal)
    .then((annotations) => {
      annotationCache.set(imageId, annotations);
      trimAnnotationCache();
      return annotations;
    })
    .finally(() => {
      annotationInflight.delete(imageId);
    });

  annotationInflight.set(imageId, request);
  return request;
}

function addAnnotationToCache(annotation: Annotation) {
  const cachedAnnotations = annotationCache.get(annotation.image_id);
  if (!cachedAnnotations) return;
  annotationCache.set(annotation.image_id, [...cachedAnnotations, annotation]);
  trimAnnotationCache();
}

function updateAnnotationInCache(annotation: Annotation) {
  const cachedAnnotations = annotationCache.get(annotation.image_id);
  if (!cachedAnnotations) return;
  annotationCache.set(
    annotation.image_id,
    cachedAnnotations.map((cached) => cached.id === annotation.id ? annotation : cached)
  );
}

function removeAnnotationFromCache(annotationId: number, imageId?: number) {
  if (imageId !== undefined && annotationCache.has(imageId)) {
    const cachedAnnotations = annotationCache.get(imageId) ?? [];
    annotationCache.set(imageId, cachedAnnotations.filter((annotation) => annotation.id !== annotationId));
    return;
  }

  for (const [cachedImageId, cachedAnnotations] of annotationCache.entries()) {
    if (cachedAnnotations.some((annotation) => annotation.id === annotationId)) {
      annotationCache.set(cachedImageId, cachedAnnotations.filter((annotation) => annotation.id !== annotationId));
      return;
    }
  }
}

function trimAnnotationCache() {
  while (annotationCache.size > MAX_ANNOTATION_CACHE_ENTRIES) {
    const firstKey = annotationCache.keys().next().value;
    if (firstKey === undefined) break;
    annotationCache.delete(firstKey);
  }
}

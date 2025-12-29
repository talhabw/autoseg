import { create } from 'zustand';
import type { Annotation, Label, RefinePoint } from '../types';
import * as api from '../api/client';

interface AnnotationState {
  // State
  annotations: Annotation[];
  labels: Label[];
  selectedAnnotationId: number | null;
  selectedLabelId: number | null;
  refinePoints: RefinePoint[];
  isLoading: boolean;

  // Actions
  loadAnnotations: (imageId: number) => Promise<void>;
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
  loadAnnotations: async (imageId) => {
    set({ isLoading: true });
    try {
      const annotations = await api.listAnnotations(imageId);
      set({ annotations, selectedAnnotationId: null, refinePoints: [] });
    } finally {
      set({ isLoading: false });
    }
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
    set((state) => ({
      annotations: [...state.annotations, annotation],
      selectedAnnotationId: annotation.id,
    }));
    return annotation;
  },

  updateAnnotation: async (annotationId, data) => {
    const updated = await api.updateAnnotation(annotationId, data);
    set((state) => ({
      annotations: state.annotations.map((a) =>
        a.id === annotationId ? updated : a
      ),
    }));
  },

  deleteAnnotation: async (annotationId) => {
    await api.deleteAnnotation(annotationId);
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

  addRefinePoint: (point) => {
    set((state) => ({
      refinePoints: [...state.refinePoints, point],
    }));
  },

  clearRefinePoints: () => {
    set({ refinePoints: [] });
  },

  clearAnnotations: () => {
    set({ annotations: [], selectedAnnotationId: null, refinePoints: [] });
  },
}));

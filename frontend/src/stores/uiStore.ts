import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { InteractionMode } from '../types';
import * as api from '../api/client';
import { toast } from 'sonner';

export type EmbedModel = 
  | 'vitb16' | 'vitl16' | 'vith16' 
  | 'pixio_vitb16' | 'pixio_vitl16' | 'pixio_vith16' | 'pixio_vit1b16';
export type ReviewFilter = 'all' | 'pending' | 'approved' | 'rejected';

// Keys for localStorage
const STORAGE_KEY = 'autoseg-ui-settings';

interface UIState {
  // Persisted settings
  lastProjectPath: string | null;
  embedModel: EmbedModel;
  maskOpacity: number;
  sizeMinRatio: number; // Min allowed size ratio (e.g. 0.8)
  sizeMaxRatio: number; // Max allowed size ratio (e.g. 1.2)
  stopOnSizeMismatch: boolean;  // If true, stop propagation when size differs; if false, use fallback
  topK: number; // Number of peak candidates to try during propagation
  
  // Session state (not persisted)
  mode: InteractionMode;
  trackModeEnabled: boolean;
  reviewModeEnabled: boolean;
  reviewFilter: ReviewFilter;
  samLoaded: boolean;
  propagationLoaded: boolean;
  isLoadingModel: boolean;
  isPropagating: boolean;  // Track if propagation is in progress
  
  // Modal states
  showCreateProjectModal: boolean;
  showOpenProjectModal: boolean;
  showExportModal: boolean;
  showSettingsModal: boolean;
  
  // Status message (persistent only, e.g. loading)
  statusMessage: string;
  
  // Actions
  setMode: (mode: InteractionMode) => void;
  setTrackMode: (enabled: boolean) => void;
  setReviewMode: (enabled: boolean) => void;
  setReviewFilter: (filter: ReviewFilter) => void;
  setMaskOpacity: (opacity: number) => void;
  setEmbedModel: (model: EmbedModel) => Promise<void>;
  setLastProjectPath: (path: string | null) => void;
  loadSAM: () => Promise<void>;
  loadPropagation: () => Promise<void>;
  checkModelStatus: () => Promise<void>;
  
  // Modal actions
  setShowCreateProjectModal: (show: boolean) => void;
  setShowOpenProjectModal: (show: boolean) => void;
  setShowExportModal: (show: boolean) => void;
  setShowSettingsModal: (show: boolean) => void;
  setSizeMinRatio: (ratio: number) => void;
  setSizeMaxRatio: (ratio: number) => void;
  setStopOnSizeMismatch: (stop: boolean) => void;
  setTopK: (k: number) => void;
  
  // Status
  setStatusMessage: (message: string) => void;
  setIsPropagating: (value: boolean) => void;
  
  addToast: (message: string, type?: 'success' | 'error' | 'warning' | 'info', duration?: number) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      // Persisted settings (initial values, will be overridden from storage)
      lastProjectPath: null,
      embedModel: 'vith16',
      maskOpacity: 0.5,
      sizeMinRatio: 0.8,
      sizeMaxRatio: 1.2,
      stopOnSizeMismatch: true,  // Default: stop on size mismatch (safer)
      topK: 5, // Try 5 peak candidates by default
      
      // Session state
      mode: 'view',
      trackModeEnabled: false,
      reviewModeEnabled: false,
      reviewFilter: 'all',
      samLoaded: false,
      propagationLoaded: false,
      isLoadingModel: false,
      isPropagating: false,
      showCreateProjectModal: false,
      showOpenProjectModal: false,
      showExportModal: false,
      showSettingsModal: false,
      statusMessage: 'Ready',

      // Actions
      setMode: (mode) => set({ mode }),

      setTrackMode: (enabled) => set({ trackModeEnabled: enabled }),

      setReviewMode: (enabled) => set({ reviewModeEnabled: enabled, reviewFilter: enabled ? 'pending' : 'all' }),

      setReviewFilter: (filter) => set({ reviewFilter: filter }),

      setMaskOpacity: (opacity) => set({ maskOpacity: Math.max(0, Math.min(1, opacity)) }),

      setEmbedModel: async (model) => {
        const { propagationLoaded } = get();
        
        // Unload old model if it was loaded
        if (propagationLoaded) {
          set({ statusMessage: 'Unloading previous model...' });
          try {
            await api.unloadEmbedModel();
          } catch (err) {
            console.error('Failed to unload embed model:', err);
          }
        }
        
        set({ embedModel: model, propagationLoaded: false, statusMessage: `Model set to ${model}` });
      },

      setLastProjectPath: (path) => set({ lastProjectPath: path }),

      loadSAM: async () => {
        set({ isLoadingModel: true, statusMessage: 'Loading SAM model...' });
        try {
          await api.loadSAM();
          set({ samLoaded: true, statusMessage: 'SAM model loaded' });
        } catch (err) {
          set({ statusMessage: 'Failed to load SAM model' });
          throw err;
        } finally {
          set({ isLoadingModel: false });
        }
      },

      loadPropagation: async () => {
        const { embedModel } = get();
        set({ isLoadingModel: true, statusMessage: `Loading tracking models (${embedModel})...` });
        try {
          await api.loadPropagation('cuda', embedModel);
          set({ samLoaded: true, propagationLoaded: true, statusMessage: `Tracking models loaded (${embedModel})` });
        } catch (err) {
          set({ statusMessage: 'Failed to load tracking models' });
          throw err;
        } finally {
          set({ isLoadingModel: false });
        }
      },

      checkModelStatus: async () => {
        try {
          const [samStatus, propStatus] = await Promise.all([
            api.getSAMStatus(),
            api.getPropagationStatus(),
          ]);
          set({
            samLoaded: samStatus.loaded,
            propagationLoaded: propStatus.loaded,
          });
        } catch {
          // Ignore errors during status check
        }
      },

      setShowCreateProjectModal: (show) => set({ showCreateProjectModal: show }),
      setShowOpenProjectModal: (show) => set({ showOpenProjectModal: show }),
      setShowExportModal: (show) => set({ showExportModal: show }),
      setShowSettingsModal: (show) => set({ showSettingsModal: show }),
      setSizeMinRatio: (ratio) => set({ sizeMinRatio: Math.max(0.1, Math.min(2.0, ratio)) }),
      setSizeMaxRatio: (ratio) => set({ sizeMaxRatio: Math.max(0.1, Math.min(5.0, ratio)) }),
      setStopOnSizeMismatch: (stop) => set({ stopOnSizeMismatch: stop }),
      setTopK: (k) => set({ topK: Math.max(1, Math.min(10, k)) }),

      setStatusMessage: (message) => set({ statusMessage: message }),
      setIsPropagating: (value) => set({ isPropagating: value }),
      
      addToast: (message, type = 'info', duration) => {
        const options = duration ? { duration } : undefined;
        if (type === 'success') {
          toast.success(message, options);
        } else if (type === 'error') {
          toast.error(message, options);
        } else if (type === 'warning') {
          toast.warning(message, options);
        } else {
          toast.info(message, options);
        }
      },
    }),
    {
      name: STORAGE_KEY,
      // Only persist these specific fields
      partialize: (state) => ({
        lastProjectPath: state.lastProjectPath,
        embedModel: state.embedModel,
        maskOpacity: state.maskOpacity,
        sizeMinRatio: state.sizeMinRatio,
        sizeMaxRatio: state.sizeMaxRatio,
        stopOnSizeMismatch: state.stopOnSizeMismatch,
        topK: state.topK,
      }),
    }
  )
);

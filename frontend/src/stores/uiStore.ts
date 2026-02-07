import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { InteractionMode, PropagationMode } from '../types';
import * as api from '../api/client';
import { toast } from 'sonner';

export type EmbedModel = 
  | 'vitb16' | 'vitl16' | 'vith16' 
  | 'pixio_vitb16' | 'pixio_vitl16' | 'pixio_vith16' | 'pixio_vit1b16';
export type ReviewFilter = 'all' | 'pending' | 'approved' | 'rejected';

// Keys for localStorage
const STORAGE_KEY = 'autoseg-ui-settings';

interface PerformanceProgress {
  current: number;
  total: number;
  successCount: number;
  failedCount: number;
  skippedCount: number;
  startTime: number;
  failedImageIndex: number | null;
  failedLabels: string[];
}

interface UIState {
  // Persisted settings
  lastProjectPath: string | null;
  embedModel: EmbedModel;
  maskOpacity: number;
  sizeMinRatio: number; // Min allowed size ratio (e.g. 0.8)
  sizeMaxRatio: number; // Max allowed size ratio (e.g. 1.2)
  stopOnSizeMismatch: boolean;  // If true, stop propagation when size differs; if false, use fallback
  topK: number; // Number of peak candidates to try during propagation
  propagationMode: PropagationMode; // 'peak', 'dense', or 'auto'
  iouVerify: boolean; // Whether to verify results against dense prediction
  iouThreshold: number; // Minimum IoU with dense prediction to accept
  autoNext: boolean; // Auto-advance to next image after propagation
  propagationFailureMode: 'stop' | 'skip'; // 'stop' = stop on failure, 'skip' = skip failed image and continue
  
  // Performance mode settings
  performanceMode: boolean; // When true, skip UI updates during auto-propagation
  
  // SAM settings
  samMaskThreshold: number; // Logit threshold for mask generation (-2.0 to 2.0)
  samMultimaskOutput: boolean; // Generate multiple mask candidates
  samMinRegionArea: number; // Minimum pixels to keep (removes small islands)
  samKeepLargestRegion: boolean; // Always keep largest connected region
  
  // Session state (not persisted)
  mode: InteractionMode;
  trackModeEnabled: boolean;
  reviewModeEnabled: boolean;
  reviewFilter: ReviewFilter;
  samLoaded: boolean;
  propagationLoaded: boolean;
  isLoadingModel: boolean;
  isPropagating: boolean;  // Track if propagation is in progress
  performanceProgress: PerformanceProgress | null; // Progress tracking for batch propagation
  
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
  setPropagationMode: (mode: PropagationMode) => void;
  setIouVerify: (verify: boolean) => void;
  setIouThreshold: (threshold: number) => void;
  setAutoNext: (enabled: boolean) => void;
  setPropagationFailureMode: (mode: 'stop' | 'skip') => void;
  
  // SAM settings actions
  setSamMaskThreshold: (threshold: number) => Promise<void>;
  setSamMultimaskOutput: (enabled: boolean) => Promise<void>;
  setSamMinRegionArea: (area: number) => Promise<void>;
  setSamKeepLargestRegion: (enabled: boolean) => Promise<void>;
  syncSamSettings: () => Promise<void>;
  
  // Status
  setStatusMessage: (message: string) => void;
  setIsPropagating: (value: boolean) => void;
  
  // Performance mode actions
  setPerformanceMode: (enabled: boolean) => void;
  setPerformanceProgress: (progress: PerformanceProgress | null) => void;
  updatePerformanceProgress: (update: Partial<PerformanceProgress>) => void;
  
  addToast: (message: string, type?: 'success' | 'error' | 'warning' | 'info', duration?: number) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      // Persisted settings (initial values, will be overridden from storage)
      lastProjectPath: null,
      embedModel: 'vitl16', // Default: DINOv3 ViT-L
      maskOpacity: 0.5,
      sizeMinRatio: 0.5, // Min allowed size ratio (0.5x)
      sizeMaxRatio: 2.0, // Max allowed size ratio (2.0x)
      stopOnSizeMismatch: true,  // Default: stop on size mismatch (safer)
      topK: 5, // Try 5 peak candidates by default
      propagationMode: 'auto', // Default: auto mode tries peak then dense
      iouVerify: true, // Default: verify results
      iouThreshold: 0.3, // Default: 30% IoU threshold
      autoNext: false, // Default: manual navigation
      propagationFailureMode: 'stop', // Default: stop on failure (safer)
      
      // Performance mode settings
      performanceMode: false, // Default: normal mode with UI updates
      
      // SAM settings
      samMaskThreshold: 0.0, // Default: 0.0 (standard logit threshold)
      samMultimaskOutput: true, // Default: generate multiple candidates
      samMinRegionArea: 1200, // Default: remove regions smaller than 1200 pixels
      samKeepLargestRegion: true, // Default: always keep largest region
      
      // Session state
      mode: 'view',
      trackModeEnabled: false,
      reviewModeEnabled: false,
      reviewFilter: 'all',
      samLoaded: false,
      propagationLoaded: false,
      isLoadingModel: false,
      isPropagating: false,
      performanceProgress: null,
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
        const { embedModel, syncSamSettings } = get();
        set({ isLoadingModel: true, statusMessage: `Loading tracking models (${embedModel})...` });
        try {
          await api.loadPropagation('cuda', embedModel);
          set({ samLoaded: true, propagationLoaded: true, statusMessage: `Tracking models loaded (${embedModel})` });
          // Sync SAM settings from backend after loading
          await syncSamSettings();
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
      setPropagationMode: (mode) => set({ propagationMode: mode }),
      setIouVerify: (verify) => set({ iouVerify: verify }),
      setIouThreshold: (threshold) => set({ iouThreshold: Math.max(0, Math.min(1, threshold)) }),
      setAutoNext: (enabled) => set({ autoNext: enabled }),
      setPropagationFailureMode: (mode) => set({ propagationFailureMode: mode }),

      // SAM settings - sync with backend
      setSamMaskThreshold: async (threshold) => {
        const clamped = Math.max(-2.0, Math.min(2.0, threshold));
        set({ samMaskThreshold: clamped });
        try {
          await api.updateSAMSettings({ mask_threshold: clamped });
        } catch (err) {
          console.error('Failed to update SAM mask threshold:', err);
        }
      },
      
      setSamMultimaskOutput: async (enabled) => {
        set({ samMultimaskOutput: enabled });
        try {
          await api.updateSAMSettings({ multimask_output: enabled });
        } catch (err) {
          console.error('Failed to update SAM multimask output:', err);
        }
      },
      
      setSamMinRegionArea: async (area) => {
        const clamped = Math.max(0, area);
        set({ samMinRegionArea: clamped });
        try {
          await api.updateSAMSettings({ min_region_area: clamped });
        } catch (err) {
          console.error('Failed to update SAM min region area:', err);
        }
      },
      
      setSamKeepLargestRegion: async (enabled) => {
        set({ samKeepLargestRegion: enabled });
        try {
          await api.updateSAMSettings({ keep_largest_region: enabled });
        } catch (err) {
          console.error('Failed to update SAM keep largest region:', err);
        }
      },
      
      syncSamSettings: async () => {
        try {
          const settings = await api.getSAMSettings();
          set({
            samMaskThreshold: settings.mask_threshold,
            samMultimaskOutput: settings.multimask_output,
            samMinRegionArea: settings.min_region_area,
            samKeepLargestRegion: settings.keep_largest_region,
          });
        } catch (err) {
          console.error('Failed to sync SAM settings:', err);
        }
      },

      setStatusMessage: (message) => set({ statusMessage: message }),
      setIsPropagating: (value) => set({ isPropagating: value }),
      
      // Performance mode actions
      setPerformanceMode: (enabled) => set({ performanceMode: enabled }),
      setPerformanceProgress: (progress) => set({ performanceProgress: progress }),
      updatePerformanceProgress: (update) => set((state) => ({
        performanceProgress: state.performanceProgress 
          ? { ...state.performanceProgress, ...update }
          : null,
      })),
      
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
        propagationMode: state.propagationMode,
        iouVerify: state.iouVerify,
        iouThreshold: state.iouThreshold,
        autoNext: state.autoNext,
        performanceMode: state.performanceMode,
      }),
    }
  )
);

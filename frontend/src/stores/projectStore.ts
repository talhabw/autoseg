import { create } from 'zustand';
import type { Project, ImageRecord } from '../types';
import * as api from '../api/client';
import { useUIStore } from './uiStore';

interface ProjectState {
  // State
  project: Project | null;
  images: ImageRecord[];
  currentImageIndex: number;
  isLoading: boolean;
  error: string | null;

  // Computed
  currentImage: ImageRecord | null;
  imageCount: number;

  // Actions
  createProject: (projectDir: string, imageDir: string, name: string) => Promise<void>;
  openProject: (projectDir: string) => Promise<void>;
  closeProject: () => Promise<void>;
  loadImages: () => Promise<void>;
  setCurrentImageIndex: (index: number) => void;
  nextImage: () => void;
  prevImage: () => void;
  tryOpenLastProject: () => Promise<boolean>;
  resyncImages: () => Promise<{ added: number; removed: number }>;
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  // Initial state
  project: null,
  images: [],
  currentImageIndex: 0,
  isLoading: false,
  error: null,

  // Computed getters
  get currentImage() {
    const { images, currentImageIndex } = get();
    return images[currentImageIndex] ?? null;
  },

  get imageCount() {
    return get().images.length;
  },

  // Actions
  createProject: async (projectDir, imageDir, name) => {
    set({ isLoading: true, error: null });
    try {
      const project = await api.createProject(projectDir, imageDir, name);
      set({ project, currentImageIndex: 0 });
      await get().loadImages();
      // Save last project path
      useUIStore.getState().setLastProjectPath(projectDir);
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to create project' });
      throw err;
    } finally {
      set({ isLoading: false });
    }
  },

  openProject: async (projectDir) => {
    set({ isLoading: true, error: null });
    try {
      const project = await api.openProject(projectDir);
      
      // Try to restore last image index
      let lastIndex = 0;
      try {
        const savedIndex = await api.getSetting('last_image_index');
        if (savedIndex !== null) {
          lastIndex = parseInt(savedIndex, 10) || 0;
        }
      } catch {
        // Ignore errors getting last index
      }
      
      set({ project, currentImageIndex: lastIndex });
      await get().loadImages();
      
      // Validate index is within bounds
      const { images } = get();
      if (lastIndex >= images.length) {
        set({ currentImageIndex: 0 });
      }
      
      // Save last project path
      useUIStore.getState().setLastProjectPath(projectDir);
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to open project' });
      throw err;
    } finally {
      set({ isLoading: false });
    }
  },

  closeProject: async () => {
    try {
      await api.closeProject();
    } finally {
      set({ project: null, images: [], currentImageIndex: 0, error: null });
    }
  },

  loadImages: async () => {
    try {
      const images = await api.listImages();
      set({ images });
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to load images' });
    }
  },

  setCurrentImageIndex: (index) => {
    const { images } = get();
    if (index >= 0 && index < images.length) {
      set({ currentImageIndex: index });
    }
  },

  nextImage: () => {
    const { currentImageIndex, images } = get();
    if (currentImageIndex < images.length - 1) {
      const newIndex = currentImageIndex + 1;
      set({ currentImageIndex: newIndex });
    }
  },

  prevImage: () => {
    const { currentImageIndex } = get();
    if (currentImageIndex > 0) {
      const newIndex = currentImageIndex - 1;
      set({ currentImageIndex: newIndex });
    }
  },

  // Try to open last project from localStorage
  tryOpenLastProject: async () => {
    const lastPath = useUIStore.getState().lastProjectPath;
    if (!lastPath) return false;
    
    try {
      await get().openProject(lastPath);
      return true;
    } catch {
      // Clear invalid last project path
      useUIStore.getState().setLastProjectPath(null);
      return false;
    }
  },

  resyncImages: async () => {
    const { project } = get();
    if (!project) throw new Error('No project loaded');

    try {
      const result = await api.resyncImages();

      // Reload images to get updated list
      await get().loadImages();

      // Reset index if current is out of bounds
      const { images, currentImageIndex } = get();
      if (currentImageIndex >= images.length) {
        set({ currentImageIndex: Math.max(0, images.length - 1) });
      }

      return { added: result.added, removed: result.removed };
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to resync images' });
      throw err;
    }
  },
}));

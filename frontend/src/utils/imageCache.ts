const MAX_CACHED_IMAGES = 24;

type ImageCacheEntry = {
  image?: HTMLImageElement;
  promise?: Promise<HTMLImageElement>;
};

const imageCache = new Map<string, ImageCacheEntry>();

export function getCachedImage(src: string): HTMLImageElement | null {
  const entry = imageCache.get(src);
  return entry?.image ?? null;
}

export function preloadImage(src: string): Promise<HTMLImageElement> {
  const existing = imageCache.get(src);
  if (existing?.image) return Promise.resolve(existing.image);
  if (existing?.promise) return existing.promise;

  const promise = new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => {
      imageCache.set(src, { image });
      trimImageCache();
      resolve(image);
    };
    image.onerror = () => {
      imageCache.delete(src);
      reject(new Error(`Failed to preload image: ${src}`));
    };
    image.src = src;
  });

  imageCache.set(src, { promise });
  trimImageCache();
  return promise;
}

function trimImageCache() {
  while (imageCache.size > MAX_CACHED_IMAGES) {
    const firstKey = imageCache.keys().next().value;
    if (!firstKey) break;
    imageCache.delete(firstKey);
  }
}

/**
 * RLE (Run-Length Encoding) utilities for mask decoding
 * 
 * Supports both:
 * - Simple RLE: counts is number[]
 * - COCO RLE: counts is a compressed string (from pycocotools)
 */

export interface RLEMask {
  counts: number[] | string;
  size: [number, number];  // [height, width]
}

/**
 * Decode COCO-style compressed RLE string to run lengths
 * This matches pycocotools' LEB128 variant encoding
 */
function decodeCocoCounts(s: string): number[] {
  const cnts: number[] = [];
  let p = 0;
  
  while (p < s.length) {
    let x = 0;
    let k = 0;
    let more = true;
    
    while (more) {
      const c = s.charCodeAt(p) - 48;  // '0' = 48
      x |= (c & 0x1f) << (5 * k);
      more = (c & 0x20) !== 0;
      p++;
      k++;
      if (!more && (c & 0x10) !== 0) {
        x |= -1 << (5 * k);
      }
    }
    
    if (cnts.length > 2) {
      x += cnts[cnts.length - 2];
    }
    cnts.push(x);
  }
  
  return cnts;
}

/**
 * Decode RLE counts to binary mask array
 * Masks are stored in column-major (Fortran) order
 */
function countsToMask(counts: number[], height: number, width: number): Uint8Array {
  const mask = new Uint8Array(height * width);
  let idx = 0;
  let val = 0;
  
  for (const count of counts) {
    for (let i = 0; i < count && idx < mask.length; i++) {
      mask[idx++] = val;
    }
    val = 1 - val;
  }
  
  return mask;
}

/**
 * Decode RLE mask to binary mask array
 */
export function decodeRLE(rle: RLEMask): Uint8Array {
  const [height, width] = rle.size;
  
  let counts: number[];
  if (typeof rle.counts === 'string') {
    counts = decodeCocoCounts(rle.counts);
  } else {
    counts = rle.counts;
  }
  
  return countsToMask(counts, height, width);
}

/**
 * Convert mask (column-major) to RGBA image data for canvas rendering
 */
export function maskToImageData(
  mask: Uint8Array,
  width: number,
  height: number,
  color: string,
  opacity: number = 0.5
): ImageData {
  const imageData = new ImageData(width, height);
  const data = imageData.data;
  
  // Parse color
  const r = parseInt(color.slice(1, 3), 16);
  const g = parseInt(color.slice(3, 5), 16);
  const b = parseInt(color.slice(5, 7), 16);
  const a = Math.floor(opacity * 255);
  
  // Mask is in column-major (Fortran) order
  // mask[x * height + y] = value at (x, y)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const maskIdx = x * height + y;  // Column-major
      const imgIdx = (y * width + x) * 4;  // Row-major RGBA
      
      if (mask[maskIdx]) {
        data[imgIdx] = r;
        data[imgIdx + 1] = g;
        data[imgIdx + 2] = b;
        data[imgIdx + 3] = a;
      }
    }
  }
  
  return imageData;
}

/**
 * Create a canvas element from mask data
 */
export function maskToCanvas(
  rle: RLEMask,
  color: string,
  opacity: number = 0.5
): HTMLCanvasElement {
  const [height, width] = rle.size;
  
  let mask: Uint8Array;
  try {
    mask = decodeRLE(rle);
  } catch (err) {
    console.error('Failed to decode RLE:', err);
    // Return empty canvas
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }
  
  const imageData = maskToImageData(mask, width, height, color, opacity);
  
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.putImageData(imageData, 0, 0);
  }
  
  return canvas;
}

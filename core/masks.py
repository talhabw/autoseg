"""
Mask utilities - RLE encoding/decoding and mask operations.
"""

import numpy as np
from typing import Optional, Any

# Try to import pycocotools for efficient RLE encoding
try:
    from pycocotools import mask as mask_utils
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False


def mask_to_rle(mask: np.ndarray) -> dict:
    """
    Convert a binary mask to RLE (Run-Length Encoding) format.
    
    Args:
        mask: Binary mask of shape (H, W) with dtype bool or uint8
        
    Returns:
        RLE dict with 'counts' (str or list) and 'size' [H, W]
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    
    # Ensure mask is uint8 and Fortran-contiguous (required by pycocotools)
    mask = np.asfortranarray(mask.astype(np.uint8))
    
    if HAS_PYCOCOTOOLS:
        rle = mask_utils.encode(mask)
        # Decode bytes to string for JSON serialization
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    else:
        # Fallback: simple RLE without compression
        return _simple_rle_encode(mask)


def rle_to_mask(rle: dict, height: Optional[int] = None, width: Optional[int] = None) -> np.ndarray:
    """
    Convert RLE encoding back to binary mask.
    
    Args:
        rle: RLE dict with 'counts' and 'size'
        height: Optional height (uses rle['size'] if not provided)
        width: Optional width (uses rle['size'] if not provided)
        
    Returns:
        Binary mask of shape (H, W) with dtype uint8
    """
    if height is None or width is None:
        height, width = rle['size']
    
    if HAS_PYCOCOTOOLS:
        # Encode counts back to bytes if needed
        rle_copy = rle.copy()
        if isinstance(rle_copy['counts'], str):
            rle_copy['counts'] = rle_copy['counts'].encode('utf-8')
        mask = mask_utils.decode(rle_copy)
        return mask
    else:
        return _simple_rle_decode(rle, height, width)


def _simple_rle_encode(mask: np.ndarray) -> dict:
    """Simple RLE encoding without pycocotools."""
    pixels = mask.flatten(order='F')  # Fortran order (column-major)
    
    # Find runs
    runs = []
    prev = 0
    count = 0
    
    for pixel in pixels:
        if pixel == prev:
            count += 1
        else:
            runs.append(count)
            count = 1
            prev = pixel
    runs.append(count)
    
    # RLE starts with the count of zeros
    if mask.flat[0] != 0:
        runs.insert(0, 0)
    
    return {
        'counts': runs,
        'size': list(mask.shape)
    }


def _simple_rle_decode(rle: dict, height: int, width: int) -> np.ndarray:
    """Simple RLE decoding without pycocotools."""
    counts = rle['counts']
    
    if isinstance(counts, str):
        # This is compressed format, we need pycocotools
        raise RuntimeError("Compressed RLE format requires pycocotools")
    
    # Uncompressed format: list of run lengths
    pixels = []
    val = 0
    for count in counts:
        pixels.extend([val] * count)
        val = 1 - val
    
    mask = np.array(pixels, dtype=np.uint8).reshape((height, width), order='F')
    return mask


def mask_to_bbox(mask: np.ndarray) -> list[float]:
    """
    Get bounding box from binary mask.
    
    Args:
        mask: Binary mask of shape (H, W)
        
    Returns:
        Bounding box [x1, y1, x2, y2] in pixel coordinates
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any():
        return [0, 0, 0, 0]
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    return [float(x1), float(y1), float(x2 + 1), float(y2 + 1)]


def mask_area(mask: np.ndarray) -> int:
    """Get the area (number of pixels) of a mask."""
    return int(np.sum(mask > 0))


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    return float(intersection / union)


def resize_mask(mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize a binary mask to target size.
    
    Args:
        mask: Binary mask of shape (H, W)
        target_size: Target (height, width)
        
    Returns:
        Resized mask
    """
    import cv2
    
    target_h, target_w = target_size
    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(np.uint8)

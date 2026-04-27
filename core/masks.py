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
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle
    else:
        # Fallback: simple RLE without compression
        return _simple_rle_encode(mask)


def rle_to_mask(
    rle: dict, height: Optional[int] = None, width: Optional[int] = None
) -> np.ndarray:
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
        height, width = rle["size"]

    if HAS_PYCOCOTOOLS:
        # Encode counts back to bytes if needed
        rle_copy = rle.copy()
        if isinstance(rle_copy["counts"], str):
            rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
        mask = mask_utils.decode(rle_copy)
        return mask
    else:
        return _simple_rle_decode(rle, height, width)


def _simple_rle_encode(mask: np.ndarray) -> dict:
    """Simple RLE encoding without pycocotools."""
    pixels = mask.flatten(order="F")  # Fortran order (column-major)

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

    return {"counts": runs, "size": list(mask.shape)}


def _simple_rle_decode(rle: dict, height: int, width: int) -> np.ndarray:
    """Simple RLE decoding without pycocotools."""
    counts = rle["counts"]

    if isinstance(counts, str):
        # This is compressed format, we need pycocotools
        raise RuntimeError("Compressed RLE format requires pycocotools")

    # Uncompressed format: list of run lengths
    pixels = []
    val = 0
    for count in counts:
        pixels.extend([val] * count)
        val = 1 - val

    mask = np.array(pixels, dtype=np.uint8).reshape((height, width), order="F")
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


def remove_small_regions(
    mask: np.ndarray,
    min_area: int = 100,
    keep_largest: bool = True,
) -> np.ndarray:
    """
    Remove small disconnected regions from a binary mask.

    Args:
        mask: Binary mask of shape (H, W)
        min_area: Minimum area in pixels to keep a region (default 100)
        keep_largest: If True, ONLY keep the largest region (ignores min_area)
                     If False, keep all regions >= min_area

    Returns:
        Cleaned binary mask with small regions removed
    """
    import logging
    from scipy import ndimage

    logger = logging.getLogger(__name__)

    if mask.sum() == 0:
        return mask

    # Label connected components
    labeled_mask, num_features = ndimage.label(mask)

    if num_features == 0:
        return mask

    if num_features == 1:
        # Only one region, keep it
        region_size = int(mask.sum())
        logger.debug(f"[remove_small_regions] Single region detected: {region_size} px")
        return mask

    # Get sizes of each region
    region_sizes = ndimage.sum(mask, labeled_mask, range(1, num_features + 1))

    # Log all region sizes for debugging
    sorted_sizes = sorted(enumerate(region_sizes), key=lambda x: x[1], reverse=True)
    logger.debug(
        f"[remove_small_regions] Found {num_features} regions, keep_largest={keep_largest}, min_area={min_area}"
    )
    for i, (idx, size) in enumerate(sorted_sizes[:10]):  # Show top 10
        label = idx + 1
        logger.debug(f"  Region {i + 1}: {int(size)} px (label={label})")
    if len(sorted_sizes) > 10:
        logger.debug(f"  ... and {len(sorted_sizes) - 10} more smaller regions")

    # Find regions to keep
    if keep_largest:
        # ONLY keep the largest region - ignore all others
        largest_idx = np.argmax(region_sizes) + 1  # +1 because labels start at 1
        regions_to_keep = [largest_idx]
        logger.debug(
            f"  -> Keeping ONLY largest region (label={largest_idx}, {int(region_sizes[largest_idx - 1])} px)"
        )
    else:
        # Keep all regions that meet min_area threshold
        regions_to_keep = [
            i + 1 for i, size in enumerate(region_sizes) if size >= min_area
        ]

        if not regions_to_keep:
            # If nothing meets criteria, keep largest
            largest_idx = np.argmax(region_sizes) + 1
            regions_to_keep = [largest_idx]
            logger.debug(f"  -> No regions >= {min_area} px, keeping largest")
        else:
            logger.debug(
                f"  -> Keeping {len(regions_to_keep)} regions >= {min_area} px"
            )

    # Create cleaned mask
    cleaned_mask = np.isin(labeled_mask, regions_to_keep).astype(np.uint8)

    removed_count = num_features - len(regions_to_keep)
    if removed_count > 0:
        logger.debug(f"  -> Removed {removed_count} small regions")

    return cleaned_mask


def prune_thin_artifacts(
    mask: np.ndarray,
    kernel_size: int = 3,
    min_area_ratio: float = 0.6,
) -> np.ndarray:
    """Remove thin connected protrusions from a mask conservatively.

    This is intended for tracked masks that occasionally grow thin whisker-like
    branches off the main object. A small opening removes narrow attachments,
    then a closing restores the main body. If that would damage the mask too
    much, the original mask is returned unchanged.
    """
    import cv2

    binary_mask = (mask > 0).astype(np.uint8)
    original_area = int(binary_mask.sum())

    if original_area == 0:
        return binary_mask

    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    if kernel_size <= 1:
        return binary_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    if int(opened_mask.sum()) == 0:
        return binary_mask

    cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = remove_small_regions(cleaned_mask, min_area=0, keep_largest=True)

    cleaned_area = int(cleaned_mask.sum())
    if cleaned_area == 0:
        return binary_mask

    if cleaned_area / original_area < min_area_ratio:
        return binary_mask

    return cleaned_mask.astype(np.uint8)


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
        mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(np.uint8)

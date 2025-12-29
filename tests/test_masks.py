"""
Tests for mask utilities
"""

import numpy as np
import pytest

from core.masks import (
    mask_to_rle, rle_to_mask, mask_to_bbox, mask_area, mask_iou, resize_mask
)


def test_simple_rle_roundtrip():
    """Test simple RLE encoding/decoding roundtrip."""
    # Create a simple binary mask
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 1  # 4x4 square in the middle
    
    # Encode
    rle = mask_to_rle(mask)
    
    # Decode
    decoded = rle_to_mask(rle)
    
    # Check they match
    np.testing.assert_array_equal(mask, decoded)


def test_mask_to_bbox():
    """Test bounding box extraction from mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 30:70] = 1
    
    bbox = mask_to_bbox(mask)
    
    assert bbox == [30.0, 20.0, 70.0, 50.0]


def test_mask_to_bbox_empty():
    """Test bbox from empty mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    bbox = mask_to_bbox(mask)
    assert bbox == [0, 0, 0, 0]


def test_mask_area():
    """Test mask area calculation."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:10, 0:10] = 1  # 100 pixels
    
    assert mask_area(mask) == 100


def test_mask_iou():
    """Test IoU calculation."""
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[0:50, 0:50] = 1  # 2500 pixels
    
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[25:75, 25:75] = 1  # 2500 pixels
    
    # Intersection: [25:50, 25:50] = 625 pixels
    # Union: 2500 + 2500 - 625 = 4375 pixels
    expected_iou = 625 / 4375
    
    iou = mask_iou(mask1, mask2)
    assert abs(iou - expected_iou) < 0.01


def test_mask_iou_identical():
    """Test IoU of identical masks."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:50, 0:50] = 1
    
    assert mask_iou(mask, mask) == 1.0


def test_mask_iou_disjoint():
    """Test IoU of non-overlapping masks."""
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[0:10, 0:10] = 1
    
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[50:60, 50:60] = 1
    
    assert mask_iou(mask1, mask2) == 0.0


def test_resize_mask():
    """Test mask resizing."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 1
    
    resized = resize_mask(mask, (50, 50))
    
    assert resized.shape == (50, 50)
    # Center should still be 1
    assert resized[25, 25] == 1

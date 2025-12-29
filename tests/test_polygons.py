"""
Tests for polygon utilities.
"""

import numpy as np
import pytest

from core.polygons import (
    mask_to_contours,
    largest_contour,
    simplify_contour,
    contour_to_polygon,
    normalize_polygon,
    polygon_to_flat_list,
    mask_to_polygon,
    mask_to_yolo_polygon,
    polygon_area,
    validate_yolo_polygon,
)


class TestMaskToContours:
    """Tests for mask_to_contours function."""
    
    def test_simple_rectangle(self):
        """Test contour extraction from rectangular mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 1
        
        contours = mask_to_contours(mask)
        
        assert len(contours) >= 1
        assert contours[0].shape[1] == 1  # Nx1x2 format
        assert contours[0].shape[2] == 2
    
    def test_empty_mask(self):
        """Test with empty mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        contours = mask_to_contours(mask)
        
        assert len(contours) == 0
    
    def test_multiple_regions(self):
        """Test with multiple separate regions."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        mask[60:90, 60:90] = 1
        
        contours = mask_to_contours(mask)
        
        assert len(contours) == 2


class TestLargestContour:
    """Tests for largest_contour function."""
    
    def test_selects_largest(self):
        """Test that largest contour is selected."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # Small: 20x20 = 400
        mask[50:90, 50:90] = 1  # Large: 40x40 = 1600
        
        contours = mask_to_contours(mask)
        largest = largest_contour(contours)
        
        assert largest is not None
        # The largest should have roughly the area of the bigger square
        import cv2
        area = cv2.contourArea(largest)
        assert area > 1000  # Should be close to 1600
    
    def test_empty_list(self):
        """Test with empty contour list."""
        result = largest_contour([])
        assert result is None


class TestMaskToPolygon:
    """Tests for mask_to_polygon function."""
    
    def test_rectangle_mask(self):
        """Test polygon extraction from rectangular mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 1
        
        polygon = mask_to_polygon(mask)
        
        assert polygon is not None
        assert len(polygon) >= 4  # Rectangle should have at least 4 points
        
        # Check points are in expected region
        for x, y in polygon:
            assert 25 <= x <= 75
            assert 15 <= y <= 85
    
    def test_empty_mask_returns_none(self):
        """Test that empty mask returns None."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        polygon = mask_to_polygon(mask)
        
        assert polygon is None
    
    def test_simplification(self):
        """Test that simplification reduces points."""
        # Create a circle-ish mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2_available = True
        try:
            import cv2
            cv2.circle(mask, (50, 50), 30, 1, -1)
        except:
            cv2_available = False
        
        if cv2_available:
            polygon_simplified = mask_to_polygon(mask, epsilon_px=3.0)
            polygon_detailed = mask_to_polygon(mask, epsilon_px=0.5)
            
            assert polygon_simplified is not None
            assert polygon_detailed is not None
            # Simplified should have fewer or equal points
            assert len(polygon_simplified) <= len(polygon_detailed)


class TestMaskToYoloPolygon:
    """Tests for mask_to_yolo_polygon function."""
    
    def test_normalized_coordinates(self):
        """Test that coordinates are normalized to [0, 1]."""
        mask = np.zeros((100, 200), dtype=np.uint8)  # H=100, W=200
        mask[25:75, 50:150] = 1
        
        polygon = mask_to_yolo_polygon(mask, width=200, height=100)
        
        assert polygon is not None
        
        # Check all values are in [0, 1]
        for val in polygon:
            assert 0.0 <= val <= 1.0
        
        # Check it's a flat list with even number of values
        assert len(polygon) % 2 == 0
        assert len(polygon) >= 6  # At least 3 points
    
    def test_empty_mask_returns_none(self):
        """Test that empty mask returns None."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        polygon = mask_to_yolo_polygon(mask, width=100, height=100)
        
        assert polygon is None


class TestNormalizePolygon:
    """Tests for normalize_polygon function."""
    
    def test_normalization(self):
        """Test coordinate normalization."""
        polygon = [(100, 50), (200, 50), (200, 100), (100, 100)]
        
        normalized = normalize_polygon(polygon, width=400, height=200)
        
        expected = [(0.25, 0.25), (0.5, 0.25), (0.5, 0.5), (0.25, 0.5)]
        
        for (nx, ny), (ex, ey) in zip(normalized, expected):
            assert abs(nx - ex) < 0.001
            assert abs(ny - ey) < 0.001


class TestPolygonToFlatList:
    """Tests for polygon_to_flat_list function."""
    
    def test_flattening(self):
        """Test polygon flattening."""
        polygon = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
        
        flat = polygon_to_flat_list(polygon)
        
        assert flat == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


class TestPolygonArea:
    """Tests for polygon_area function."""
    
    def test_square_area(self):
        """Test area of a square."""
        # Unit square
        polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
        
        area = polygon_area(polygon)
        
        assert abs(area - 1.0) < 0.001
    
    def test_triangle_area(self):
        """Test area of a triangle."""
        # Triangle with base 2 and height 2
        polygon = [(0, 0), (2, 0), (1, 2)]
        
        area = polygon_area(polygon)
        
        assert abs(area - 2.0) < 0.001  # 0.5 * base * height = 0.5 * 2 * 2 = 2
    
    def test_degenerate_polygon(self):
        """Test with too few points."""
        polygon = [(0, 0), (1, 1)]  # Only 2 points
        
        area = polygon_area(polygon)
        
        assert area == 0.0


class TestValidateYoloPolygon:
    """Tests for validate_yolo_polygon function."""
    
    def test_valid_polygon(self):
        """Test validation of valid polygon."""
        polygon = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # 3 points
        
        is_valid, msg = validate_yolo_polygon(polygon)
        
        assert is_valid
        assert msg == "OK"
    
    def test_none_polygon(self):
        """Test validation of None."""
        is_valid, msg = validate_yolo_polygon(None)
        
        assert not is_valid
        assert "None" in msg
    
    def test_too_few_points(self):
        """Test validation with too few points."""
        polygon = [0.1, 0.2, 0.3, 0.4]  # Only 2 points
        
        is_valid, msg = validate_yolo_polygon(polygon)
        
        assert not is_valid
        assert "few" in msg.lower()
    
    def test_odd_coordinates(self):
        """Test validation with odd number of coordinates."""
        polygon = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # 7 values
        
        is_valid, msg = validate_yolo_polygon(polygon)
        
        assert not is_valid
        assert "Odd" in msg
    
    def test_out_of_range(self):
        """Test validation with out of range values."""
        polygon = [0.1, 0.2, 1.5, 0.4, 0.5, 0.6]  # 1.5 is out of range
        
        is_valid, msg = validate_yolo_polygon(polygon)
        
        assert not is_valid
        assert "range" in msg.lower()

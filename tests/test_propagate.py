"""
Tests for propagation module.
"""

import numpy as np
import pytest

from ml.propagate import generate_candidates, _bbox_iou, BboxCandidate


class TestGenerateCandidates:
    """Tests for generate_candidates function."""
    
    def test_generates_candidates(self):
        """Test that candidates are generated."""
        prev_bbox = [100, 100, 200, 200]  # 100x100 box
        img_w, img_h = 640, 480
        
        candidates = generate_candidates(prev_bbox, img_w, img_h)
        
        assert len(candidates) > 0
        
    def test_candidates_within_bounds(self):
        """Test that all candidates are within image bounds."""
        prev_bbox = [100, 100, 200, 200]
        img_w, img_h = 640, 480
        
        candidates = generate_candidates(prev_bbox, img_w, img_h)
        
        for bbox in candidates:
            x1, y1, x2, y2 = bbox
            assert 0 <= x1 < img_w
            assert 0 <= y1 < img_h
            assert x1 < x2 <= img_w
            assert y1 < y2 <= img_h
    
    def test_candidates_near_previous(self):
        """Test that candidates are near the previous bbox."""
        prev_bbox = [100, 100, 200, 200]
        prev_cx = 150
        prev_cy = 150
        img_w, img_h = 640, 480
        
        candidates = generate_candidates(prev_bbox, img_w, img_h)
        
        # At least some candidates should be close to original
        close_count = 0
        for bbox in candidates:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            if dist < 100:  # Within 100 pixels
                close_count += 1
        
        assert close_count > 0
    
    def test_edge_bbox(self):
        """Test with bbox at image edge."""
        prev_bbox = [0, 0, 50, 50]  # Top-left corner
        img_w, img_h = 640, 480
        
        candidates = generate_candidates(prev_bbox, img_w, img_h)
        
        # Should still generate valid candidates
        assert len(candidates) > 0
        for bbox in candidates:
            x1, y1, x2, y2 = bbox
            assert 0 <= x1 < x2 <= img_w
            assert 0 <= y1 < y2 <= img_h
    
    def test_various_scales(self):
        """Test that candidates include different scales."""
        prev_bbox = [100, 100, 200, 200]
        prev_area = 100 * 100
        img_w, img_h = 640, 480
        
        candidates = generate_candidates(prev_bbox, img_w, img_h, num_scales=5)
        
        # Calculate areas
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in candidates]
        
        # Should have some smaller and some larger than original
        smaller = sum(1 for a in areas if a < prev_area * 0.9)
        larger = sum(1 for a in areas if a > prev_area * 1.1)
        
        assert smaller > 0
        assert larger > 0


class TestBboxIou:
    """Tests for bbox IoU calculation."""
    
    def test_identical_boxes(self):
        """Test IoU of identical boxes is 1."""
        bbox = [100, 100, 200, 200]
        iou = _bbox_iou(bbox, bbox)
        assert abs(iou - 1.0) < 1e-6
    
    def test_no_overlap(self):
        """Test IoU of non-overlapping boxes is 0."""
        bbox1 = [0, 0, 50, 50]
        bbox2 = [100, 100, 150, 150]
        iou = _bbox_iou(bbox1, bbox2)
        assert iou == 0.0
    
    def test_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        bbox1 = [0, 0, 100, 100]
        bbox2 = [50, 50, 150, 150]
        
        # Intersection: 50x50 = 2500
        # Union: 2 * 10000 - 2500 = 17500
        # IoU = 2500 / 17500 ≈ 0.143
        
        iou = _bbox_iou(bbox1, bbox2)
        assert 0.1 < iou < 0.2
    
    def test_contained_box(self):
        """Test IoU when one box contains another."""
        bbox1 = [0, 0, 100, 100]
        bbox2 = [25, 25, 75, 75]
        
        # Intersection = smaller box = 50x50 = 2500
        # Union = larger box = 10000
        # IoU = 0.25
        
        iou = _bbox_iou(bbox1, bbox2)
        assert abs(iou - 0.25) < 0.01


class TestBboxCandidate:
    """Tests for BboxCandidate dataclass."""
    
    def test_creation(self):
        """Test creating a BboxCandidate."""
        candidate = BboxCandidate(
            bbox_xyxy=[100, 100, 200, 200],
            score=0.85,
            source="propagated"
        )
        
        assert candidate.bbox_xyxy == [100, 100, 200, 200]
        assert candidate.score == 0.85
        assert candidate.source == "propagated"
    
    def test_sorting(self):
        """Test sorting candidates by score."""
        candidates = [
            BboxCandidate([0, 0, 50, 50], 0.5),
            BboxCandidate([100, 100, 150, 150], 0.9),
            BboxCandidate([200, 200, 250, 250], 0.7),
        ]
        
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        
        assert sorted_candidates[0].score == 0.9
        assert sorted_candidates[1].score == 0.7
        assert sorted_candidates[2].score == 0.5

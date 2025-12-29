"""
Polygon utilities for converting masks to YOLO-seg format polygons.
"""

import numpy as np
import cv2
from typing import Optional


def mask_to_contours(mask: np.ndarray) -> list[np.ndarray]:
    """
    Extract contours from a binary mask.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 1/255
        
    Returns:
        List of contours, each as Nx1x2 array of points
    """
    # Ensure mask is uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Ensure binary (0 or 255)
    if mask.max() == 1:
        mask = mask * 255
    
    # Find contours
    contours, hierarchy = cv2.findContours(
        mask, 
        cv2.RETR_EXTERNAL,  # Only external contours
        cv2.CHAIN_APPROX_SIMPLE  # Compress horizontal/vertical segments
    )
    
    return list(contours)


def largest_contour(contours: list[np.ndarray]) -> Optional[np.ndarray]:
    """
    Get the largest contour by area.
    
    Args:
        contours: List of contours
        
    Returns:
        Largest contour or None if empty
    """
    if not contours:
        return None
    
    return max(contours, key=cv2.contourArea)


def simplify_contour(contour: np.ndarray, epsilon_px: float = 1.5) -> np.ndarray:
    """
    Simplify a contour using Douglas-Peucker algorithm.
    
    Args:
        contour: Contour as Nx1x2 array
        epsilon_px: Approximation accuracy in pixels
        
    Returns:
        Simplified contour
    """
    return cv2.approxPolyDP(contour, epsilon_px, closed=True)


def contour_to_polygon(contour: np.ndarray) -> list[tuple[float, float]]:
    """
    Convert OpenCV contour to list of (x, y) points.
    
    Args:
        contour: Contour as Nx1x2 array
        
    Returns:
        List of (x, y) tuples
    """
    # Squeeze the middle dimension
    points = contour.squeeze()
    
    # Handle edge case of single point
    if len(points.shape) == 1:
        return [(float(points[0]), float(points[1]))]
    
    return [(float(p[0]), float(p[1])) for p in points]


def normalize_polygon(
    polygon: list[tuple[float, float]], 
    width: int, 
    height: int
) -> list[tuple[float, float]]:
    """
    Normalize polygon coordinates to [0, 1] range.
    
    Args:
        polygon: List of (x, y) pixel coordinates
        width: Image width
        height: Image height
        
    Returns:
        List of (x, y) normalized coordinates
    """
    return [(x / width, y / height) for x, y in polygon]


def polygon_to_flat_list(polygon: list[tuple[float, float]]) -> list[float]:
    """
    Flatten polygon to list of coordinates [x1, y1, x2, y2, ...].
    
    Args:
        polygon: List of (x, y) tuples
        
    Returns:
        Flat list [x1, y1, x2, y2, ...]
    """
    result = []
    for x, y in polygon:
        result.extend([x, y])
    return result


def mask_to_polygon(
    mask: np.ndarray,
    epsilon_px: float = 1.5,
    min_points: int = 3,
) -> Optional[list[tuple[float, float]]]:
    """
    Convert binary mask to polygon (pixel coordinates).
    
    Uses the largest contour and simplifies it.
    
    Args:
        mask: Binary mask (H, W)
        epsilon_px: Simplification tolerance in pixels
        min_points: Minimum number of points required
        
    Returns:
        List of (x, y) pixel coordinates or None if no valid polygon
    """
    contours = mask_to_contours(mask)
    
    if not contours:
        return None
    
    # Get largest contour
    contour = largest_contour(contours)
    
    if contour is None or len(contour) < min_points:
        return None
    
    # Simplify
    simplified = simplify_contour(contour, epsilon_px)
    
    # Check minimum points after simplification
    if len(simplified) < min_points:
        # Try with smaller epsilon
        simplified = simplify_contour(contour, epsilon_px / 2)
        if len(simplified) < min_points:
            # Use original if still too few
            simplified = contour
    
    # Convert to polygon
    polygon = contour_to_polygon(simplified)
    
    # Ensure minimum points
    if len(polygon) < min_points:
        return None
    
    return polygon


def mask_to_yolo_polygon(
    mask: np.ndarray,
    width: int,
    height: int,
    epsilon_px: float = 1.5,
) -> Optional[list[float]]:
    """
    Convert binary mask to YOLO-seg format polygon.
    
    Args:
        mask: Binary mask (H, W)
        width: Image width (for normalization)
        height: Image height (for normalization)
        epsilon_px: Simplification tolerance in pixels
        
    Returns:
        Flat list [x1, y1, x2, y2, ...] with normalized coordinates [0,1]
        or None if no valid polygon
    """
    polygon = mask_to_polygon(mask, epsilon_px)
    
    if polygon is None:
        return None
    
    # Normalize coordinates
    normalized = normalize_polygon(polygon, width, height)
    
    # Clamp to [0, 1] range (in case of rounding issues)
    clamped = [
        (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
        for x, y in normalized
    ]
    
    # Flatten
    return polygon_to_flat_list(clamped)


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    """
    Calculate the area of a polygon using the shoelace formula.
    
    Args:
        polygon: List of (x, y) coordinates
        
    Returns:
        Area (in whatever units the coordinates are in)
    """
    n = len(polygon)
    if n < 3:
        return 0.0
    
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    
    return abs(area) / 2.0


def validate_yolo_polygon(polygon: list[float]) -> tuple[bool, str]:
    """
    Validate a YOLO-seg format polygon.
    
    Args:
        polygon: Flat list [x1, y1, x2, y2, ...]
        
    Returns:
        (is_valid, error_message)
    """
    if polygon is None:
        return False, "Polygon is None"
    
    if len(polygon) < 6:  # At least 3 points
        return False, f"Too few coordinates: {len(polygon)} (need at least 6)"
    
    if len(polygon) % 2 != 0:
        return False, f"Odd number of coordinates: {len(polygon)}"
    
    # Check all values are in [0, 1]
    for i, val in enumerate(polygon):
        if not isinstance(val, (int, float)):
            return False, f"Non-numeric value at index {i}: {val}"
        if val < 0 or val > 1:
            return False, f"Value out of range [0,1] at index {i}: {val}"
    
    return True, "OK"

"""
Core data models for AutoSeg.

Dataclasses representing the main entities in the annotation system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
import json


@dataclass
class Project:
    """Represents an annotation project."""
    id: int
    name: str
    root_dir: str
    db_path: str
    created_at: datetime
    settings_json: Optional[str] = None
    
    @property
    def settings(self) -> dict:
        """Parse settings JSON to dict."""
        if self.settings_json:
            return json.loads(self.settings_json)
        return {}
    
    def with_settings(self, **kwargs) -> 'Project':
        """Return a new Project with updated settings."""
        current = self.settings
        current.update(kwargs)
        return Project(
            id=self.id,
            name=self.name,
            root_dir=self.root_dir,
            db_path=self.db_path,
            created_at=self.created_at,
            settings_json=json.dumps(current)
        )


@dataclass
class Label:
    """Represents a label/class for annotations."""
    id: int
    project_id: int
    name: str
    color_hex: str
    created_at: datetime


@dataclass
class ImageRecord:
    """Represents an image in a project."""
    id: int
    project_id: int
    path: str  # Relative path from project root
    width: int
    height: int
    order_index: int  # For sequential ordering


@dataclass
class Annotation:
    """Represents an annotation on an image."""
    id: int
    image_id: int
    label_id: int
    bbox_xyxy_json: Optional[str] = None  # JSON: [x1, y1, x2, y2] in pixels
    polygon_norm_json: Optional[str] = None  # JSON: [x1, y1, x2, y2, ...] normalized 0-1
    mask_rle_json: Optional[str] = None  # JSON: RLE encoded mask
    source: str = "manual"  # "manual" | "propagated" | "sam"
    confidence: Optional[float] = None
    status: str = "approved"  # "approved" | "needs_review" | "rejected"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @property
    def bbox_xyxy(self) -> Optional[list[float]]:
        """Parse bbox JSON to list."""
        if self.bbox_xyxy_json:
            return json.loads(self.bbox_xyxy_json)
        return None
    
    @property
    def polygon_norm(self) -> Optional[list[float]]:
        """Parse polygon JSON to list."""
        if self.polygon_norm_json:
            return json.loads(self.polygon_norm_json)
        return None
    
    @property
    def mask_rle(self) -> Optional[dict]:
        """Parse mask RLE JSON to dict."""
        if self.mask_rle_json:
            return json.loads(self.mask_rle_json)
        return None


def bbox_to_json(bbox: list[float]) -> str:
    """Convert bbox list to JSON string."""
    return json.dumps(bbox)


def polygon_to_json(polygon: list[float]) -> str:
    """Convert polygon list to JSON string."""
    return json.dumps(polygon)


def mask_rle_to_json(rle: dict) -> str:
    """Convert RLE dict to JSON string."""
    return json.dumps(rle)

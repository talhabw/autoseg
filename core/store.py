"""
ProjectStore - CRUD operations for projects, images, labels, and annotations.
"""

import os
import re
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional
from PIL import Image

from core.db import init_db, migrate_db, get_connection, _set_kv, _get_kv
from core.models import (
    Project, Label, ImageRecord, Annotation,
    bbox_to_json, polygon_to_json, mask_rle_to_json
)


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Default label colors (will cycle through these)
DEFAULT_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F8B500", "#00CED1", "#FF69B4", "#32CD32", "#FFD700",
]


def natural_sort_key(s: str):
    """
    Key function for natural sorting of strings.
    E.g., sorts "img2.jpg" before "img10.jpg"
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', s)
    ]


class ProjectStore:
    """
    Handles all database operations for a project.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize store with database path.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
    
    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = get_connection(self.db_path)
        return self._conn
    
    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def commit(self):
        """Commit current transaction."""
        self.conn.commit()
    
    # ==================== Project Operations ====================
    
    @classmethod
    def create_project(
        cls,
        project_dir: str,
        image_dir: str,
        name: str
    ) -> 'Project':
        """
        Create a new project from an image directory.
        
        Args:
            project_dir: Directory where project files will be stored
            image_dir: Directory containing images to import
            name: Name of the project
            
        Returns:
            Created Project instance
        """
        project_dir = os.path.abspath(project_dir)
        image_dir = os.path.abspath(image_dir)
        
        # Create project directory if needed
        os.makedirs(project_dir, exist_ok=True)
        
        # Database path
        db_path = os.path.join(project_dir, "autoseg.db")
        
        # Initialize database
        init_db(db_path)
        
        # Create store instance
        store = cls(db_path)
        
        try:
            # Create project record
            cursor = store.conn.execute(
                """
                INSERT INTO projects (name, root_dir, db_path, settings_json)
                VALUES (?, ?, ?, ?)
                """,
                (name, project_dir, db_path, json.dumps({"image_dir": image_dir}))
            )
            project_id = cursor.lastrowid
            
            # Scan and import images
            store._import_images(project_id, image_dir)
            
            store.commit()
            
            # Fetch and return the project
            return store._get_project(project_id)
        finally:
            store.close()
    
    @classmethod
    def load_project(cls, project_dir: str) -> 'Project':
        """
        Load an existing project.
        
        Args:
            project_dir: Directory containing the project
            
        Returns:
            Loaded Project instance
        """
        project_dir = os.path.abspath(project_dir)
        db_path = os.path.join(project_dir, "autoseg.db")
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"No project database found at {db_path}")
        
        # Run any pending migrations
        migrate_db(db_path)
        
        # Load project
        store = cls(db_path)
        try:
            cursor = store.conn.execute(
                "SELECT * FROM projects ORDER BY id LIMIT 1"
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError("No project found in database")
            
            return store._row_to_project(row)
        finally:
            store.close()
    
    def _get_project(self, project_id: int) -> Project:
        """Get project by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM projects WHERE id = ?",
            (project_id,)
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Project {project_id} not found")
        return self._row_to_project(row)
    
    def _row_to_project(self, row: sqlite3.Row) -> Project:
        """Convert database row to Project object."""
        return Project(
            id=row['id'],
            name=row['name'],
            root_dir=row['root_dir'],
            db_path=row['db_path'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
            settings_json=row['settings_json']
        )
    
    def _import_images(self, project_id: int, image_dir: str) -> None:
        """Import all images from a directory."""
        image_dir = Path(image_dir)
        
        # Find all image files
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        # Sort naturally
        image_files = sorted(set(image_files), key=lambda p: natural_sort_key(p.name))
        
        # Import each image
        for idx, img_path in enumerate(image_files):
            # Get image dimensions
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Warning: Could not read image {img_path}: {e}")
                continue
            
            # Store relative path from image_dir
            rel_path = str(img_path.relative_to(image_dir.parent) if image_dir.parent != img_path.parent else img_path.name)
            
            self.conn.execute(
                """
                INSERT INTO images (project_id, path, width, height, order_index)
                VALUES (?, ?, ?, ?, ?)
                """,
                (project_id, str(img_path), width, height, idx)
            )
    
    # ==================== Image Operations ====================
    
    def list_images(self, project_id: int) -> list[ImageRecord]:
        """
        List all images in a project.
        
        Args:
            project_id: ID of the project
            
        Returns:
            List of ImageRecord objects ordered by order_index
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM images 
            WHERE project_id = ? 
            ORDER BY order_index
            """,
            (project_id,)
        )
        return [self._row_to_image(row) for row in cursor.fetchall()]
    
    def get_image_by_index(self, project_id: int, order_index: int) -> Optional[ImageRecord]:
        """
        Get image by its order index.
        
        Args:
            project_id: ID of the project
            order_index: Order index of the image
            
        Returns:
            ImageRecord or None if not found
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM images 
            WHERE project_id = ? AND order_index = ?
            """,
            (project_id, order_index)
        )
        row = cursor.fetchone()
        return self._row_to_image(row) if row else None
    
    def get_image_by_id(self, image_id: int) -> Optional[ImageRecord]:
        """Get image by its ID."""
        cursor = self.conn.execute(
            "SELECT * FROM images WHERE id = ?",
            (image_id,)
        )
        row = cursor.fetchone()
        return self._row_to_image(row) if row else None
    
    def get_image_count(self, project_id: int) -> int:
        """Get total number of images in project."""
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM images WHERE project_id = ?",
            (project_id,)
        )
        return cursor.fetchone()[0]
    
    def _row_to_image(self, row: sqlite3.Row) -> ImageRecord:
        """Convert database row to ImageRecord object."""
        return ImageRecord(
            id=row['id'],
            project_id=row['project_id'],
            path=row['path'],
            width=row['width'],
            height=row['height'],
            order_index=row['order_index']
        )
    
    # ==================== Label Operations ====================
    
    def upsert_label(
        self,
        project_id: int,
        name: str,
        color_hex: Optional[str] = None
    ) -> Label:
        """
        Create or update a label.
        
        Args:
            project_id: ID of the project
            name: Name of the label
            color_hex: Color in hex format (e.g., "#FF0000")
            
        Returns:
            Created or existing Label
        """
        # Check if label exists
        cursor = self.conn.execute(
            "SELECT * FROM labels WHERE project_id = ? AND name = ?",
            (project_id, name)
        )
        existing = cursor.fetchone()
        
        if existing:
            if color_hex and existing['color_hex'] != color_hex:
                # Update color
                self.conn.execute(
                    "UPDATE labels SET color_hex = ? WHERE id = ?",
                    (color_hex, existing['id'])
                )
                self.commit()
            return self._row_to_label(existing)
        
        # Generate color if not provided
        if not color_hex:
            # Get count of existing labels for color cycling
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM labels WHERE project_id = ?",
                (project_id,)
            )
            count = cursor.fetchone()[0]
            color_hex = DEFAULT_COLORS[count % len(DEFAULT_COLORS)]
        
        # Create new label
        cursor = self.conn.execute(
            """
            INSERT INTO labels (project_id, name, color_hex)
            VALUES (?, ?, ?)
            """,
            (project_id, name, color_hex)
        )
        self.commit()
        
        return self.get_label_by_id(cursor.lastrowid)
    
    def list_labels(self, project_id: int) -> list[Label]:
        """
        List all labels in a project.
        
        Args:
            project_id: ID of the project
            
        Returns:
            List of Label objects
        """
        cursor = self.conn.execute(
            "SELECT * FROM labels WHERE project_id = ? ORDER BY name",
            (project_id,)
        )
        return [self._row_to_label(row) for row in cursor.fetchall()]
    
    def get_label_by_id(self, label_id: int) -> Optional[Label]:
        """Get label by its ID."""
        cursor = self.conn.execute(
            "SELECT * FROM labels WHERE id = ?",
            (label_id,)
        )
        row = cursor.fetchone()
        return self._row_to_label(row) if row else None
    
    def get_label_by_name(self, project_id: int, name: str) -> Optional[Label]:
        """Get label by its name."""
        cursor = self.conn.execute(
            "SELECT * FROM labels WHERE project_id = ? AND name = ?",
            (project_id, name)
        )
        row = cursor.fetchone()
        return self._row_to_label(row) if row else None
    
    def _row_to_label(self, row: sqlite3.Row) -> Label:
        """Convert database row to Label object."""
        return Label(
            id=row['id'],
            project_id=row['project_id'],
            name=row['name'],
            color_hex=row['color_hex'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now()
        )
    
    # ==================== Annotation Operations ====================
    
    def create_annotation(
        self,
        image_id: int,
        label_id: int,
        bbox_xyxy: list[float],
        source: str = "manual",
        status: str = "approved",
        confidence: Optional[float] = None,
        mask_rle: Optional[dict] = None,
        polygon_norm: Optional[list[float]] = None,
    ) -> Annotation:
        """
        Create a new annotation.
        
        Args:
            image_id: ID of the image
            label_id: ID of the label
            bbox_xyxy: Bounding box [x1, y1, x2, y2] in pixels
            source: Source of annotation ("manual", "propagated", "sam")
            status: Status ("approved", "needs_review", "rejected")
            confidence: Confidence score (0-1)
            mask_rle: Optional RLE encoded mask
            polygon_norm: Optional normalized polygon
            
        Returns:
            Created Annotation
        """
        cursor = self.conn.execute(
            """
            INSERT INTO annotations 
            (image_id, label_id, bbox_xyxy_json, source, status, confidence, mask_rle_json, polygon_norm_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id, label_id, bbox_to_json(bbox_xyxy), source, status, confidence,
                mask_rle_to_json(mask_rle) if mask_rle else None,
                polygon_to_json(polygon_norm) if polygon_norm else None,
            )
        )
        self.commit()
        
        return self.get_annotation_by_id(cursor.lastrowid)
    
    def update_annotation(self, annotation_id: int, **fields) -> Annotation:
        """
        Update an annotation.
        
        Args:
            annotation_id: ID of the annotation
            **fields: Fields to update (bbox_xyxy, polygon_norm, mask_rle, 
                      label_id, source, status, confidence)
                      
        Returns:
            Updated Annotation
        """
        allowed_fields = {
            'bbox_xyxy', 'polygon_norm', 'mask_rle',
            'label_id', 'source', 'status', 'confidence'
        }
        
        updates = []
        values = []
        
        for key, value in fields.items():
            if key not in allowed_fields:
                continue
            
            if key == 'bbox_xyxy':
                updates.append("bbox_xyxy_json = ?")
                values.append(bbox_to_json(value) if value else None)
            elif key == 'polygon_norm':
                updates.append("polygon_norm_json = ?")
                values.append(polygon_to_json(value) if value else None)
            elif key == 'mask_rle':
                updates.append("mask_rle_json = ?")
                values.append(mask_rle_to_json(value) if value else None)
            else:
                updates.append(f"{key} = ?")
                values.append(value)
        
        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            values.append(annotation_id)
            
            self.conn.execute(
                f"UPDATE annotations SET {', '.join(updates)} WHERE id = ?",
                values
            )
            self.commit()
        
        return self.get_annotation_by_id(annotation_id)
    
    def delete_annotation(self, annotation_id: int) -> None:
        """
        Delete an annotation.
        
        Args:
            annotation_id: ID of the annotation to delete
        """
        self.conn.execute(
            "DELETE FROM annotations WHERE id = ?",
            (annotation_id,)
        )
        self.commit()
    
    def list_annotations(self, image_id: int) -> list[Annotation]:
        """
        List all annotations for an image.
        
        Args:
            image_id: ID of the image
            
        Returns:
            List of Annotation objects
        """
        cursor = self.conn.execute(
            "SELECT * FROM annotations WHERE image_id = ? ORDER BY id",
            (image_id,)
        )
        return [self._row_to_annotation(row) for row in cursor.fetchall()]
    
    def get_annotation_by_id(self, annotation_id: int) -> Optional[Annotation]:
        """Get annotation by its ID."""
        cursor = self.conn.execute(
            "SELECT * FROM annotations WHERE id = ?",
            (annotation_id,)
        )
        row = cursor.fetchone()
        return self._row_to_annotation(row) if row else None
    
    def list_annotations_by_status(
        self,
        project_id: int,
        status: str
    ) -> list[tuple[Annotation, ImageRecord]]:
        """
        List annotations by status with their images.
        
        Args:
            project_id: ID of the project
            status: Status to filter by
            
        Returns:
            List of (Annotation, ImageRecord) tuples
        """
        cursor = self.conn.execute(
            """
            SELECT a.*, i.* FROM annotations a
            JOIN images i ON a.image_id = i.id
            WHERE i.project_id = ? AND a.status = ?
            ORDER BY i.order_index, a.id
            """,
            (project_id, status)
        )
        
        results = []
        for row in cursor.fetchall():
            # Split row into annotation and image parts
            ann = Annotation(
                id=row['id'],
                image_id=row['image_id'],
                label_id=row['label_id'],
                bbox_xyxy_json=row['bbox_xyxy_json'],
                polygon_norm_json=row['polygon_norm_json'],
                mask_rle_json=row['mask_rle_json'],
                source=row['source'],
                confidence=row['confidence'],
                status=row['status'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
            )
            img = ImageRecord(
                id=row[11],  # After annotation fields
                project_id=row[12],
                path=row[13],
                width=row[14],
                height=row[15],
                order_index=row[16]
            )
            results.append((ann, img))
        
        return results
    
    def _row_to_annotation(self, row: sqlite3.Row) -> Annotation:
        """Convert database row to Annotation object."""
        return Annotation(
            id=row['id'],
            image_id=row['image_id'],
            label_id=row['label_id'],
            bbox_xyxy_json=row['bbox_xyxy_json'],
            polygon_norm_json=row['polygon_norm_json'],
            mask_rle_json=row['mask_rle_json'],
            source=row['source'],
            confidence=row['confidence'],
            status=row['status'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
        )
    
    # ==================== KV Store Operations ====================
    
    def set_setting(self, key: str, value: str) -> None:
        """Set a setting value."""
        _set_kv(self.conn, key, value)
        self.commit()
    
    def get_setting(self, key: str) -> Optional[str]:
        """Get a setting value."""
        return _get_kv(self.conn, key)

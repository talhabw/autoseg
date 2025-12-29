"""
YOLO-seg format export.

Exports annotations to Ultralytics YOLO segmentation format:
- images/train/*.jpg
- images/val/*.jpg  
- labels/train/*.txt
- labels/val/*.txt
- data.yaml
"""

import os
import shutil
import random
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from core.store import ProjectStore
from core.models import Project, Label, ImageRecord, Annotation


@dataclass
class ExportReport:
    """Report of export operation."""
    total_images: int
    train_images: int
    val_images: int
    total_annotations: int
    annotations_with_polygon: int
    annotations_without_polygon: int
    skipped_images: int  # Images with no valid annotations
    labels: list[str]
    output_dir: str
    warnings: list[str]


def export_yolo_seg(
    project_dir: str,
    out_dir: str,
    split: dict[str, float] = None,
    seed: int = 42,
    approved_only: bool = True
) -> ExportReport:
    """
    Export project to YOLO segmentation format.
    
    Args:
        project_dir: Path to the project directory
        out_dir: Output directory for the export
        split: Train/val split ratios, e.g., {"train": 0.8, "val": 0.2}
        seed: Random seed for reproducible splits
        approved_only: Only export approved annotations
        
    Returns:
        ExportReport with statistics
    """
    if split is None:
        split = {"train": 0.8, "val": 0.2}
    
    # Load project
    project = ProjectStore.load_project(project_dir)
    store = ProjectStore(project.db_path)
    
    try:
        return _do_export(store, project, out_dir, split, seed, approved_only)
    finally:
        store.close()


def _do_export(
    store: ProjectStore,
    project: Project,
    out_dir: str,
    split: dict[str, float],
    seed: int,
    approved_only: bool
) -> ExportReport:
    """Perform the actual export."""
    
    warnings = []
    
    # Create output directories
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for subset in split.keys():
        (out_path / "images" / subset).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / subset).mkdir(parents=True, exist_ok=True)
    
    # Get labels and create mapping
    labels = store.list_labels(project.id)
    label_to_idx = {label.id: idx for idx, label in enumerate(labels)}
    label_names = [label.name for label in labels]
    
    if not labels:
        warnings.append("No labels defined in project")
    
    # Get all images
    images = store.list_images(project.id)
    
    if not images:
        warnings.append("No images in project")
        return ExportReport(
            total_images=0,
            train_images=0,
            val_images=0,
            total_annotations=0,
            annotations_with_polygon=0,
            annotations_without_polygon=0,
            skipped_images=0,
            labels=label_names,
            output_dir=out_dir,
            warnings=warnings
        )
    
    # Shuffle and split images
    random.seed(seed)
    image_indices = list(range(len(images)))
    random.shuffle(image_indices)
    
    # Calculate split points
    split_points = {}
    cumulative = 0
    for subset, ratio in split.items():
        start = cumulative
        cumulative += int(len(images) * ratio)
        split_points[subset] = (start, cumulative)
    
    # Ensure last split gets remaining images
    last_subset = list(split.keys())[-1]
    split_points[last_subset] = (split_points[last_subset][0], len(images))
    
    # Assign images to splits
    image_splits = {}
    for subset, (start, end) in split_points.items():
        for idx in image_indices[start:end]:
            image_splits[images[idx].id] = subset
    
    # Export each image
    stats = {
        "total_annotations": 0,
        "annotations_with_polygon": 0,
        "annotations_without_polygon": 0,
        "skipped_images": 0,
        "train_images": 0,
        "val_images": 0,
    }
    
    for image in images:
        subset = image_splits[image.id]
        
        # Get annotations
        annotations = store.list_annotations(image.id)
        
        if approved_only:
            annotations = [a for a in annotations if a.status == "approved"]
        
        # Filter to annotations with valid data
        valid_annotations = []
        for ann in annotations:
            stats["total_annotations"] += 1
            
            if ann.polygon_norm:
                stats["annotations_with_polygon"] += 1
                valid_annotations.append(ann)
            elif ann.bbox_xyxy:
                # For now, we'll still include bbox-only annotations
                # They'll be exported as bbox (will need polygon later)
                stats["annotations_without_polygon"] += 1
                valid_annotations.append(ann)
        
        if not valid_annotations:
            stats["skipped_images"] += 1
            continue
        
        # Copy image
        src_path = Path(image.path)
        if not src_path.exists():
            warnings.append(f"Image not found: {image.path}")
            stats["skipped_images"] += 1
            continue
        
        # Determine output filename (keep original extension)
        out_name = f"{image.id:06d}{src_path.suffix}"
        img_out_path = out_path / "images" / subset / out_name
        
        shutil.copy2(src_path, img_out_path)
        
        # Write label file
        label_out_path = out_path / "labels" / subset / f"{image.id:06d}.txt"
        
        with open(label_out_path, 'w') as f:
            for ann in valid_annotations:
                class_idx = label_to_idx.get(ann.label_id, 0)
                
                if ann.polygon_norm:
                    # Full polygon format: class_idx x1 y1 x2 y2 ...
                    polygon = ann.polygon_norm
                    coords_str = ' '.join(f"{v:.6f}" for v in polygon)
                    f.write(f"{class_idx} {coords_str}\n")
                elif ann.bbox_xyxy:
                    # Convert bbox to normalized polygon (4 corners)
                    bbox = ann.bbox_xyxy
                    x1, y1, x2, y2 = bbox
                    # Normalize to 0-1
                    x1_n = x1 / image.width
                    y1_n = y1 / image.height
                    x2_n = x2 / image.width
                    y2_n = y2 / image.height
                    # Clamp to valid range
                    x1_n = max(0, min(1, x1_n))
                    y1_n = max(0, min(1, y1_n))
                    x2_n = max(0, min(1, x2_n))
                    y2_n = max(0, min(1, y2_n))
                    # Write as rectangle polygon (4 points)
                    f.write(f"{class_idx} {x1_n:.6f} {y1_n:.6f} {x2_n:.6f} {y1_n:.6f} {x2_n:.6f} {y2_n:.6f} {x1_n:.6f} {y2_n:.6f}\n")
        
        # Update stats
        if subset == "train":
            stats["train_images"] += 1
        elif subset == "val":
            stats["val_images"] += 1
    
    # Write data.yaml
    data_yaml_path = out_path / "data.yaml"
    with open(data_yaml_path, 'w') as f:
        f.write(f"# AutoSeg export - {project.name}\n")
        f.write(f"path: {out_path.absolute()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"\n")
        f.write(f"# Classes\n")
        f.write(f"names:\n")
        for idx, name in enumerate(label_names):
            f.write(f"  {idx}: {name}\n")
    
    # Add warning if many annotations lack polygons
    if stats["annotations_without_polygon"] > 0:
        pct = stats["annotations_without_polygon"] / max(1, stats["total_annotations"]) * 100
        warnings.append(
            f"{stats['annotations_without_polygon']} annotations ({pct:.0f}%) exported as bbox rectangles "
            f"(no segmentation polygon). Run SAM segmentation to generate proper polygons."
        )
    
    return ExportReport(
        total_images=len(images),
        train_images=stats["train_images"],
        val_images=stats["val_images"],
        total_annotations=stats["total_annotations"],
        annotations_with_polygon=stats["annotations_with_polygon"],
        annotations_without_polygon=stats["annotations_without_polygon"],
        skipped_images=stats["skipped_images"],
        labels=label_names,
        output_dir=out_dir,
        warnings=warnings
    )


def verify_yolo_seg_export(out_dir: str) -> tuple[bool, list[str]]:
    """
    Verify a YOLO-seg export is valid.
    
    Args:
        out_dir: Path to the export directory
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    out_path = Path(out_dir)
    
    # Check data.yaml exists
    data_yaml = out_path / "data.yaml"
    if not data_yaml.exists():
        errors.append("data.yaml not found")
        return False, errors
    
    # Check directories exist
    for subset in ["train", "val"]:
        img_dir = out_path / "images" / subset
        lbl_dir = out_path / "labels" / subset
        
        if not img_dir.exists():
            errors.append(f"Missing directory: images/{subset}")
        if not lbl_dir.exists():
            errors.append(f"Missing directory: labels/{subset}")
    
    if errors:
        return False, errors
    
    # Verify label files
    for subset in ["train", "val"]:
        lbl_dir = out_path / "labels" / subset
        
        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    tokens = line.split()
                    
                    # Must have at least 7 tokens: class + 3 points (6 coords)
                    if len(tokens) < 7:
                        errors.append(
                            f"{lbl_file.name}:{line_num}: Too few tokens ({len(tokens)}), "
                            f"need at least 7 (class + 3 points)"
                        )
                        continue
                    
                    # Token count must be odd (class_idx + pairs of x,y)
                    if len(tokens) % 2 == 0:
                        errors.append(
                            f"{lbl_file.name}:{line_num}: Token count must be odd, got {len(tokens)}"
                        )
                        continue
                    
                    # First token should be integer class index
                    try:
                        class_idx = int(tokens[0])
                        if class_idx < 0:
                            errors.append(
                                f"{lbl_file.name}:{line_num}: Invalid class index {class_idx}"
                            )
                    except ValueError:
                        errors.append(
                            f"{lbl_file.name}:{line_num}: Class index not an integer: {tokens[0]}"
                        )
                        continue
                    
                    # Remaining tokens should be floats in [0, 1]
                    for i, tok in enumerate(tokens[1:], 1):
                        try:
                            val = float(tok)
                            if val < 0 or val > 1:
                                errors.append(
                                    f"{lbl_file.name}:{line_num}: Coordinate {i} out of range [0,1]: {val}"
                                )
                        except ValueError:
                            errors.append(
                                f"{lbl_file.name}:{line_num}: Invalid float: {tok}"
                            )
    
    return len(errors) == 0, errors

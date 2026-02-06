"""
Bounding box format exports.

Supports:
- YOLO-detect: Ultralytics YOLO detection format (normalized xywh)
- COCO: COCO JSON format for object detection
"""

import os
import shutil
import random
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from core.store import ProjectStore
from core.models import Project, Label, ImageRecord, Annotation


@dataclass
class BboxExportReport:
    """Report of bbox export operation."""

    format: str  # "yolo-detect" or "coco"
    total_images: int
    train_images: int
    val_images: int
    total_annotations: int
    skipped_images: int
    labels: list[str]
    output_dir: str
    warnings: list[str]


def export_yolo_detect(
    project_dir: str,
    out_dir: str,
    split: Optional[dict[str, float]] = None,
    seed: int = 42,
    approved_only: bool = True,
    include_negative: bool = False,
    labels_only: bool = False,
    labels_colocate: bool = False,
) -> BboxExportReport:
    """
    Export project to YOLO detection format (bbox only).

    Format: class_idx center_x center_y width height (normalized 0-1)

    Args:
        project_dir: Path to the project directory
        out_dir: Output directory for the export
        split: Train/val split ratios, e.g., {"train": 0.8, "val": 0.2}
        seed: Random seed for reproducible splits
        approved_only: Only export approved annotations
        include_negative: Include images without annotations as negative examples

    Returns:
        BboxExportReport with statistics
    """
    if split is None:
        split = {"train": 0.8, "val": 0.2}

    project = ProjectStore.load_project(project_dir)
    store = ProjectStore(project.db_path)

    try:
        return _do_yolo_detect_export(
            store,
            project,
            out_dir,
            split,
            seed,
            approved_only,
            include_negative,
            labels_only,
            labels_colocate,
        )
    finally:
        store.close()


def _do_yolo_detect_export(
    store: ProjectStore,
    project: Project,
    out_dir: str,
    split: dict[str, float],
    seed: int,
    approved_only: bool,
    include_negative: bool,
    labels_only: bool = False,
    labels_colocate: bool = False,
) -> BboxExportReport:
    """Perform YOLO-detect export."""

    warnings = []
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Create directories (skip images dir if labels_only)
    if not labels_only:
        for subset in split.keys():
            (out_path / "images" / subset).mkdir(parents=True, exist_ok=True)
            (out_path / "labels" / subset).mkdir(parents=True, exist_ok=True)
    elif not labels_colocate:
        # Labels-only: single labels folder, no train/val split
        (out_path / "labels").mkdir(parents=True, exist_ok=True)

    labels = store.list_labels(project.id)
    label_to_idx = {label.id: idx for idx, label in enumerate(labels)}
    label_names = [label.name for label in labels]

    if not labels:
        warnings.append("No labels defined in project")

    images = store.list_images(project.id)

    if not images:
        warnings.append("No images in project")
        return BboxExportReport(
            format="yolo-detect",
            total_images=0,
            train_images=0,
            val_images=0,
            total_annotations=0,
            skipped_images=0,
            labels=label_names,
            output_dir=out_dir,
            warnings=warnings,
        )

    # Shuffle and split
    random.seed(seed)
    image_indices = list(range(len(images)))
    random.shuffle(image_indices)

    split_points = {}
    cumulative = 0
    for subset, ratio in split.items():
        start = cumulative
        cumulative += int(len(images) * ratio)
        split_points[subset] = (start, cumulative)

    last_subset = list(split.keys())[-1]
    split_points[last_subset] = (split_points[last_subset][0], len(images))

    image_splits = {}
    for subset, (start, end) in split_points.items():
        for idx in image_indices[start:end]:
            image_splits[images[idx].id] = subset

    stats = {
        "total_annotations": 0,
        "skipped_images": 0,
        "train_images": 0,
        "val_images": 0,
    }

    for image in images:
        subset = image_splits[image.id]
        annotations = store.list_annotations(image.id)

        if approved_only:
            annotations = [a for a in annotations if a.status == "approved"]

        # Get annotations with bboxes
        valid_annotations = [a for a in annotations if a.bbox_xyxy]
        stats["total_annotations"] += len(valid_annotations)

        if not valid_annotations:
            if include_negative:
                # Include as negative example with empty label file
                src_path = Path(image.path)
                if not src_path.exists():
                    warnings.append(f"Image not found: {image.path}")
                    stats["skipped_images"] += 1
                    continue

                # Copy image (unless labels_only mode)
                if not labels_only:
                    out_name = f"{image.id:06d}{src_path.suffix}"
                    img_out_path = out_path / "images" / subset / out_name
                    shutil.copy2(src_path, img_out_path)

                # Write empty label file (use original name if labels_only)
                if labels_only:
                    label_stem = src_path.stem
                    if labels_colocate:
                        label_out_path = src_path.parent / f"{label_stem}.txt"
                    else:
                        label_out_path = out_path / "labels" / f"{label_stem}.txt"
                else:
                    label_out_path = (
                        out_path / "labels" / subset / f"{image.id:06d}.txt"
                    )
                label_out_path.touch()  # Create empty file

                if subset == "train":
                    stats["train_images"] += 1
                elif subset == "val":
                    stats["val_images"] += 1
                continue
            else:
                stats["skipped_images"] += 1
                continue

        # Copy image (unless labels_only mode)
        src_path = Path(image.path)
        if not src_path.exists():
            warnings.append(f"Image not found: {image.path}")
            stats["skipped_images"] += 1
            continue

        if not labels_only:
            out_name = f"{image.id:06d}{src_path.suffix}"
            img_out_path = out_path / "images" / subset / out_name
            shutil.copy2(src_path, img_out_path)

        # Write label file (use original name if labels_only)
        if labels_only:
            label_stem = src_path.stem
            if labels_colocate:
                label_out_path = src_path.parent / f"{label_stem}.txt"
            else:
                label_out_path = out_path / "labels" / f"{label_stem}.txt"
        else:
            label_out_path = out_path / "labels" / subset / f"{image.id:06d}.txt"

        with open(label_out_path, "w") as f:
            for ann in valid_annotations:
                class_idx = label_to_idx.get(ann.label_id, 0)
                bbox = ann.bbox_xyxy
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox

                # Convert to center + width/height, normalized
                cx = ((x1 + x2) / 2) / image.width
                cy = ((y1 + y2) / 2) / image.height
                w = (x2 - x1) / image.width
                h = (y2 - y1) / image.height

                # Clamp to valid range
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                w = max(0, min(1, w))
                h = max(0, min(1, h))

                f.write(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        if subset == "train":
            stats["train_images"] += 1
        elif subset == "val":
            stats["val_images"] += 1

    # Write data.yaml (skip if labels_only)
    if not labels_only:
        data_yaml_path = out_path / "data.yaml"
        with open(data_yaml_path, "w") as f:
            f.write(f"# AutoSeg bbox export - {project.name}\n")
            f.write(f"path: {out_path.absolute()}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n")
            f.write(f"\n")
            f.write(f"# Classes\n")
            f.write(f"names:\n")
            for idx, name in enumerate(label_names):
                f.write(f"  {idx}: {name}\n")

    return BboxExportReport(
        format="yolo-detect",
        total_images=len(images),
        train_images=stats["train_images"],
        val_images=stats["val_images"],
        total_annotations=stats["total_annotations"],
        skipped_images=stats["skipped_images"],
        labels=label_names,
        output_dir=out_dir,
        warnings=warnings,
    )


def export_coco(
    project_dir: str,
    out_dir: str,
    split: Optional[dict[str, float]] = None,
    seed: int = 42,
    approved_only: bool = True,
    include_segmentation: bool = False,
    include_negative: bool = False,
    labels_only: bool = False,
) -> BboxExportReport:
    """
    Export project to COCO JSON format.

    Creates:
    - images/train/*.jpg
    - images/val/*.jpg
    - annotations/train.json
    - annotations/val.json

    Args:
        project_dir: Path to the project directory
        out_dir: Output directory for the export
        split: Train/val split ratios
        seed: Random seed for reproducible splits
        approved_only: Only export approved annotations
        include_segmentation: Include polygon segmentation if available
        include_negative: Include images without annotations as negative examples
        labels_only: Export only annotation files, skip copying images

    Returns:
        BboxExportReport with statistics
    """
    if split is None:
        split = {"train": 0.8, "val": 0.2}

    project = ProjectStore.load_project(project_dir)
    store = ProjectStore(project.db_path)

    try:
        return _do_coco_export(
            store,
            project,
            out_dir,
            split,
            seed,
            approved_only,
            include_segmentation,
            include_negative,
            labels_only,
        )
    finally:
        store.close()


def _do_coco_export(
    store: ProjectStore,
    project: Project,
    out_dir: str,
    split: dict[str, float],
    seed: int,
    approved_only: bool,
    include_segmentation: bool,
    include_negative: bool,
    labels_only: bool,
) -> BboxExportReport:
    """Perform COCO format export."""

    warnings = []
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "annotations").mkdir(parents=True, exist_ok=True)

    if not labels_only:
        for subset in split.keys():
            (out_path / "images" / subset).mkdir(parents=True, exist_ok=True)

    labels = store.list_labels(project.id)
    label_to_idx = {
        label.id: idx + 1 for idx, label in enumerate(labels)
    }  # COCO uses 1-indexed
    label_names = [label.name for label in labels]

    if not labels:
        warnings.append("No labels defined in project")

    images = store.list_images(project.id)

    if not images:
        warnings.append("No images in project")
        return BboxExportReport(
            format="coco",
            total_images=0,
            train_images=0,
            val_images=0,
            total_annotations=0,
            skipped_images=0,
            labels=label_names,
            output_dir=out_dir,
            warnings=warnings,
        )

    # Shuffle and split
    random.seed(seed)
    image_indices = list(range(len(images)))
    random.shuffle(image_indices)

    split_points = {}
    cumulative = 0
    for subset, ratio in split.items():
        start = cumulative
        cumulative += int(len(images) * ratio)
        split_points[subset] = (start, cumulative)

    last_subset = list(split.keys())[-1]
    split_points[last_subset] = (split_points[last_subset][0], len(images))

    image_splits = {}
    for subset, (start, end) in split_points.items():
        for idx in image_indices[start:end]:
            image_splits[images[idx].id] = subset

    # Build COCO structures per subset
    coco_data = {
        subset: {
            "info": {
                "description": f"AutoSeg export - {project.name}",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "categories": [
                {"id": idx + 1, "name": label.name, "supercategory": ""}
                for idx, label in enumerate(labels)
            ],
            "images": [],
            "annotations": [],
        }
        for subset in split.keys()
    }

    stats = {
        "total_annotations": 0,
        "skipped_images": 0,
        "train_images": 0,
        "val_images": 0,
    }

    annotation_id = 1

    for image in images:
        subset = image_splits[image.id]
        annotations = store.list_annotations(image.id)

        if approved_only:
            annotations = [a for a in annotations if a.status == "approved"]

        valid_annotations = [a for a in annotations if a.bbox_xyxy]

        if not valid_annotations:
            if include_negative:
                # Include as negative example (image with no annotations)
                src_path = Path(image.path)
                if not src_path.exists():
                    warnings.append(f"Image not found: {image.path}")
                    stats["skipped_images"] += 1
                    continue

                out_name = f"{image.id:06d}{src_path.suffix}"
                file_name = str(src_path) if labels_only else f"{subset}/{out_name}"

                if not labels_only:
                    img_out_path = out_path / "images" / subset / out_name
                    shutil.copy2(src_path, img_out_path)

                # Add image to COCO (with no annotations)
                coco_data[subset]["images"].append(
                    {
                        "id": image.id,
                        "file_name": file_name,
                        "width": image.width,
                        "height": image.height,
                    }
                )

                if subset == "train":
                    stats["train_images"] += 1
                elif subset == "val":
                    stats["val_images"] += 1
                continue
            else:
                stats["skipped_images"] += 1
                continue

        # Copy image
        src_path = Path(image.path)
        if not src_path.exists():
            warnings.append(f"Image not found: {image.path}")
            stats["skipped_images"] += 1
            continue

        out_name = f"{image.id:06d}{src_path.suffix}"
        file_name = str(src_path) if labels_only else f"{subset}/{out_name}"

        if not labels_only:
            img_out_path = out_path / "images" / subset / out_name
            shutil.copy2(src_path, img_out_path)

        # Add image to COCO
        coco_data[subset]["images"].append(
            {
                "id": image.id,
                "file_name": file_name,
                "width": image.width,
                "height": image.height,
            }
        )

        # Add annotations
        for ann in valid_annotations:
            bbox = ann.bbox_xyxy
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox

            # COCO bbox format: [x, y, width, height]
            coco_bbox = [x1, y1, x2 - x1, y2 - y1]
            area = (x2 - x1) * (y2 - y1)

            coco_ann = {
                "id": annotation_id,
                "image_id": image.id,
                "category_id": label_to_idx.get(ann.label_id, 1),
                "bbox": coco_bbox,
                "area": area,
                "iscrowd": 0,
            }

            # Optionally include segmentation polygon
            if include_segmentation and ann.polygon_norm:
                polygon = ann.polygon_norm
                # Convert normalized coords to pixel coords
                pixel_polygon = []
                for i in range(0, len(polygon), 2):
                    px = polygon[i] * image.width
                    py = polygon[i + 1] * image.height
                    pixel_polygon.extend([px, py])
                coco_ann["segmentation"] = [pixel_polygon]

            coco_data[subset]["annotations"].append(coco_ann)
            annotation_id += 1
            stats["total_annotations"] += 1

        if subset == "train":
            stats["train_images"] += 1
        elif subset == "val":
            stats["val_images"] += 1

    # Write annotation files
    for subset, data in coco_data.items():
        ann_path = out_path / "annotations" / f"{subset}.json"
        with open(ann_path, "w") as f:
            json.dump(data, f, indent=2)

    return BboxExportReport(
        format="coco",
        total_images=len(images),
        train_images=stats["train_images"],
        val_images=stats["val_images"],
        total_annotations=stats["total_annotations"],
        skipped_images=stats["skipped_images"],
        labels=label_names,
        output_dir=out_dir,
        warnings=warnings,
    )

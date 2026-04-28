"""Utilities for turning YOLO detections into AutoSeg annotations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional
import logging

import numpy as np
from PIL import Image

from core.masks import mask_to_bbox, mask_to_rle
from core.polygons import mask_to_yolo_polygon
from core.store import ProjectStore
from ml.segment import get_segment_service
from ml.yolo import get_yolo_service

logger = logging.getLogger(__name__)

YOLO_SOURCES = {"yolo", "yolo_sam", "yolo_seg"}


@dataclass
class YoloAnnotateOptions:
    """Configuration for YOLO-assisted annotation."""

    model_path: str
    confidence: float = 0.25
    iou: float = 0.7
    imgsz: Optional[int] = None
    max_detections: int = 300
    device: str = "cuda"
    class_filter: Optional[list[str]] = None
    use_sam: bool = True
    use_yolo_masks: bool = True
    status: str = "pending"
    duplicate_threshold: float = 0.85
    replace_existing_yolo: bool = False


@dataclass
class YoloRunSummary:
    """Summary for a YOLO annotation run."""

    images_processed: int = 0
    detections: int = 0
    created: int = 0
    skipped_duplicates: int = 0
    failed: int = 0
    sam_failures: int = 0
    labels_created: int = 0
    per_image: list[dict] = field(default_factory=list)

    def add_image(self, image_summary: "YoloRunSummary") -> None:
        self.images_processed += image_summary.images_processed
        self.detections += image_summary.detections
        self.created += image_summary.created
        self.skipped_duplicates += image_summary.skipped_duplicates
        self.failed += image_summary.failed
        self.sam_failures += image_summary.sam_failures
        self.labels_created += image_summary.labels_created

    def to_dict(self) -> dict:
        return asdict(self)


def run_yolo_on_project(
    store: ProjectStore,
    project_id: int,
    options: YoloAnnotateOptions,
) -> YoloRunSummary:
    """Run YOLO annotation on every image in a project."""
    summary = YoloRunSummary()
    for image in store.list_images(project_id):
        image_summary = run_yolo_on_image(store, image.id, options)
        summary.add_image(image_summary)
        summary.per_image.append(
            {
                "image_id": image.id,
                "order_index": image.order_index,
                "detections": image_summary.detections,
                "created": image_summary.created,
                "skipped_duplicates": image_summary.skipped_duplicates,
                "failed": image_summary.failed,
                "sam_failures": image_summary.sam_failures,
            }
        )
    return summary


def run_yolo_on_image(
    store: ProjectStore,
    image_id: int,
    options: YoloAnnotateOptions,
) -> YoloRunSummary:
    """Run YOLO annotation on one image and persist created annotations."""
    image = store.get_image_by_id(image_id)
    if image is None:
        raise ValueError(f"Image {image_id} not found")

    if options.replace_existing_yolo:
        _delete_existing_yolo_annotations(store, image_id)

    yolo_service = get_yolo_service(options.model_path, options.device)
    detections = yolo_service.predict(
        image_path=image.path,
        confidence=options.confidence,
        iou=options.iou,
        imgsz=options.imgsz,
        max_detections=options.max_detections,
        class_filter=options.class_filter,
    )

    summary = YoloRunSummary(images_processed=1, detections=len(detections))
    existing_annotations = store.list_annotations(image_id)
    batch_bboxes: list[list[float]] = []
    segment_service = None
    image_rgb = None

    for detection in detections:
        if _has_duplicate(
            detection.bbox_xyxy,
            existing_annotations,
            batch_bboxes,
            options.duplicate_threshold,
        ):
            summary.skipped_duplicates += 1
            continue

        label_before = store.get_label_by_name(image.project_id, detection.class_name)
        label = store.upsert_label(image.project_id, detection.class_name)
        if label_before is None:
            summary.labels_created += 1

        bbox = detection.bbox_xyxy
        mask_rle = None
        polygon = None
        source = "yolo"

        if options.use_sam:
            try:
                if segment_service is None:
                    if image_rgb is None:
                        with Image.open(image.path) as pil_img:
                            image_rgb = np.array(pil_img.convert("RGB"))
                    segment_service = get_segment_service(device=options.device)
                    if not segment_service.is_loaded():
                        segment_service.load_model()
                    segment_service.set_image(image_rgb, f"yolo:{image.id}")

                mask, _sam_score, refined_bbox = segment_service.segment_with_bbox(bbox)
                bbox = refined_bbox
                mask_rle = mask_to_rle(mask)
                polygon = mask_to_yolo_polygon(mask, image.width, image.height)
                source = "yolo_sam"
            except Exception as exc:
                summary.sam_failures += 1
                logger.warning(
                    "SAM refinement failed for YOLO detection on image %s: %s",
                    image_id,
                    exc,
                )

        if mask_rle is None and options.use_yolo_masks and detection.mask is not None:
            bbox = mask_to_bbox(detection.mask)
            mask_rle = mask_to_rle(detection.mask)
            polygon = mask_to_yolo_polygon(detection.mask, image.width, image.height)
            source = "yolo_seg"

        try:
            annotation = store.create_annotation(
                image_id=image_id,
                label_id=label.id,
                bbox_xyxy=bbox,
                source=source,
                status=options.status,
                confidence=detection.confidence,
                mask_rle=mask_rle,
                polygon_norm=polygon,
            )
            summary.created += 1
            batch_bboxes.append(annotation.bbox_xyxy or bbox)
        except Exception as exc:
            summary.failed += 1
            logger.warning("Failed to create YOLO annotation: %s", exc)

    return summary


def _delete_existing_yolo_annotations(store: ProjectStore, image_id: int) -> None:
    """Delete only annotations produced by previous YOLO runs."""
    for annotation in store.list_annotations(image_id):
        if annotation.source in YOLO_SOURCES:
            store.delete_annotation(annotation.id)


def _has_duplicate(
    bbox: list[float],
    existing_annotations,
    batch_bboxes: list[list[float]],
    threshold: float,
) -> bool:
    """Check whether a detection duplicates an existing or just-created bbox."""
    if threshold <= 0:
        return False

    for annotation in existing_annotations:
        if annotation.bbox_xyxy and _bbox_iou(bbox, annotation.bbox_xyxy) >= threshold:
            return True

    return any(_bbox_iou(bbox, existing_bbox) >= threshold for existing_bbox in batch_bboxes)


def _bbox_iou(bbox1: list[float], bbox2: list[float]) -> float:
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    return float(intersection / union) if union > 0 else 0.0

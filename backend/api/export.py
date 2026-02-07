"""
Export API endpoints
"""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from core.export_yolo import export_yolo_seg, verify_yolo_seg_export
from core.export_bbox import export_yolo_detect, export_coco
from core.validate import validate_project, ValidationWarning
from backend.api.projects import get_project, get_store

router = APIRouter()


class ExportRequest(BaseModel):
    output_dir: str
    train_split: float = 0.8
    seed: int = 42
    approved_only: bool = True
    include_negative: bool = False  # Include unlabeled images as negative examples
    labels_only: bool = False  # Export only label files (no image copying)
    labels_colocate: bool = True  # Place label files next to original images


class ExportResponse(BaseModel):
    train_images: int
    val_images: int
    total_annotations: int
    warnings: list[str]
    is_valid: bool
    validation_errors: list[str]


@router.post("/yolo", response_model=ExportResponse)
async def export_yolo(request: ExportRequest):
    """Export project to YOLO-seg format."""
    project = get_project()

    # Validate output directory
    try:
        os.makedirs(request.output_dir, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid output directory: {e}")

    # Run export
    try:
        report = export_yolo_seg(
            project_dir=project.root_dir,
            out_dir=request.output_dir,
            split={"train": request.train_split, "val": 1.0 - request.train_split},
            seed=request.seed,
            approved_only=request.approved_only,
            include_negative=request.include_negative,
            labels_only=request.labels_only,
            labels_colocate=request.labels_colocate,
        )

        # Verify export (labels-only export intentionally skips data.yaml/images)
        if request.labels_only:
            is_valid, errors = True, []
        else:
            is_valid, errors = verify_yolo_seg_export(request.output_dir)

        return ExportResponse(
            train_images=report.train_images,
            val_images=report.val_images,
            total_annotations=report.total_annotations,
            warnings=report.warnings,
            is_valid=is_valid,
            validation_errors=errors[:10] if errors else [],  # Limit error count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")


class ValidationWarningResponse(BaseModel):
    annotation_id: int
    severity: str
    code: str
    message: str


class ValidateResponse(BaseModel):
    total_images: int
    total_annotations: int
    error_count: int
    warning_count: int
    is_valid: bool
    errors: list[ValidationWarningResponse]
    warnings: list[ValidationWarningResponse]


@router.get("/validate", response_model=ValidateResponse)
async def validate():
    """Validate all annotations in the current project."""
    project = get_project()
    store = get_store()

    report = validate_project(store, project.id)

    return ValidateResponse(
        total_images=report.total_images,
        total_annotations=report.total_annotations,
        error_count=report.error_count,
        warning_count=report.warning_count,
        is_valid=report.is_valid,
        errors=[
            ValidationWarningResponse(
                annotation_id=w.annotation_id,
                severity=w.severity,
                code=w.code,
                message=w.message,
            )
            for w in report.errors[:50]  # Limit to 50 errors
        ],
        warnings=[
            ValidationWarningResponse(
                annotation_id=w.annotation_id,
                severity=w.severity,
                code=w.code,
                message=w.message,
            )
            for w in report.warnings[:50]  # Limit to 50 warnings
        ],
    )


class BboxExportRequest(BaseModel):
    output_dir: str
    format: str = "yolo-detect"  # "yolo-detect" or "coco"
    train_split: float = 0.8
    seed: int = 42
    approved_only: bool = True
    include_segmentation: bool = False  # COCO only: include polygon if available
    include_negative: bool = False  # Include unlabeled images as negative examples
    labels_only: bool = False  # Export only label/annotation files (no image copying)
    labels_colocate: bool = True  # YOLO-detect only: place labels next to images


class BboxExportResponse(BaseModel):
    format: str
    train_images: int
    val_images: int
    total_annotations: int
    warnings: list[str]


@router.post("/bbox", response_model=BboxExportResponse)
async def export_bbox(request: BboxExportRequest):
    """Export project bounding boxes to YOLO-detect or COCO format."""
    project = get_project()

    try:
        os.makedirs(request.output_dir, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid output directory: {e}")

    try:
        split = {"train": request.train_split, "val": 1.0 - request.train_split}

        if request.format == "coco":
            report = export_coco(
                project_dir=project.root_dir,
                out_dir=request.output_dir,
                split=split,
                seed=request.seed,
                approved_only=request.approved_only,
                include_segmentation=request.include_segmentation,
                include_negative=request.include_negative,
                labels_only=request.labels_only,
            )
        else:  # yolo-detect
            report = export_yolo_detect(
                project_dir=project.root_dir,
                out_dir=request.output_dir,
                split=split,
                seed=request.seed,
                approved_only=request.approved_only,
                include_negative=request.include_negative,
                labels_only=request.labels_only,
                labels_colocate=request.labels_colocate,
            )

        return BboxExportResponse(
            format=report.format,
            train_images=report.train_images,
            val_images=report.val_images,
            total_annotations=report.total_annotations,
            warnings=report.warnings,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

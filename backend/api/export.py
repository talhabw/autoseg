"""
Export API endpoints
"""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from core.export_yolo import export_yolo_seg, verify_yolo_seg_export
from core.validate import validate_project, ValidationWarning
from backend.api.projects import get_project, get_store

router = APIRouter()


class ExportRequest(BaseModel):
    output_dir: str
    train_split: float = 0.8
    seed: int = 42
    approved_only: bool = True


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
        )

        # Verify export
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

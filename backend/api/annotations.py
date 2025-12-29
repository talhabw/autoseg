"""
Annotations API endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from core.models import bbox_to_json, polygon_to_json, mask_rle_to_json
from backend.api.projects import get_store

router = APIRouter()


class AnnotationResponse(BaseModel):
    id: int
    image_id: int
    label_id: int
    bbox: Optional[list[float]] = None
    polygon: Optional[list[float]] = None
    mask_rle: Optional[dict] = None
    source: str
    confidence: Optional[float] = None
    status: str


class CreateAnnotationRequest(BaseModel):
    image_id: int
    label_id: int
    bbox: list[float]  # [x1, y1, x2, y2]
    source: str = "manual"
    status: str = "approved"
    confidence: Optional[float] = None
    mask_rle: Optional[dict] = None
    polygon: Optional[list[float]] = None


class UpdateAnnotationRequest(BaseModel):
    label_id: Optional[int] = None
    bbox: Optional[list[float]] = None
    polygon: Optional[list[float]] = None
    mask_rle: Optional[dict] = None
    status: Optional[str] = None
    confidence: Optional[float] = None


def annotation_to_response(ann) -> AnnotationResponse:
    """Convert Annotation to AnnotationResponse."""
    return AnnotationResponse(
        id=ann.id,
        image_id=ann.image_id,
        label_id=ann.label_id,
        bbox=ann.bbox_xyxy,
        polygon=ann.polygon_norm,
        mask_rle=ann.mask_rle,
        source=ann.source,
        confidence=ann.confidence,
        status=ann.status,
    )


@router.get("", response_model=list[AnnotationResponse])
async def list_annotations(image_id: int):
    """List annotations for an image."""
    store = get_store()
    annotations = store.list_annotations(image_id)
    return [annotation_to_response(ann) for ann in annotations]


@router.get("/{annotation_id}", response_model=AnnotationResponse)
async def get_annotation(annotation_id: int):
    """Get annotation by ID."""
    store = get_store()
    ann = store.get_annotation_by_id(annotation_id)

    if ann is None:
        raise HTTPException(status_code=404, detail="Annotation not found")

    return annotation_to_response(ann)


@router.post("", response_model=AnnotationResponse)
async def create_annotation(request: CreateAnnotationRequest):
    """Create a new annotation."""
    store = get_store()

    try:
        ann = store.create_annotation(
            image_id=request.image_id,
            label_id=request.label_id,
            bbox_xyxy=request.bbox,
            source=request.source,
            status=request.status,
            confidence=request.confidence,
            mask_rle=request.mask_rle,
            polygon_norm=request.polygon,
        )
        return annotation_to_response(ann)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{annotation_id}", response_model=AnnotationResponse)
async def update_annotation(annotation_id: int, request: UpdateAnnotationRequest):
    """Update an annotation."""
    store = get_store()

    # Build update fields - use field names that store.update_annotation expects
    update_fields = {}
    if request.label_id is not None:
        update_fields["label_id"] = request.label_id
    if request.bbox is not None:
        update_fields["bbox_xyxy"] = request.bbox
    if request.polygon is not None:
        update_fields["polygon_norm"] = request.polygon
    if request.mask_rle is not None:
        update_fields["mask_rle"] = request.mask_rle
    if request.status is not None:
        update_fields["status"] = request.status
    if request.confidence is not None:
        update_fields["confidence"] = request.confidence

    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    try:
        ann = store.update_annotation(annotation_id, **update_fields)
        return annotation_to_response(ann)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{annotation_id}")
async def delete_annotation(annotation_id: int):
    """Delete an annotation."""
    store = get_store()

    try:
        store.delete_annotation(annotation_id)
        return {"status": "deleted", "id": annotation_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/all/{project_id}")
async def delete_all_annotations(project_id: int):
    """Delete all annotations in a project. USE WITH CAUTION."""
    store = get_store()
    
    try:
        # Get all images in project
        images = store.list_images(project_id)
        total_deleted = 0
        
        for image in images:
            annotations = store.list_annotations(image.id)
            for ann in annotations:
                store.delete_annotation(ann.id)
                total_deleted += 1
        
        return {"status": "deleted", "count": total_deleted}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

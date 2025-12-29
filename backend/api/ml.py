"""
ML API endpoints - SAM segmentation, embedding, propagation
"""

import gc
import logging
import traceback
import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from PIL import Image

from ml.segment import get_segment_service, clear_segment_service
from ml.embed import (
    get_embed_service,
    set_default_model,
    clear_embed_service,
    get_available_models,
)
from ml.propagate import (
    PropagateService,
    PropagationSizeMismatchError,
    PropagationNotFoundError,
)
from core.masks import mask_to_rle, rle_to_mask, mask_iou
from core.polygons import mask_to_yolo_polygon
from backend.api.projects import get_store

logger = logging.getLogger(__name__)

router = APIRouter()

# Only track propagate service locally (it's a composite that depends on the singletons)
# Segment and embed services use global singletons from their respective modules
_propagate_service: Optional[PropagateService] = None
_current_embed_model: Optional[str] = None  # Track which model is loaded


class LoadModelRequest(BaseModel):
    device: str = "cuda"
    embed_model: str = "vith16"  # DINOv3: vitb16/vitl16/vith16, Pixio: pixio_vitb16/pixio_vitl16/pixio_vith16/pixio_vit1b16


class SegmentRequest(BaseModel):
    image_id: int
    bbox: list[float]  # [x1, y1, x2, y2]
    pos_points: Optional[list[list[float]]] = None  # [[x, y], ...]
    neg_points: Optional[list[list[float]]] = None  # [[x, y], ...]


class SegmentResponse(BaseModel):
    mask_rle: dict
    polygon: list[float]
    score: float
    bbox: list[float]  # refined bbox


class PropagateRequest(BaseModel):
    source_image_id: int
    target_image_id: int
    source_annotation_id: int
    size_min_ratio: float = 0.8  # Min allowed size ratio (e.g., 0.8x)
    size_max_ratio: float = 1.2  # Max allowed size ratio (e.g., 1.2x)
    stop_on_size_mismatch: bool = (
        True  # If True, return None when no size-OK result; if False, use fallback
    )
    skip_duplicate_threshold: float = (
        0.9  # Skip if IoU with existing annotation >= this (0 = disabled)
    )
    top_k: int = 5  # Number of peak candidates to try


class PropagateResponse(BaseModel):
    bbox: list[float]
    mask_rle: dict
    polygon: list[float]
    confidence: float
    fallback_used: bool = False  # True if size-mismatch fallback was used
    area_ratio: float = 1.0  # Ratio of new area to old area
    duplicate_skipped: bool = False  # True if propagation was skipped due to duplicate
    duplicate_iou: float = (
        0.0  # IoU with the overlapping annotation (if duplicate_skipped)
    )


def _load_image(image_id: int) -> tuple[np.ndarray, int, int]:
    """Load image by ID and return as RGB array with dimensions."""
    store = get_store()
    image = store.get_image_by_id(image_id)

    if image is None:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

    try:
        pil_img = Image.open(image.path).convert("RGB")
        return np.array(pil_img), image.width, image.height
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load image: {e}")


# ==================== SAM Endpoints ====================


@router.post("/sam/load")
async def load_sam(request: LoadModelRequest):
    """Load SAM model."""
    try:
        segment_service = get_segment_service(device=request.device)
        if not segment_service.is_loaded():
            segment_service.load_model()
        return {"status": "loaded", "device": request.device}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load SAM: {e}")


@router.get("/sam/status")
async def sam_status():
    """Check if SAM is loaded."""
    try:
        segment_service = get_segment_service()
        return {"loaded": segment_service.is_loaded()}
    except Exception:
        return {"loaded": False}


@router.post("/unload")
async def unload_all_models():
    """Unload all ML models to free GPU memory."""
    global _propagate_service, _current_embed_model

    unloaded = []

    # Unload propagate first (it depends on embed and segment)
    if _propagate_service is not None:
        _propagate_service.unload_model()
        _propagate_service = None
        unloaded.append("Propagation")

    # Clear global singletons - this properly frees GPU memory
    clear_segment_service()
    unloaded.append("SAM")

    clear_embed_service()
    _current_embed_model = None
    unloaded.append("Embedding")

    # Force garbage collection and clear CUDA cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all GPU ops complete

    logger.info(f"Unloaded models: {unloaded}")
    return {"status": "unloaded", "models": unloaded}


@router.post("/unload/embed")
async def unload_embed_model():
    """Unload embedding model to free GPU memory (for model switching)."""
    global _propagate_service, _current_embed_model

    unloaded = []

    # Unload propagate first since it depends on embed
    if _propagate_service is not None:
        _propagate_service.unload_model()
        _propagate_service = None
        unloaded.append("Propagation")

    # Clear global singleton in embed module
    clear_embed_service()
    _current_embed_model = None
    unloaded.append("Embedding")

    # Force garbage collection and clear CUDA cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all GPU ops complete

    logger.info(f"Unloaded embed models: {unloaded}")
    return {"status": "unloaded", "models": unloaded}


@router.post("/segment", response_model=SegmentResponse)
async def segment(request: SegmentRequest):
    """Run SAM segmentation on a bounding box."""
    segment_service = get_segment_service()

    if not segment_service.is_loaded():
        raise HTTPException(
            status_code=400, detail="SAM not loaded. Call /api/ml/sam/load first"
        )

    # Load image
    image_rgb, width, height = _load_image(request.image_id)

    # Set image for segmentation
    segment_service.set_image(image_rgb, str(request.image_id))

    # Convert points format
    pos_points = None
    neg_points = None
    if request.pos_points:
        pos_points = [(p[0], p[1]) for p in request.pos_points]
    if request.neg_points:
        neg_points = [(p[0], p[1]) for p in request.neg_points]

    # Run segmentation
    try:
        mask, score, refined_bbox = segment_service.segment_with_bbox(
            bbox_xyxy=request.bbox, pos_points=pos_points, neg_points=neg_points
        )

        # Convert mask to RLE and polygon
        rle = mask_to_rle(mask)
        polygon = mask_to_yolo_polygon(mask, width, height)

        return SegmentResponse(
            mask_rle=rle, polygon=polygon, score=float(score), bbox=refined_bbox
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")


# ==================== Embedding Endpoints ====================


@router.post("/embed/load")
async def load_embed(request: LoadModelRequest):
    """Load embedding model (DINOv2)."""
    try:
        # Check if we need to switch models
        current_service = get_embed_service(device=request.device)
        if current_service._model_name != request.embed_model:
            logger.info(
                f"Switching embed model from {current_service._model_name} to {request.embed_model}"
            )
            clear_embed_service()

        # Set default and load
        set_default_model(request.embed_model)
        embed_service = get_embed_service(device=request.device)

        if not embed_service.is_loaded():
            embed_service.load_model()

        return {
            "status": "loaded",
            "device": request.device,
            "model": request.embed_model,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load embedding model: {e}"
        )


@router.get("/embed/status")
async def embed_status():
    """Check if embedding model is loaded."""
    try:
        embed_service = get_embed_service()
        return {"loaded": embed_service.is_loaded(), "model": embed_service._model_name}
    except Exception:
        return {"loaded": False}


@router.get("/embed/models")
async def get_available_embed_models():
    """Get list of available embedding models."""
    return {"models": get_available_models()}


# ==================== Propagation Endpoints ====================


@router.post("/propagate/load")
async def load_propagation(request: LoadModelRequest):
    """Load both SAM and embedding models for propagation.

    Available embed_model options: vitb16, vitl16, vith16
    """
    global _propagate_service, _current_embed_model

    try:
        # Check if we need to switch models (check service directly, not just _current_embed_model)
        # This handles cases where embed_service was loaded separately via /embed/load
        embed_service = get_embed_service(device=request.device)

        if (
            _current_embed_model != request.embed_model
            or embed_service._model_name != request.embed_model
        ):
            logger.info(
                f"Switching embed model to {request.embed_model} (current: {_current_embed_model}, service: {embed_service._model_name})"
            )
            clear_embed_service()
            if _propagate_service is not None:
                _propagate_service = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Set default embedding model before getting the service
        set_default_model(request.embed_model)
        logger.info(f"Using embedding model: {request.embed_model}")

        # Load SAM using global singleton
        segment_service = get_segment_service(device=request.device)
        if not segment_service.is_loaded():
            segment_service.load_model()

        # Load embedding using global singleton (with the new default model)
        embed_service = get_embed_service(device=request.device)
        if not embed_service.is_loaded():
            embed_service.load_model()

        _current_embed_model = request.embed_model

        # Create propagation service using the singletons
        _propagate_service = PropagateService(embed_service, segment_service)

        return {
            "status": "loaded",
            "device": request.device,
            "embed_model": request.embed_model,
        }
    except Exception as e:
        logger.error(f"Failed to load propagation models: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Failed to load propagation models: {e}"
        )


@router.get("/propagate/status")
async def propagation_status():
    """Check if propagation models are loaded."""
    try:
        segment_service = get_segment_service()
        embed_service = get_embed_service()

        sam_loaded = segment_service.is_loaded()
        embed_loaded = embed_service.is_loaded()

        return {
            "loaded": sam_loaded and embed_loaded,
            "sam_loaded": sam_loaded,
            "embed_loaded": embed_loaded,
        }
    except Exception:
        return {
            "loaded": False,
            "sam_loaded": False,
            "embed_loaded": False,
        }


@router.post("/propagate", response_model=PropagateResponse)
async def propagate(request: PropagateRequest):
    """Propagate an annotation from source to target image."""
    global _propagate_service

    if _propagate_service is None:
        raise HTTPException(
            status_code=400,
            detail="Propagation not loaded. Call /api/ml/propagate/load first",
        )

    store = get_store()

    # Get source annotation
    ann = store.get_annotation_by_id(request.source_annotation_id)
    if ann is None:
        raise HTTPException(status_code=404, detail="Source annotation not found")

    source_bbox = ann.bbox_xyxy
    if source_bbox is None:
        raise HTTPException(status_code=400, detail="Source annotation has no bbox")

    source_mask = None
    if ann.mask_rle:
        source_mask = rle_to_mask(ann.mask_rle)

    # Load images
    source_image, src_w, src_h = _load_image(request.source_image_id)
    target_image, tgt_w, tgt_h = _load_image(request.target_image_id)

    try:
        result = _propagate_service.propagate_annotation(
            source_image=source_image,
            source_bbox=source_bbox,
            source_mask=source_mask,
            target_image=target_image,
            source_image_id=str(request.source_image_id),
            target_image_id=str(request.target_image_id),
            annotation_id=request.source_annotation_id,
            top_k=request.top_k,
            size_min_ratio=request.size_min_ratio,
            size_max_ratio=request.size_max_ratio,
            stop_on_size_mismatch=request.stop_on_size_mismatch,
        )

        if result is None:
            raise HTTPException(
                status_code=400,
                detail="Propagation failed - could not find object in target",
            )

        new_bbox, new_mask, confidence, fallback_used, area_ratio = result

        # Check for duplicates with existing annotations on target image
        duplicate_skipped = False
        duplicate_iou = 0.0

        if request.skip_duplicate_threshold > 0:
            # Get existing annotations on target image
            target_annotations = store.list_annotations(request.target_image_id)

            for existing_ann in target_annotations:
                if existing_ann.mask_rle:
                    try:
                        existing_mask = rle_to_mask(existing_ann.mask_rle)
                        iou = mask_iou(new_mask, existing_mask)

                        if iou >= request.skip_duplicate_threshold:
                            duplicate_skipped = True
                            duplicate_iou = iou
                            logger.info(
                                f"Skipping duplicate annotation (IoU={iou:.3f} with ann {existing_ann.id})"
                            )
                            break
                    except Exception as e:
                        logger.warning(f"Failed to compare masks: {e}")
                        continue

        # Convert to RLE and polygon
        rle = mask_to_rle(new_mask)
        polygon = mask_to_yolo_polygon(new_mask, tgt_w, tgt_h)

        return PropagateResponse(
            bbox=new_bbox,
            mask_rle=rle,
            polygon=polygon,
            confidence=float(confidence),
            fallback_used=fallback_used,
            area_ratio=area_ratio,
            duplicate_skipped=duplicate_skipped,
            duplicate_iou=duplicate_iou,
        )
    except PropagationSizeMismatchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PropagationNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Propagation failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Propagation failed: {e}")

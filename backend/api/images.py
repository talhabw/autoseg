"""
Images API endpoints
"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io

from backend.api.projects import get_store, get_project

router = APIRouter()


class ImageResponse(BaseModel):
    id: int
    project_id: int
    path: str
    width: int
    height: int
    order_index: int


@router.get("", response_model=list[ImageResponse])
async def list_images():
    """List all images in the current project."""
    store = get_store()
    project = get_project()

    images = store.list_images(project.id)
    return [
        ImageResponse(
            id=img.id,
            project_id=img.project_id,
            path=img.path,
            width=img.width,
            height=img.height,
            order_index=img.order_index,
        )
        for img in images
    ]


@router.get("/{image_id}", response_model=ImageResponse)
async def get_image(image_id: int):
    """Get image metadata by ID."""
    store = get_store()
    image = store.get_image_by_id(image_id)

    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    return ImageResponse(
        id=image.id,
        project_id=image.project_id,
        path=image.path,
        width=image.width,
        height=image.height,
        order_index=image.order_index,
    )


@router.get("/by-index/{order_index}", response_model=ImageResponse)
async def get_image_by_index(order_index: int):
    """Get image by order index."""
    store = get_store()
    project = get_project()

    image = store.get_image_by_index(project.id, order_index)

    if image is None:
        raise HTTPException(status_code=404, detail="Image not found at index")

    return ImageResponse(
        id=image.id,
        project_id=image.project_id,
        path=image.path,
        width=image.width,
        height=image.height,
        order_index=image.order_index,
    )


@router.get("/{image_id}/file")
async def get_image_file(image_id: int):
    """Serve the image file."""
    store = get_store()
    image = store.get_image_by_id(image_id)

    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    if not os.path.exists(image.path):
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    return FileResponse(
        image.path, media_type="image/jpeg", filename=os.path.basename(image.path)
    )


@router.get("/{image_id}/thumbnail")
async def get_image_thumbnail(image_id: int, size: int = 200):
    """Serve a thumbnail of the image."""
    store = get_store()
    image = store.get_image_by_id(image_id)

    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    if not os.path.exists(image.path):
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    # Generate thumbnail
    try:
        img = Image.open(image.path)
        img.thumbnail((size, size))

        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        from fastapi.responses import StreamingResponse

        return StreamingResponse(buffer, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate thumbnail: {e}"
        )


@router.get("/count")
async def get_image_count():
    """Get total number of images in project."""
    store = get_store()
    project = get_project()
    count = store.get_image_count(project.id)
    return {"count": count}


@router.get("/with-status/{status}")
async def get_images_with_status(status: str):
    """Get list of image indices that have annotations with a given status."""
    store = get_store()
    project = get_project()
    
    # Get all annotations with this status
    results = store.list_annotations_by_status(project.id, status)
    
    # Extract unique image order indices
    image_indices = sorted(set(img.order_index for _ann, img in results))
    
    return {
        "status": status,
        "image_indices": image_indices,
        "count": len(image_indices)
    }

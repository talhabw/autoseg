"""
Labels API endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.api.projects import get_store, get_project

router = APIRouter()


class LabelResponse(BaseModel):
    id: int
    project_id: int
    name: str
    color: str


class CreateLabelRequest(BaseModel):
    name: str
    color: Optional[str] = None  # If not provided, auto-assign


@router.get("", response_model=list[LabelResponse])
async def list_labels():
    """List all labels in the current project."""
    store = get_store()
    project = get_project()

    labels = store.list_labels(project.id)
    return [
        LabelResponse(
            id=label.id,
            project_id=label.project_id,
            name=label.name,
            color=label.color_hex,
        )
        for label in labels
    ]


@router.get("/{label_id}", response_model=LabelResponse)
async def get_label(label_id: int):
    """Get label by ID."""
    store = get_store()
    label = store.get_label_by_id(label_id)

    if label is None:
        raise HTTPException(status_code=404, detail="Label not found")

    return LabelResponse(
        id=label.id, project_id=label.project_id, name=label.name, color=label.color_hex
    )


@router.post("", response_model=LabelResponse)
async def create_label(request: CreateLabelRequest):
    """Create or get existing label."""
    store = get_store()
    project = get_project()

    try:
        label = store.upsert_label(
            project_id=project.id, name=request.name, color_hex=request.color
        )
        return LabelResponse(
            id=label.id,
            project_id=label.project_id,
            name=label.name,
            color=label.color_hex,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

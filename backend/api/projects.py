"""
Projects API endpoints
"""

import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from core.store import ProjectStore
from core.models import Project

router = APIRouter()

# Store the current project state
_current_store: Optional[ProjectStore] = None
_current_project: Optional[Project] = None


class CreateProjectRequest(BaseModel):
    project_dir: str
    image_dir: str
    name: str


class ProjectResponse(BaseModel):
    id: int
    name: str
    root_dir: str
    image_count: int

    @classmethod
    def from_project(cls, project: Project, store: ProjectStore):
        count = store.get_image_count(project.id)
        return cls(
            id=project.id,
            name=project.name,
            root_dir=project.root_dir,
            image_count=count,
        )


def get_store() -> ProjectStore:
    """Get the current project store."""
    if _current_store is None:
        raise HTTPException(status_code=400, detail="No project loaded")
    return _current_store


def get_project() -> Project:
    """Get the current project."""
    if _current_project is None:
        raise HTTPException(status_code=400, detail="No project loaded")
    return _current_project


@router.post("", response_model=ProjectResponse)
async def create_project(request: CreateProjectRequest):
    """Create a new project."""
    global _current_store, _current_project

    try:
        project = ProjectStore.create_project(
            project_dir=request.project_dir,
            image_dir=request.image_dir,
            name=request.name,
        )
        _current_store = ProjectStore(project.db_path)
        _current_project = project
        return ProjectResponse.from_project(project, _current_store)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class OpenProjectRequest(BaseModel):
    project_dir: str


@router.post("/open", response_model=ProjectResponse)
async def open_project(request: OpenProjectRequest):
    """Open an existing project and resync images."""
    global _current_store, _current_project

    if not os.path.exists(request.project_dir):
        raise HTTPException(status_code=404, detail="Project directory not found")

    db_path = os.path.join(request.project_dir, "autoseg.db")
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Project database not found")

    try:
        project = ProjectStore.load_project(request.project_dir)
        _current_store = ProjectStore(project.db_path)
        _current_project = project

        # Auto-resync images on open
        settings = json.loads(project.settings_json) if project.settings_json else {}
        image_dir = settings.get("image_dir")

        if image_dir and os.path.exists(image_dir):
            try:
                resync_result = _current_store.resync_images(project.id, image_dir)
                if resync_result["added"] > 0 or resync_result["removed"] > 0:
                    print(
                        f"Resynced images: +{resync_result['added']}, -{resync_result['removed']}"
                    )
            except Exception as e:
                print(f"Warning: Image resync failed: {e}")

        return ProjectResponse.from_project(project, _current_store)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/current", response_model=Optional[ProjectResponse])
async def get_current_project():
    """Get the currently loaded project."""
    if _current_project is None:
        return None
    return ProjectResponse.from_project(_current_project, _current_store)


@router.post("/close")
async def close_project():
    """Close the current project."""
    global _current_store, _current_project

    if _current_store:
        _current_store.close()
    _current_store = None
    _current_project = None
    return {"status": "closed"}


@router.get("/settings/{key}")
async def get_setting(key: str):
    """Get a project setting."""
    store = get_store()
    value = store.get_setting(key)
    return {"key": key, "value": value}


@router.put("/settings/{key}")
async def set_setting(key: str, value: str):
    """Set a project setting."""
    store = get_store()
    store.set_setting(key, value)
    return {"key": key, "value": value}


class ResyncImagesResponse(BaseModel):
    added: int
    removed: int
    unchanged: int
    total: int


@router.post("/resync-images", response_model=ResyncImagesResponse)
async def resync_images():
    """
    Rescan the image directory and update the database.

    - Adds new images found in the directory
    - Removes records for images that no longer exist
    - Preserves existing annotations for unchanged images
    """
    store = get_store()
    project = get_project()

    # Get image_dir from project settings
    settings = json.loads(project.settings_json) if project.settings_json else {}
    image_dir = settings.get("image_dir")

    if not image_dir:
        raise HTTPException(
            status_code=400,
            detail="Project does not have an image directory configured",
        )

    if not os.path.exists(image_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Image directory not found: {image_dir}",
        )

    try:
        result = store.resync_images(project.id, image_dir)
        total = store.get_image_count(project.id)
        return ResyncImagesResponse(
            added=result["added"],
            removed=result["removed"],
            unchanged=result["unchanged"],
            total=total,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
Projects API endpoints
"""

import os
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
    """Open an existing project."""
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

"""
Filesystem API routes for folder browsing
"""

import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("/list")
async def list_directory(
    path: str = Query("/", description="Directory path to list"),
    dirs_only: bool = Query(True, description="Only return directories"),
):
    """
    List contents of a directory for folder browser UI.
    
    Security: Only allows browsing existing directories, no file reads.
    """
    try:
        # Expand user home directory
        if path.startswith("~"):
            path = os.path.expanduser(path)
        
        # Resolve to absolute path
        dir_path = Path(path).resolve()
        
        # Ensure it's a directory
        if not dir_path.exists():
            raise HTTPException(status_code=404, detail=f"Path does not exist: {path}")
        
        if not dir_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")
        
        # Get parent path
        parent = str(dir_path.parent) if dir_path.parent != dir_path else None
        
        # List directory contents
        entries = []
        try:
            for entry in sorted(dir_path.iterdir()):
                try:
                    # Skip hidden files/folders
                    if entry.name.startswith('.'):
                        continue
                    
                    is_dir = entry.is_dir()
                    
                    # Skip files if dirs_only
                    if dirs_only and not is_dir:
                        continue
                    
                    entries.append({
                        "name": entry.name,
                        "path": str(entry),
                        "is_dir": is_dir,
                    })
                except PermissionError:
                    # Skip entries we can't access
                    continue
        except PermissionError:
            raise HTTPException(status_code=403, detail=f"Permission denied: {path}")
        
        return {
            "path": str(dir_path),
            "parent": parent,
            "entries": entries,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/home")
async def get_home_directory():
    """Get the user's home directory path."""
    return {
        "path": os.path.expanduser("~"),
    }

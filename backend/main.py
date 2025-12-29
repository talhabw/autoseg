"""
FastAPI application entry point
"""

import sys
import logging
import traceback
from pathlib import Path

# Add parent directory to path so we can import core and ml modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from backend.config import CORS_ORIGINS, API_HOST, API_PORT
from backend.api import projects, images, annotations, labels, ml, export, files

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("AutoSeg API starting...")
    yield
    # Shutdown
    logger.info("AutoSeg API shutting down...")


app = FastAPI(
    title="AutoSeg API",
    description="Image annotation API with SAM segmentation and tracking",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log all unhandled exceptions with full traceback."""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}:")
    logger.error(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(projects.router, prefix="/api/projects", tags=["Projects"])
app.include_router(images.router, prefix="/api/images", tags=["Images"])
app.include_router(annotations.router, prefix="/api/annotations", tags=["Annotations"])
app.include_router(labels.router, prefix="/api/labels", tags=["Labels"])
app.include_router(ml.router, prefix="/api/ml", tags=["ML"])
app.include_router(export.router, prefix="/api/export", tags=["Export"])
app.include_router(files.router, prefix="/api/files", tags=["Files"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "autoseg-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)

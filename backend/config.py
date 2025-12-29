"""
Backend configuration
"""

import os
from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).parent.parent  # autoseg/
CACHE_DIR = ROOT_DIR / "cache"
MODELS_DIR = ROOT_DIR / "models"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# CORS origins (frontend URL)
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

# SAM model settings
SAM_MODEL_PATH = os.getenv("SAM_MODEL_PATH", str(MODELS_DIR / "sam2.1_hiera_large.pt"))
SAM_MODEL_CONFIG = os.getenv("SAM_MODEL_CONFIG", "sam2.1_hiera_l.yaml")

# Embedding model settings
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "dinov2_vitl14")

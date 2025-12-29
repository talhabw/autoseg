"""
Vision Encoder Embedding Service - Feature extraction for object tracking/propagation.

Supports DINOv3 and Pixio vision encoders.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Union
import logging

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from PIL import Image
import sys
from functools import partial
import torch.nn as nn

from ml.models_config import DINOV3_MODELS, PIXIO_MODELS, ALL_MODELS

logger = logging.getLogger(__name__)

# Model paths
_MODELS_DIR = Path(__file__).parent.parent / "models"
_DINOV3_REPO = Path(__file__).parent.parent / "third_party" / "dinov3"

# Setup Pixio path
_PIXIO_INNER = _DINOV3_REPO.parent / "pixio" / "pixio"
if str(_PIXIO_INNER) not in sys.path:
    sys.path.insert(0, str(_PIXIO_INNER))

try:
    from pixio import PixioViT
except ImportError:
    PixioViT = None
    logger.warning("Could not import PixioViT. Pixio models will not check_available.")

# Global default model setting
_DEFAULT_MODEL = "vitb16"


def set_default_model(model_name: str):
    """Set the default model to use."""
    global _DEFAULT_MODEL
    if model_name not in ALL_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: {list(ALL_MODELS.keys())}"
        )
    _DEFAULT_MODEL = model_name
    backend = "Pixio" if model_name.startswith("pixio_") else "DINOv3"
    logger.info(f"Default model set to: {model_name} ({backend})")


def get_available_models() -> list[dict]:
    """
    Get list of all supported models with availability status.

    Returns:
        List of dicts: {id, name, available, description, download_url, weights_file}
    """
    models = []

    # Check DINOv3 models
    for model_id, config in DINOV3_MODELS.items():
        weights_path = _MODELS_DIR / config["weights"]
        models.append(
            {
                "id": model_id,
                "name": config.get("description", model_id),
                "available": weights_path.exists(),
                "download_url": config.get("download_url"),
                "weights_file": config["weights"],
            }
        )

    # Check Pixio models
    for model_id, config in PIXIO_MODELS.items():
        weights_path = _MODELS_DIR / config["weights"]
        models.append(
            {
                "id": model_id,
                "name": config.get("description", model_id),
                "available": weights_path.exists(),
                "download_url": config.get("download_url"),
                "weights_file": config["weights"],
            }
        )

    return models


def get_default_model() -> str:
    """Get the default model name, prioritizing available ones."""
    # If currently selected model is available, keep it
    current_config = ALL_MODELS.get(_DEFAULT_MODEL)
    if current_config:
        weights_path = _MODELS_DIR / current_config["weights"]
        if weights_path.exists():
            return _DEFAULT_MODEL

    # Otherwise find first available model
    available = [m for m in get_available_models() if m["available"]]
    if available:
        logger.info(
            f"Default model {_DEFAULT_MODEL} not found, falling back to {available[0]['id']}"
        )
        return available[0]["id"]

    return _DEFAULT_MODEL


def make_transform(resize_size: int = 518):
    """
    Create the standard DINOv3 transform for LVD-1689M weights.

    Args:
        resize_size: Size to resize images to (should be divisible by patch_size=16)
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


class EmbedService:
    """
    Vision encoder embedding service for feature extraction.

    Supports:
    - DINOv3 models (vitb16, vitl16, vith16)
    - Pixio models (pixio_vitb16, pixio_vitl16, pixio_vith16, pixio_vit1b16)

    Provides:
    - Full image feature maps (for dense matching)
    - Object descriptors from bbox regions (for tracking)
    - Caching of computed features
    """

    def __init__(
        self,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize embed service.

        Args:
            device: Device to run on ('cuda' or 'cpu')
            cache_dir: Directory for caching embeddings (optional)
            model_name: Model variant.
        """
        self.device = device
        self.model = None
        self.transform = None
        self._model_name = model_name or _DEFAULT_MODEL
        self._embed_dim = None
        self._patch_size = 16
        self._img_size = 518  # Divisible by 16, good resolution
        self._is_pixio = self._model_name.startswith("pixio_")

        # Cache settings
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for current session
        self._feature_cache: dict[str, torch.Tensor] = {}
        self._descriptor_cache: dict[str, np.ndarray] = {}

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def load_model(self, weights_path: Optional[str] = None):
        """
        Load vision encoder model (DINOv3 or Pixio).

        Args:
            weights_path: Path to model weights (uses default based on model_name if None)
        """
        # Determine backend and get config
        self._is_pixio = self._model_name.startswith("pixio_")

        if self._is_pixio:
            self._img_size = 512  # Pixio requires input size divisible by patch_size (16), 518 is not
            model_config = PIXIO_MODELS.get(self._model_name)
            backend_name = "Pixio"
        else:
            self._img_size = 518  # DINOv3 default
            model_config = DINOV3_MODELS.get(self._model_name)
            backend_name = "DINOv3"

        if not model_config:
            raise ValueError(f"Unknown model: {self._model_name}")

        # Determine weights path
        if weights_path is None:
            weights_path = str(_MODELS_DIR / model_config["weights"])

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Model weights not found: {weights_path}\n"
                f"Please download {model_config['weights']} to {_MODELS_DIR}"
            )

        logger.info(f"Loading {backend_name} {self._model_name} from {weights_path}...")

        try:
            if self._is_pixio:
                if PixioViT is None:
                    raise ImportError(
                        "PixioViT not available. Cannot load Pixio model."
                    )

                # Instantiate model manualy
                self.model = PixioViT(
                    img_size=256,
                    patch_size=16,
                    embed_dim=model_config["embed_dim"],
                    depth=model_config["depth"],
                    num_heads=model_config["num_heads"],
                    mlp_ratio=4,
                    n_cls_tokens=8,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )

                # Load weights
                logger.info(f"Loading weights from {weights_path}")
                state_dict = torch.load(weights_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
            else:
                # DINOv3 uses 'weights' kwarg via hub
                self.model = torch.hub.load(
                    str(_DINOV3_REPO),
                    model_config["hub_name"],
                    source="local",
                    weights=weights_path,
                )

            self.model = self.model.to(self.device)
            self.model.eval()

            # Get embedding dimension from config (both models have embed_dim attr but config is safer)
            self._embed_dim = model_config["embed_dim"]

            # Create transform
            self.transform = make_transform(self._img_size)

            logger.info(
                f"{backend_name} {self._model_name} loaded: embed_dim={self._embed_dim}, patch_size={self._patch_size}"
            )

        except Exception as e:
            logger.error(f"Failed to load {backend_name}: {e}")
            raise

    def unload_model(self):
        """Unload model and clear caches."""
        # Move model to CPU before deletion to ensure CUDA tensors are freed
        if self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass  # Model might not be on CUDA
            del self.model
        if self.transform is not None:
            del self.transform

        self.model = None
        self.transform = None

        # Clear tensor caches - these may hold CUDA tensors
        for key in list(self._feature_cache.keys()):
            tensor = self._feature_cache.pop(key)
            del tensor
        for key in list(self._descriptor_cache.keys()):
            arr = self._descriptor_cache.pop(key)
            del arr

        # Force garbage collection before clearing CUDA cache
        import gc

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _get_cache_key(self, image_id: str, suffix: str = "") -> str:
        """Generate cache key for an image."""
        return f"{image_id}{suffix}"

    def _get_disk_cache_path(self, cache_key: str, ext: str = ".pt") -> Optional[Path]:
        """Get path for disk cache file."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{cache_key}{ext}"

    def get_image_features(
        self,
        image: Union[np.ndarray, Image.Image, str],
        image_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Extract dense feature map from image.

        Args:
            image: RGB image as numpy array (H,W,3), PIL Image, or path
            image_id: Unique ID for caching
            use_cache: Whether to use cached features

        Returns:
            Feature tensor of shape (1, num_patches, embed_dim)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Check memory cache
        if use_cache and image_id:
            cache_key = self._get_cache_key(image_id, "_features")
            if cache_key in self._feature_cache:
                return self._feature_cache[cache_key]

            # Check disk cache
            disk_path = self._get_disk_cache_path(cache_key)
            if disk_path and disk_path.exists():
                features = torch.load(disk_path, map_location=self.device)
                self._feature_cache[cache_key] = features
                return features

        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Transform and add batch dimension
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract features
        with torch.inference_mode():
            if self._is_pixio:
                # Pixio returns a list of block dicts:
                # [{'patch_tokens': ..., 'cls_tokens': ..., 'patch_tokens_norm': ..., 'cls_tokens_norm': ...}, ...]
                # We want the last block's normalized patch tokens
                output = self.model(img_tensor)  # Returns features from all blocks
                patch_features = output[-1][
                    "patch_tokens_norm"
                ]  # (B, num_patches, embed_dim)
            else:
                # DINOv3 returns a dict with keys:
                # - x_norm_clstoken: (B, embed_dim)
                # - x_norm_patchtokens: (B, num_patches, embed_dim)
                # - x_storage_tokens: (B, 4, embed_dim)
                # - x_prenorm: (B, 1+4+num_patches, embed_dim)
                output = self.model.forward_features(img_tensor)
                patch_features = output[
                    "x_norm_patchtokens"
                ]  # (1, num_patches, embed_dim)

        # Cache
        if use_cache and image_id:
            cache_key = self._get_cache_key(image_id, "_features")
            self._feature_cache[cache_key] = patch_features

            # Save to disk if cache_dir set
            disk_path = self._get_disk_cache_path(cache_key)
            if disk_path:
                torch.save(patch_features, disk_path)

        return patch_features

    def get_cls_token(
        self,
        image: Union[np.ndarray, Image.Image, str],
        image_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Extract CLS token (global image representation).

        Args:
            image: RGB image
            image_id: Optional ID for caching

        Returns:
            CLS token tensor of shape (1, embed_dim)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            if self._is_pixio:
                # Pixio has 8 cls tokens - average them for a single representation
                output = self.model(img_tensor)
                cls_tokens = output[-1]["cls_tokens_norm"]  # (1, 8, embed_dim)
                cls_token = cls_tokens.mean(dim=1)  # (1, embed_dim)
            else:
                output = self.model.forward_features(img_tensor)
                cls_token = output["x_norm_clstoken"]  # (1, embed_dim)

        return cls_token

    def get_object_descriptor(
        self,
        image: Union[np.ndarray, Image.Image, str],
        bbox_xyxy: list[float],
        mask: Optional[np.ndarray] = None,
        image_id: Optional[str] = None,
        annotation_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Compute object descriptor by pooling features in bbox region.

        Args:
            image: RGB image
            bbox_xyxy: Bounding box [x1, y1, x2, y2] in pixel coords
            mask: Optional binary mask for more precise pooling
            image_id: Image ID for caching
            annotation_id: Annotation ID for caching
            use_cache: Whether to use cache

        Returns:
            Object descriptor as numpy array of shape (embed_dim,)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Check cache
        if use_cache and image_id and annotation_id:
            cache_key = f"{image_id}_ann{annotation_id}_desc"
            if cache_key in self._descriptor_cache:
                return self._descriptor_cache[cache_key]

        # Get full image features
        features = self.get_image_features(image, image_id, use_cache)

        # Load image to get dimensions
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
            img_w, img_h = img.size
        elif isinstance(image, np.ndarray):
            img_h, img_w = image.shape[:2]
        else:
            img_w, img_h = image.size

        # Calculate feature map dimensions
        num_patches_h = self._img_size // self._patch_size
        num_patches_w = self._img_size // self._patch_size

        # Reshape features to spatial layout
        # features: (1, num_patches, embed_dim)
        feat_map = features.reshape(1, num_patches_h, num_patches_w, -1)
        feat_map = feat_map.permute(0, 3, 1, 2)  # (1, embed_dim, H, W)

        # Convert bbox to feature map coordinates
        x1, y1, x2, y2 = bbox_xyxy
        fx1 = int(x1 / img_w * num_patches_w)
        fy1 = int(y1 / img_h * num_patches_h)
        fx2 = int(np.ceil(x2 / img_w * num_patches_w))
        fy2 = int(np.ceil(y2 / img_h * num_patches_h))

        # Clamp to valid range
        fx1 = max(0, min(fx1, num_patches_w - 1))
        fy1 = max(0, min(fy1, num_patches_h - 1))
        fx2 = max(fx1 + 1, min(fx2, num_patches_w))
        fy2 = max(fy1 + 1, min(fy2, num_patches_h))

        # Extract region features
        region_feats = feat_map[:, :, fy1:fy2, fx1:fx2]  # (1, embed_dim, rh, rw)

        # Pool to get descriptor
        if mask is not None and mask.sum() > 0:
            # Resize mask to feature map size
            mask_resized = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_resized = mask_resized.to(
                self.device
            )  # Move to same device as features
            mask_resized = F.interpolate(
                mask_resized, size=(num_patches_h, num_patches_w), mode="nearest"
            )
            # Extract region mask
            region_mask = mask_resized[:, :, fy1:fy2, fx1:fx2]

            # Check if region mask has any pixels after downsampling
            # For thin objects, the mask might be empty at this resolution
            if region_mask.sum() > 0:
                # Masked average pooling
                region_feats_masked = region_feats * region_mask
                descriptor = region_feats_masked.sum(dim=(2, 3)) / region_mask.sum()
            else:
                # Mask is empty after downsampling (thin object) - fall back to non-masked
                logger.debug(
                    f"Mask empty after downsampling for bbox {bbox_xyxy}, using non-masked pooling"
                )
                descriptor = region_feats.mean(dim=(2, 3))
        else:
            # Simple average pooling
            descriptor = region_feats.mean(dim=(2, 3))

        # Normalize descriptor
        descriptor = F.normalize(descriptor, dim=-1)
        descriptor = descriptor.squeeze(0).cpu().numpy()

        # Cache
        if use_cache and image_id and annotation_id:
            cache_key = f"{image_id}_ann{annotation_id}_desc"
            self._descriptor_cache[cache_key] = descriptor

        return descriptor

    def compute_similarity(
        self,
        descriptor1: np.ndarray,
        descriptor2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two descriptors.

        Args:
            descriptor1: First descriptor
            descriptor2: Second descriptor

        Returns:
            Similarity score in [-1, 1]
        """
        # Descriptors should already be normalized
        return float(np.dot(descriptor1, descriptor2))

    def find_best_match_in_features(
        self,
        query_descriptor: np.ndarray,
        target_features: torch.Tensor,
        search_bbox: Optional[list[float]] = None,
        img_w: int = 0,
        img_h: int = 0,
    ) -> tuple[int, int, float]:
        """
        Find the patch in target features most similar to query descriptor.

        Args:
            query_descriptor: Query descriptor (embed_dim,)
            target_features: Target feature map (1, num_patches, embed_dim)
            search_bbox: Optional bbox to restrict search [x1, y1, x2, y2]
            img_w, img_h: Original image dimensions (needed if search_bbox provided)

        Returns:
            (patch_y, patch_x, similarity_score)
        """
        num_patches_h = self._img_size // self._patch_size
        num_patches_w = self._img_size // self._patch_size

        # Reshape to spatial
        feat_map = target_features.reshape(num_patches_h, num_patches_w, -1)

        # Convert query to tensor
        query = torch.from_numpy(query_descriptor).to(self.device)

        # Compute similarities
        similarities = F.cosine_similarity(
            feat_map.reshape(-1, feat_map.shape[-1]), query.unsqueeze(0), dim=-1
        ).reshape(num_patches_h, num_patches_w)

        # Apply search region mask if provided
        if search_bbox is not None and img_w > 0 and img_h > 0:
            x1, y1, x2, y2 = search_bbox
            fx1 = int(x1 / img_w * num_patches_w)
            fy1 = int(y1 / img_h * num_patches_h)
            fx2 = int(np.ceil(x2 / img_w * num_patches_w))
            fy2 = int(np.ceil(y2 / img_h * num_patches_h))

            # Create mask
            mask = torch.zeros_like(similarities)
            mask[fy1:fy2, fx1:fx2] = 1
            similarities = similarities * mask + (1 - mask) * (-1)

        # Find best match
        flat_idx = similarities.argmax()
        best_y = int(flat_idx // num_patches_w)
        best_x = int(flat_idx % num_patches_w)
        best_score = float(similarities[best_y, best_x])

        return best_y, best_x, best_score

    def clear_cache(self, image_id: Optional[str] = None):
        """
        Clear cached features.

        Args:
            image_id: Clear only for this image, or all if None
        """
        if image_id is None:
            self._feature_cache.clear()
            self._descriptor_cache.clear()
        else:
            # Clear matching keys
            keys_to_remove = [k for k in self._feature_cache if k.startswith(image_id)]
            for k in keys_to_remove:
                del self._feature_cache[k]

            keys_to_remove = [
                k for k in self._descriptor_cache if k.startswith(image_id)
            ]
            for k in keys_to_remove:
                del self._descriptor_cache[k]


# Global singleton
_embed_service: Optional[EmbedService] = None


def get_embed_service(
    device: str = "cuda", cache_dir: Optional[str] = None
) -> EmbedService:
    """Get the global EmbedService instance."""
    global _embed_service

    if _embed_service is None:
        _embed_service = EmbedService(device=device, cache_dir=cache_dir)

    return _embed_service


def clear_embed_service():
    """Clear the global singleton and free GPU memory."""
    global _embed_service

    if _embed_service is not None:
        _embed_service.unload_model()
        _embed_service = None

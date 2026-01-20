"""
SAM Segmentation Service - wrapper around SAM3 for mask generation.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class CachedMask:
    """A cached segmentation mask with metadata."""

    mask: np.ndarray
    bbox: list[float]
    score: float
    area: int


class SegmentService:
    """
    SAM-based segmentation service.

    Provides mask generation from bounding boxes and point prompts.
    Includes caching for "segment everything" results.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the segmentation service.

        Args:
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = None
        self._processor = None
        self._image_state = None
        self._current_image_id = None

        # Cache for "segment everything" results per image
        # Maps image_id -> list of CachedMask
        self._everything_cache: dict[str, list[CachedMask]] = {}
        self._max_cached_images = 3  # Limit memory usage

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self._processor is not None

    def load_model(self, model_size: str = "large"):
        """
        Load SAM3 model.

        Args:
            model_size: Model size (currently ignored - SAM3 uses large model)
        """
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            logger.info(f"Loading SAM3 model on {self.device}...")

            # Build model with instance interactivity for point/box prompts
            # Note: SAM3 doesn't have model_size parameter - it uses large model
            self.model = build_sam3_image_model(
                bpe_path=None,  # Use default
                device=self.device,
                eval_mode=True,
                enable_inst_interactivity=True,
            )

            # Create processor for image handling
            self._processor = Sam3Processor(
                self.model,
                device=self.device,
            )

            logger.info("SAM3 model loaded successfully")

        except ImportError as e:
            logger.error(f"Failed to import SAM3: {e}")
            raise RuntimeError(
                "SAM3 not available. Please ensure sam3 is installed."
            ) from e
        except Exception as e:
            logger.error(f"Failed to load SAM3 model: {e}")
            raise

    def unload_model(self):
        """Unload model to free memory."""
        # Move model to CPU before deletion to ensure CUDA tensors are freed
        if self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass  # Model might not be on CUDA
            del self.model
        if self._processor is not None:
            del self._processor
        if self._image_state is not None:
            # _image_state may contain CUDA tensors
            if hasattr(self._image_state, "cpu"):
                try:
                    self._image_state.cpu()
                except Exception:
                    pass
            del self._image_state

        self.model = None
        self._processor = None
        self._image_state = None
        self._current_image_id = None

        # Clear everything cache
        self._everything_cache.clear()

        # Force garbage collection before clearing CUDA cache
        import gc

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def set_image(self, image_rgb: np.ndarray, image_id: Optional[str] = None):
        """
        Set the image for segmentation. Caches the image encoding.

        Args:
            image_rgb: RGB image as numpy array (H, W, 3)
            image_id: Optional unique identifier for caching
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Check if we already have this image cached
        if image_id and image_id == self._current_image_id:
            logger.debug(f"Image {image_id} already set, reusing cached state")
            return

        # Ensure image is uint8 RGB
        if image_rgb.dtype != np.uint8:
            image_rgb = (image_rgb * 255).astype(np.uint8)

        # Convert to PIL Image for Sam3Processor
        pil_image = Image.fromarray(image_rgb)

        # Use Sam3Processor to set image and compute backbone features
        with torch.inference_mode():
            self._image_state = self._processor.set_image(pil_image)

        self._current_image_id = image_id
        logger.debug(f"Set image for segmentation (shape={image_rgb.shape})")

    def segment_with_bbox(
        self,
        bbox_xyxy: list[float],
        pos_points: Optional[list[tuple[float, float]]] = None,
        neg_points: Optional[list[tuple[float, float]]] = None,
        multimask_output: bool = False,
    ) -> tuple[np.ndarray, float, list[float]]:
        """
        Generate a segmentation mask from a bounding box and optional point prompts.

        Args:
            bbox_xyxy: Bounding box [x1, y1, x2, y2] in pixel coordinates
            pos_points: Optional positive point prompts [(x, y), ...]
            neg_points: Optional negative point prompts [(x, y), ...]
            multimask_output: Whether to return multiple mask options

        Returns:
            Tuple of (mask, score, refined_bbox):
                - mask: Binary mask as numpy array (H, W)
                - score: Confidence score
                - refined_bbox: Refined bounding box [x1, y1, x2, y2]
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self._image_state is None:
            raise RuntimeError("No image set. Call set_image() first.")

        # Prepare box input (SAM3 uses xyxy format for predict_inst)
        input_box = np.array(bbox_xyxy).reshape(1, 4)

        # Prepare point inputs
        point_coords = None
        point_labels = None

        if pos_points or neg_points:
            all_points = []
            all_labels = []

            if pos_points:
                for pt in pos_points:
                    all_points.append([pt[0], pt[1]])
                    all_labels.append(1)  # Positive

            if neg_points:
                for pt in neg_points:
                    all_points.append([pt[0], pt[1]])
                    all_labels.append(0)  # Negative

            point_coords = np.array(all_points).reshape(-1, 2)
            point_labels = np.array(all_labels)

        # Run prediction
        with torch.inference_mode():
            masks, scores, logits = self.model.predict_inst(
                self._image_state,
                point_coords=point_coords,
                point_labels=point_labels,
                box=input_box,
                multimask_output=multimask_output,
            )

        # Get best mask (highest score)
        if multimask_output and len(masks) > 1:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = float(scores[best_idx])
        else:
            mask = masks[0] if len(masks.shape) > 2 else masks
            score = float(scores[0]) if len(scores) > 0 else 1.0

        # Ensure mask is 2D
        if len(mask.shape) > 2:
            mask = mask.squeeze()

        # Convert to binary uint8
        mask = (mask > 0.5).astype(np.uint8)

        # Compute refined bbox from mask
        from core.masks import mask_to_bbox

        refined_bbox = mask_to_bbox(mask)

        logger.debug(f"Segmentation complete: score={score:.3f}, area={mask.sum()}")

        return mask, score, refined_bbox

    def segment_with_points(
        self,
        pos_points: list[tuple[float, float]],
        neg_points: Optional[list[tuple[float, float]]] = None,
        multimask_output: bool = True,
    ) -> tuple[np.ndarray, float, list[float]]:
        """
        Generate a segmentation mask from point prompts only (no box).

        Args:
            pos_points: Positive point prompts [(x, y), ...]
            neg_points: Optional negative point prompts [(x, y), ...]
            multimask_output: Whether to return multiple mask options

        Returns:
            Tuple of (mask, score, bbox):
                - mask: Binary mask as numpy array (H, W)
                - score: Confidence score
                - bbox: Bounding box [x1, y1, x2, y2] derived from mask
        """
        if not pos_points:
            raise ValueError("At least one positive point is required")

        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self._image_state is None:
            raise RuntimeError("No image set. Call set_image() first.")

        # Prepare points
        all_points = []
        all_labels = []

        for pt in pos_points:
            all_points.append([pt[0], pt[1]])
            all_labels.append(1)

        if neg_points:
            for pt in neg_points:
                all_points.append([pt[0], pt[1]])
                all_labels.append(0)

        point_coords = np.array(all_points).reshape(-1, 2)
        point_labels = np.array(all_labels)

        # Run prediction without box
        with torch.inference_mode():
            masks, scores, logits = self.model.predict_inst(
                self._image_state,
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=multimask_output,
            )

        # Get best mask
        if multimask_output and len(masks) > 1:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = float(scores[best_idx])
        else:
            mask = masks[0] if len(masks.shape) > 2 else masks
            score = float(scores[0]) if len(scores) > 0 else 1.0

        # Ensure mask is 2D
        if len(mask.shape) > 2:
            mask = mask.squeeze()

        # Convert to binary uint8
        mask = (mask > 0.5).astype(np.uint8)

        # Get bbox from mask
        from core.masks import mask_to_bbox

        bbox = mask_to_bbox(mask)

        return mask, score, bbox

    def segment_with_point(
        self,
        x: float,
        y: float,
        bbox_hint: Optional[list[float]] = None,
    ) -> tuple[np.ndarray, float, list[float]]:
        """
        Segment using a single center point, optionally with bbox hint.

        Args:
            x, y: Center point coordinates
            bbox_hint: Optional bbox to help guide segmentation

        Returns:
            Tuple of (mask, score, bbox)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self._image_state is None:
            raise RuntimeError("No image set. Call set_image() first.")

        # Prepare inputs
        point_coords = np.array([[x, y]])
        point_labels = np.array([1])  # Positive point

        box = None
        if bbox_hint:
            box = np.array(bbox_hint)

        # Run prediction
        with torch.inference_mode():
            masks, scores, logits = self.model.predict_inst(
                self._image_state,
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )

        # Get best mask
        if len(masks) > 1:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = float(scores[best_idx])
        else:
            mask = masks[0] if len(masks.shape) > 2 else masks
            score = float(scores[0]) if len(scores) > 0 else 1.0

        # Ensure mask is 2D
        if len(mask.shape) > 2:
            mask = mask.squeeze()

        # Convert to binary uint8
        mask = (mask > 0.5).astype(np.uint8)

        # Get bbox from mask
        from core.masks import mask_to_bbox

        bbox = mask_to_bbox(mask)

        return mask, score, bbox

    def segment_everything(
        self,
        image_rgb: Optional[np.ndarray] = None,
        image_id: Optional[str] = None,
        min_mask_area: int = 100,
        nms_iou_threshold: float = 0.7,
        use_cache: bool = True,
    ) -> list[CachedMask]:
        """
        Segment all objects in the image using automatic mask generation.

        Uses SAM3's automatic segmentation or grid-based point prompting
        to find all objects. Results are cached per image.

        Args:
            image_rgb: RGB image as numpy array (H, W, 3). If None, uses current image.
            image_id: Unique identifier for caching
            min_mask_area: Minimum mask area to keep (in pixels)
            nms_iou_threshold: IoU threshold for non-max suppression
            use_cache: Whether to use/store cached results

        Returns:
            List of CachedMask objects sorted by area (largest first)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Check cache first
        if use_cache and image_id and image_id in self._everything_cache:
            logger.debug(f"Using cached everything masks for {image_id}")
            return self._everything_cache[image_id]

        # Set image if provided
        if image_rgb is not None:
            self.set_image(image_rgb, image_id)
        elif self._image_state is None:
            raise RuntimeError("No image set. Call set_image() or provide image_rgb.")

        # Get image dimensions from state
        # SAM3's image state is a dict with original_height and original_width
        img_h = self._image_state["original_height"]
        img_w = self._image_state["original_width"]

        logger.info(f"Running segment_everything on {img_w}x{img_h} image...")

        # Strategy: Use grid of points to prompt SAM
        # SAM3 doesn't have a built-in "segment everything" like SAM2,
        # so we use a grid-based approach
        all_masks: list[CachedMask] = []

        # Create grid of points
        grid_size = 32  # Points per dimension
        point_spacing_x = img_w / grid_size
        point_spacing_y = img_h / grid_size

        with torch.inference_mode():
            for gy in range(grid_size):
                for gx in range(grid_size):
                    point_x = (gx + 0.5) * point_spacing_x
                    point_y = (gy + 0.5) * point_spacing_y

                    point_coords = np.array([[point_x, point_y]])
                    point_labels = np.array([1])

                    try:
                        masks, scores, logits = self.model.predict_inst(
                            self._image_state,
                            point_coords=point_coords,
                            point_labels=point_labels,
                            box=None,
                            multimask_output=True,
                        )

                        # Process each mask output
                        for i in range(len(masks)):
                            mask = masks[i]
                            score = float(scores[i])

                            if len(mask.shape) > 2:
                                mask = mask.squeeze()
                            mask = (mask > 0.5).astype(np.uint8)

                            area = int(mask.sum())
                            if area < min_mask_area:
                                continue

                            from core.masks import mask_to_bbox

                            bbox = mask_to_bbox(mask)

                            all_masks.append(
                                CachedMask(mask=mask, bbox=bbox, score=score, area=area)
                            )
                    except Exception as e:
                        logger.debug(f"Grid point ({gx}, {gy}) failed: {e}")
                        continue

        logger.info(f"Found {len(all_masks)} raw masks before NMS")

        # Apply non-max suppression to remove duplicates
        if all_masks:
            all_masks = self._nms_masks(all_masks, nms_iou_threshold)

        # Sort by area (largest first)
        all_masks.sort(key=lambda m: m.area, reverse=True)

        logger.info(f"segment_everything complete: {len(all_masks)} unique masks")

        # Cache results
        if use_cache and image_id:
            # Manage cache size
            if len(self._everything_cache) >= self._max_cached_images:
                # Remove oldest entry
                oldest_key = next(iter(self._everything_cache))
                del self._everything_cache[oldest_key]

            self._everything_cache[image_id] = all_masks

        return all_masks

    def _nms_masks(
        self, masks: list[CachedMask], iou_threshold: float
    ) -> list[CachedMask]:
        """
        Apply non-max suppression to remove duplicate masks.

        Args:
            masks: List of CachedMask to filter
            iou_threshold: IoU threshold above which masks are considered duplicates

        Returns:
            Filtered list of masks
        """
        from core.masks import mask_iou

        if not masks:
            return []

        # Sort by score (highest first)
        masks = sorted(masks, key=lambda m: m.score, reverse=True)

        keep = []
        for mask in masks:
            is_duplicate = False
            for kept_mask in keep:
                iou = mask_iou(mask.mask, kept_mask.mask)
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(mask)

        return keep

    def get_cached_masks(self, image_id: str) -> Optional[list[CachedMask]]:
        """Get cached everything masks for an image if available."""
        return self._everything_cache.get(image_id)

    def clear_everything_cache(self, image_id: Optional[str] = None):
        """Clear the everything cache for a specific image or all images."""
        if image_id:
            self._everything_cache.pop(image_id, None)
        else:
            self._everything_cache.clear()

    def find_mask_at_point(
        self, image_id: str, x: float, y: float, prefer_smallest: bool = True
    ) -> Optional[CachedMask]:
        """
        Find a cached mask that contains the given point.

        Args:
            image_id: Image ID to search in cached masks
            x, y: Point coordinates
            prefer_smallest: If True, return smallest containing mask (like legacy)

        Returns:
            CachedMask if found, None otherwise
        """
        cached = self._everything_cache.get(image_id)
        if not cached:
            return None

        candidates = []
        ix, iy = int(x), int(y)

        for mask_obj in cached:
            h, w = mask_obj.mask.shape
            if 0 <= iy < h and 0 <= ix < w and mask_obj.mask[iy, ix] > 0:
                candidates.append(mask_obj)

        if not candidates:
            return None

        if prefer_smallest:
            return min(candidates, key=lambda m: m.area)
        else:
            return max(candidates, key=lambda m: m.score)

    def find_best_mask_by_iou(
        self,
        image_id: str,
        reference_mask: np.ndarray,
        min_iou: float = 0.1,
        min_area_ratio: float = 0.3,
    ) -> Optional[tuple[CachedMask, float]]:
        """
        Find the cached mask with best IoU to a reference mask.

        This is similar to the legacy approach of matching DINO predictions
        to FastSAM masks.

        Args:
            image_id: Image ID to search in cached masks
            reference_mask: Reference mask to match against
            min_iou: Minimum IoU to consider a match
            min_area_ratio: Minimum area ratio (cached_area / ref_area)

        Returns:
            Tuple of (best_mask, iou) if found, None otherwise
        """
        from core.masks import mask_iou

        cached = self._everything_cache.get(image_id)
        if not cached:
            return None

        ref_area = reference_mask.sum()
        if ref_area == 0:
            return None

        best_mask = None
        best_iou = min_iou

        for mask_obj in cached:
            # Check area ratio
            area_ratio = mask_obj.area / ref_area
            if area_ratio < min_area_ratio:
                continue

            # Compute IoU
            iou = mask_iou(mask_obj.mask, reference_mask)
            if iou > best_iou:
                best_iou = iou
                best_mask = mask_obj

        if best_mask is not None:
            return (best_mask, best_iou)
        return None


# Global singleton (lazy loaded)
_segment_service: Optional[SegmentService] = None


def get_segment_service(device: str = "cuda") -> SegmentService:
    """Get the global SegmentService instance."""
    global _segment_service

    if _segment_service is None:
        _segment_service = SegmentService(device=device)

    return _segment_service


def clear_segment_service():
    """Clear the global singleton and free GPU memory."""
    global _segment_service

    if _segment_service is not None:
        _segment_service.unload_model()
        _segment_service = None

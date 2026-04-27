"""
Propagation module - Track objects across frames using embeddings + SAM.

Supports two propagation modes:
1. Peak-based (default): Find similarity peaks, prompt SAM at those locations
2. Dense (legacy-style): Use dense patch correspondence to propagate mask probabilities
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Literal, TYPE_CHECKING
from dataclasses import dataclass
import logging

if TYPE_CHECKING:
    from ml.embed import EmbedService
    from ml.segment import SegmentService

logger = logging.getLogger(__name__)


@dataclass
class BboxCandidate:
    """A candidate bounding box with score."""

    bbox_xyxy: list[float]
    score: float
    source: str = "propagated"


@dataclass
class PropagationResult:
    """Result of a propagation attempt."""

    bbox: list[float]
    mask: np.ndarray
    confidence: float
    fallback_used: bool
    area_ratio: float
    method: str  # "peak", "dense", or "iou_match"
    iou_score: Optional[float] = None  # IoU with dense prediction (if verified)


class PropagationError(Exception):
    """Base exception for propagation failures."""

    pass


class PropagationSizeMismatchError(PropagationError):
    """Object size changed too much between frames."""

    pass


class PropagationNotFoundError(PropagationError):
    """Could not find object in target image."""

    pass


def generate_candidates(
    prev_bbox: list[float],
    img_w: int,
    img_h: int,
    search_expand: float = 2.0,
    num_scales: int = 5,
    num_positions: int = 9,
    aspect_ratio_tolerance: float = 0.2,
) -> list[list[float]]:
    """
    Generate candidate bounding boxes around the previous bbox location.

    Uses a multi-scale sliding window approach within an expanded search region.

    Args:
        prev_bbox: Previous bbox [x1, y1, x2, y2]
        img_w, img_h: Image dimensions
        search_expand: Factor to expand search region (2.0 = 2x the bbox size)
        num_scales: Number of scale variations to try
        num_positions: Number of position variations (3x3 grid = 9)
        aspect_ratio_tolerance: How much to vary aspect ratio

    Returns:
        List of candidate bboxes [[x1, y1, x2, y2], ...]
    """
    x1, y1, x2, y2 = prev_bbox
    prev_w = x2 - x1
    prev_h = y2 - y1
    prev_cx = (x1 + x2) / 2
    prev_cy = (y1 + y2) / 2

    # Define search region (expanded around previous bbox)
    search_w = prev_w * search_expand
    search_h = prev_h * search_expand

    # Scale factors to try (centered around 1.0)
    scale_range = 0.3  # ±30%
    scales = np.linspace(1.0 - scale_range, 1.0 + scale_range, num_scales)

    # Position offsets (grid within search region)
    grid_size = int(np.sqrt(num_positions))
    offset_range_x = search_w / 4  # Don't go too far
    offset_range_y = search_h / 4

    if grid_size > 1:
        offsets_x = np.linspace(-offset_range_x, offset_range_x, grid_size)
        offsets_y = np.linspace(-offset_range_y, offset_range_y, grid_size)
    else:
        offsets_x = [0]
        offsets_y = [0]

    candidates = []

    for scale in scales:
        for off_x in offsets_x:
            for off_y in offsets_y:
                # New center
                cx = prev_cx + off_x
                cy = prev_cy + off_y

                # New size
                w = prev_w * scale
                h = prev_h * scale

                # Create bbox
                new_x1 = cx - w / 2
                new_y1 = cy - h / 2
                new_x2 = cx + w / 2
                new_y2 = cy + h / 2

                # Clamp to image bounds
                new_x1 = max(0, min(new_x1, img_w - 1))
                new_y1 = max(0, min(new_y1, img_h - 1))
                new_x2 = max(new_x1 + 1, min(new_x2, img_w))
                new_y2 = max(new_y1 + 1, min(new_y2, img_h))

                # Skip if too small
                if (new_x2 - new_x1) < 10 or (new_y2 - new_y1) < 10:
                    continue

                candidates.append([new_x1, new_y1, new_x2, new_y2])

    # Remove duplicates (approximately)
    unique_candidates = []
    for cand in candidates:
        is_dup = False
        for existing in unique_candidates:
            if _bbox_iou(cand, existing) > 0.9:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(cand)

    return unique_candidates


def _bbox_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Compute IoU between two bboxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _translate_bbox_hint(
    source_bbox: list[float],
    center_x: float,
    center_y: float,
    img_w: int,
    img_h: int,
    scale: float = 1.15,
) -> list[float]:
    """Build a padded square hint around a new center point.

    Using the larger source bbox side makes the hint much more tolerant to
    object rotation than reusing the exact previous width/height.
    """
    side = max(
        1.0,
        max(source_bbox[2] - source_bbox[0], source_bbox[3] - source_bbox[1]) * scale,
    )

    x1 = center_x - side / 2
    y1 = center_y - side / 2
    x2 = center_x + side / 2
    y2 = center_y + side / 2

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(img_w), x2)
    y2 = min(float(img_h), y2)

    if x2 <= x1:
        x2 = min(float(img_w), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(img_h), y1 + 1.0)

    return [x1, y1, x2, y2]


def score_candidates(
    candidates: list[list[float]],
    prev_descriptor: np.ndarray,
    target_features: "torch.Tensor",
    prev_bbox: list[float],
    img_w: int,
    img_h: int,
    embed_service: "EmbedService",
    spatial_weight: float = 0.1,
) -> list[BboxCandidate]:
    """
    Score candidate bboxes using embedding similarity.

    Args:
        candidates: List of candidate bboxes
        prev_descriptor: Descriptor of the object from previous frame
        target_features: Feature map of target image
        prev_bbox: Previous bbox (for spatial prior)
        img_w, img_h: Image dimensions
        embed_service: EmbedService instance
        spatial_weight: Weight for spatial distance penalty

    Returns:
        List of BboxCandidate sorted by score (highest first)
    """
    num_patches_h = embed_service._img_size // embed_service._patch_size
    num_patches_w = embed_service._img_size // embed_service._patch_size

    # Reshape features to spatial
    feat_map = target_features.reshape(1, num_patches_h, num_patches_w, -1)
    feat_map = feat_map.permute(0, 3, 1, 2)  # (1, embed_dim, H, W)

    prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
    prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
    max_dist = np.sqrt(img_w**2 + img_h**2)

    scored_candidates = []

    for bbox in candidates:
        # Convert bbox to feature coordinates
        x1, y1, x2, y2 = bbox
        fx1 = int(x1 / img_w * num_patches_w)
        fy1 = int(y1 / img_h * num_patches_h)
        fx2 = int(np.ceil(x2 / img_w * num_patches_w))
        fy2 = int(np.ceil(y2 / img_h * num_patches_h))

        # Clamp
        fx1 = max(0, min(fx1, num_patches_w - 1))
        fy1 = max(0, min(fy1, num_patches_h - 1))
        fx2 = max(fx1 + 1, min(fx2, num_patches_w))
        fy2 = max(fy1 + 1, min(fy2, num_patches_h))

        # Extract and pool region features
        region_feats = feat_map[:, :, fy1:fy2, fx1:fx2]
        region_desc = region_feats.mean(dim=(2, 3))
        region_desc = F.normalize(region_desc, dim=-1)
        region_desc = region_desc.squeeze(0).cpu().numpy()

        # Compute similarity
        similarity = float(np.dot(prev_descriptor, region_desc))

        # Spatial prior (penalize distance from previous location)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dist = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
        spatial_penalty = spatial_weight * (dist / max_dist)

        # Combined score
        score = similarity - spatial_penalty

        scored_candidates.append(
            BboxCandidate(bbox_xyxy=bbox, score=score, source="propagated")
        )

    # Sort by score (highest first)
    scored_candidates.sort(key=lambda x: x.score, reverse=True)

    return scored_candidates


def propose_bboxes(
    prev_descriptor: np.ndarray,
    target_features: "torch.Tensor",
    prev_bbox: list[float],
    img_w: int,
    img_h: int,
    embed_service: "EmbedService",
    top_k: int = 5,
) -> list[BboxCandidate]:
    """
    Propose top-k bboxes for an object in a new frame.

    Args:
        prev_descriptor: Object descriptor from previous frame
        target_features: Feature map of target frame
        prev_bbox: Bbox from previous frame
        img_w, img_h: Target image dimensions
        embed_service: EmbedService instance
        top_k: Number of candidates to return

    Returns:
        Top-k BboxCandidates sorted by score
    """
    # Generate candidates
    candidates = generate_candidates(prev_bbox, img_w, img_h)

    if not candidates:
        return []

    # Score candidates
    scored = score_candidates(
        candidates,
        prev_descriptor,
        target_features,
        prev_bbox,
        img_w,
        img_h,
        embed_service,
    )

    return scored[:top_k]


class PropagateService:
    """
    Service for propagating annotations across frames.

    Combines:
    - DINOv3 embeddings for object matching
    - Candidate bbox generation
    - SAM for mask refinement

    Supports two modes:
    - "peak" (default): Find similarity peaks, prompt SAM
    - "dense": Legacy-style dense correspondence for mask propagation
    """

    def __init__(
        self,
        embed_service: "EmbedService",
        segment_service: "SegmentService",
    ):
        """
        Initialize propagation service.

        Args:
            embed_service: EmbedService for feature extraction
            segment_service: SegmentService for mask generation
        """
        self.embed_service = embed_service
        self.segment_service = segment_service

    def unload_model(self):
        """Clear references (actual unload happens in embed/segment services)."""
        self.embed_service = None
        self.segment_service = None

    def propagate_dense(
        self,
        source_mask: np.ndarray,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        top_k: int = 5,
        temperature: float = 0.2,
    ) -> np.ndarray:
        """
        Propagate mask using dense feature correspondence (legacy DINO approach).

        This computes patch-to-patch similarities and uses them to propagate
        mask probabilities from source to target, producing a coarse mask
        prediction without requiring SAM.

        Args:
            source_mask: Binary mask from source image (H, W)
            source_features: Source image features (1, num_patches, embed_dim)
            target_features: Target image features (1, num_patches, embed_dim)
            top_k: Number of top matches to consider per target patch
            temperature: Softmax temperature (lower = sharper matching)

        Returns:
            Probability map for target image (num_patches_h, num_patches_w)
        """
        # Get patch dimensions
        num_patches = self.embed_service._img_size // self.embed_service._patch_size
        embed_dim = source_features.shape[-1]

        # Reshape features to spatial layout: (H, W, D)
        source_feats = source_features.squeeze(0).reshape(
            num_patches, num_patches, embed_dim
        )
        target_feats = target_features.squeeze(0).reshape(
            num_patches, num_patches, embed_dim
        )

        # Resize mask to patch resolution
        source_mask_resized = torch.from_numpy(source_mask).float()
        source_mask_resized = F.interpolate(
            source_mask_resized.unsqueeze(0).unsqueeze(0),
            size=(num_patches, num_patches),
            mode="nearest",
        ).squeeze()

        # Create one-hot encoding (background=0, foreground=1)
        # Shape: (H, W, 2) where [..., 0] = background prob, [..., 1] = foreground prob
        source_probs = torch.zeros(
            num_patches, num_patches, 2, device=source_feats.device
        )
        source_probs[..., 0] = 1 - source_mask_resized.to(source_feats.device)
        source_probs[..., 1] = source_mask_resized.to(source_feats.device)

        # Compute dot product similarity between all target and source patches
        # target_feats: (Ht, Wt, D), source_feats: (Hs, Ws, D)
        # Result: (Ht, Wt, Hs, Ws) - similarity of each target patch to each source patch
        dot = torch.einsum("ijd,uvd->ijuv", target_feats, source_feats)

        # Flatten spatial dimensions for top-k selection
        # Shape: (Ht*Wt, Hs*Ws)
        dot_flat = dot.flatten(0, 1).flatten(1, 2)

        # Top-k filtering: keep only top-k matches per target patch
        k_th_values = torch.topk(dot_flat, dim=1, k=top_k).values[:, -1:]
        dot_flat = torch.where(
            dot_flat >= k_th_values,
            dot_flat,
            torch.tensor(-torch.inf, device=dot_flat.device),
        )

        # Softmax to get attention weights
        weights = F.softmax(dot_flat / temperature, dim=1)

        # Propagate mask probabilities: weighted sum of source probs
        # weights: (Ht*Wt, Hs*Ws), source_probs: (Hs, Ws, 2) -> (Hs*Ws, 2)
        source_probs_flat = source_probs.flatten(0, 1)  # (Hs*Ws, 2)
        target_probs = torch.mm(weights, source_probs_flat)  # (Ht*Wt, 2)

        # Normalize probabilities
        target_probs = target_probs / (target_probs.sum(dim=1, keepdim=True) + 1e-8)

        # Reshape to spatial and extract foreground probability
        target_probs = target_probs.reshape(num_patches, num_patches, 2)
        foreground_probs = target_probs[..., 1].cpu().numpy()

        return foreground_probs

    def propagate_with_dense_fallback(
        self,
        source_image: np.ndarray,
        source_bbox: list[float],
        source_mask: Optional[np.ndarray],
        target_image: np.ndarray,
        source_image_id: Optional[str] = None,
        target_image_id: Optional[str] = None,
        annotation_id: Optional[int] = None,
        mode: Literal["peak", "dense", "auto"] = "auto",
        iou_verify: bool = True,
        iou_threshold: float = 0.3,
        use_cached_masks: bool = False,
        bbox_hint_scale: float = 1.15,
        **kwargs,
    ) -> Optional[PropagationResult]:
        """
        Propagate annotation with optional dense fallback and IoU verification.

        Modes:
        - "peak": Use peak-based propagation only (current default behavior)
        - "dense": Use dense propagation to create coarse mask, then refine with SAM
        - "auto": Try peak first, fall back to dense if it fails

        IoU verification (when enabled):
        - Computes dense propagation mask as "ground truth"
        - Verifies SAM result has sufficient IoU with dense prediction
        - Rejects results with low IoU

        Args:
            source_image: Source image (RGB numpy array)
            source_bbox: Source bbox [x1, y1, x2, y2]
            source_mask: Source mask (required for dense mode)
            target_image: Target image (RGB numpy array)
            source_image_id: ID for caching source features
            target_image_id: ID for caching target features
            annotation_id: Annotation ID for caching
            mode: Propagation mode ("peak", "dense", or "auto")
            iou_verify: Whether to verify results against dense prediction
            iou_threshold: Minimum IoU with dense prediction to accept
            use_cached_masks: Whether to use/check SAM everything cache
            **kwargs: Additional arguments passed to propagate_annotation

        Returns:
            PropagationResult or None if propagation fails
        """
        from core.masks import mask_iou, mask_to_bbox

        img_h, img_w = target_image.shape[:2]

        # Get features for both images
        source_features = self.embed_service.get_image_features(
            source_image, image_id=source_image_id
        )
        target_features = self.embed_service.get_image_features(
            target_image, image_id=target_image_id
        )

        # Compute dense mask prediction if needed for dense mode or IoU verification
        dense_mask = None
        if source_mask is not None and (mode in ["dense", "auto"] or iou_verify):
            dense_probs = self.propagate_dense(
                source_mask, source_features, target_features
            )
            # Upsample to image resolution
            dense_probs_tensor = torch.from_numpy(dense_probs).unsqueeze(0).unsqueeze(0)
            dense_probs_upsampled = (
                F.interpolate(
                    dense_probs_tensor,
                    size=(img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .numpy()
            )
            dense_mask = (dense_probs_upsampled > 0.5).astype(np.uint8)

            logger.debug(f"Dense mask area: {dense_mask.sum()}")

        # Calculate source area for size checking
        source_area = (source_bbox[2] - source_bbox[0]) * (
            source_bbox[3] - source_bbox[1]
        )

        result = None
        method_used = None

        # Try peak-based propagation first (unless mode is "dense")
        if mode in ["peak", "auto"]:
            try:
                peak_result = self.propagate_annotation(
                    source_image=source_image,
                    source_bbox=source_bbox,
                    source_mask=source_mask,
                    target_image=target_image,
                    source_image_id=source_image_id,
                    target_image_id=target_image_id,
                    annotation_id=annotation_id,
                    bbox_hint_scale=bbox_hint_scale,
                    **kwargs,
                )
                if peak_result:
                    bbox, mask, confidence, fallback_used, area_ratio = peak_result

                    # IoU verification against dense prediction
                    iou_score = None
                    if iou_verify and dense_mask is not None:
                        iou_score = mask_iou(mask, dense_mask)
                        logger.debug(f"Peak result IoU with dense: {iou_score:.3f}")

                        if iou_score < iou_threshold:
                            logger.info(
                                f"Peak result rejected: IoU {iou_score:.3f} < {iou_threshold}"
                            )
                            # Don't use this result
                        else:
                            result = PropagationResult(
                                bbox=bbox,
                                mask=mask,
                                confidence=confidence,
                                fallback_used=fallback_used,
                                area_ratio=area_ratio,
                                method="peak",
                                iou_score=iou_score,
                            )
                            method_used = "peak"
                    else:
                        result = PropagationResult(
                            bbox=bbox,
                            mask=mask,
                            confidence=confidence,
                            fallback_used=fallback_used,
                            area_ratio=area_ratio,
                            method="peak",
                            iou_score=None,
                        )
                        method_used = "peak"
            except (PropagationSizeMismatchError, PropagationNotFoundError) as e:
                logger.info(f"Peak propagation failed: {e}")
                if mode == "peak":
                    raise

        # Try dense + cached mask matching (if peak failed or mode is dense)
        if result is None and mode in ["dense", "auto"] and dense_mask is not None:
            logger.info("Attempting dense propagation with SAM mask matching...")

            # Use dense mask center to prompt SAM
            # Find center of dense mask
            rows, cols = np.where(dense_mask > 0)
            if len(rows) > 0:
                center_y = int(np.mean(rows))
                center_x = int(np.mean(cols))

                # Set image for SAM
                self.segment_service.set_image(target_image, target_image_id)

                try:
                    dense_bbox = mask_to_bbox(dense_mask)
                    bbox_hint = _translate_bbox_hint(
                        dense_bbox,
                        center_x,
                        center_y,
                        img_w,
                        img_h,
                        scale=bbox_hint_scale,
                    )
                    mask, sam_score, bbox = self.segment_service.segment_with_point(
                        center_x, center_y, bbox_hint=bbox_hint
                    )

                    # Verify IoU
                    iou_score = mask_iou(mask, dense_mask)
                    if iou_score >= iou_threshold:
                        area_ratio = (
                            mask.sum() / source_area if source_area > 0 else 1.0
                        )

                        # Get descriptor similarity for confidence
                        source_desc = self.embed_service.get_object_descriptor(
                            source_image,
                            source_bbox,
                            mask=source_mask,
                            image_id=source_image_id,
                            annotation_id=annotation_id,
                        )
                        target_desc = self.embed_service.get_object_descriptor(
                            target_image, bbox, mask=mask, image_id=target_image_id
                        )
                        embed_sim = self.embed_service.compute_similarity(
                            source_desc, target_desc
                        )

                        confidence = 0.5 * embed_sim + 0.3 * sam_score + 0.2 * iou_score

                        result = PropagationResult(
                            bbox=bbox,
                            mask=mask,
                            confidence=confidence,
                            fallback_used=True,
                            area_ratio=area_ratio,
                            method="dense",
                            iou_score=iou_score,
                        )
                        method_used = "dense"
                        logger.info(
                            f"Dense propagation succeeded: IoU={iou_score:.3f}, sim={embed_sim:.3f}"
                        )
                    else:
                        logger.warning(
                            f"Dense SAM result rejected: IoU {iou_score:.3f} < {iou_threshold}"
                        )
                except Exception as e:
                    logger.warning(f"Dense SAM prompting failed: {e}")

        if result:
            logger.info(f"Propagation succeeded using method: {method_used}")
            return result

        raise PropagationNotFoundError("All propagation methods failed")

    def propagate_annotation(
        self,
        source_image: np.ndarray,
        source_bbox: list[float],
        source_mask: Optional[np.ndarray],
        target_image: np.ndarray,
        source_image_id: Optional[str] = None,
        target_image_id: Optional[str] = None,
        annotation_id: Optional[int] = None,
        bbox_hint_scale: float = 1.15,
        top_k: int = 5,
        min_score: float = 0.5,
        search_expansion: float = 0.3,
        size_min_ratio: float = 0.8,
        size_max_ratio: float = 1.2,
        stop_on_size_mismatch: bool = True,
    ) -> Optional[tuple[list[float], np.ndarray, float, bool, float]]:
        """
        Propagate an annotation from source to target image.

        Strategy:
        1. Compute similarity map between source descriptor and target features
        2. Find peak location in similarity map
        3. Use peak point to prompt SAM
        4. Verify result with embedding comparison
        5. Check size consistency - reject if bbox size differs too much from source

        Args:
            source_image: Source image (RGB numpy array)
            source_bbox: Source bbox [x1, y1, x2, y2]
            source_mask: Source mask (optional, for better descriptor)
            target_image: Target image (RGB numpy array)
            source_image_id: ID for caching source features
            target_image_id: ID for caching target features
            annotation_id: Annotation ID for caching
            top_k: Number of peak locations to try
            min_score: Minimum similarity score threshold
            search_expansion: Unused, kept for API compatibility
            size_min_ratio: Minimum area ratio (e.g. 0.8 for 80%)
            size_max_ratio: Maximum area ratio (e.g. 1.2 for 120%)
            stop_on_size_mismatch: If True, return None when no size-OK result; if False, use fallback

        Returns:
            Tuple of (bbox, mask, confidence, fallback_used, area_ratio) or None if propagation fails
            fallback_used is True if best-similarity result was used due to no size-match
            area_ratio is the size ratio (new_area / old_area)
        """
        import torch.nn.functional as F

        img_h, img_w = target_image.shape[:2]

        # Calculate source bbox area for size comparison
        source_area = (source_bbox[2] - source_bbox[0]) * (
            source_bbox[3] - source_bbox[1]
        )

        logger.info(
            f"Source bbox: {[int(x) for x in source_bbox]}, area: {source_area:.0f}"
        )

        # Get source descriptor
        source_descriptor = self.embed_service.get_object_descriptor(
            source_image,
            source_bbox,
            mask=source_mask,
            image_id=source_image_id,
            annotation_id=annotation_id,
        )

        logger.info(
            f"Source descriptor: norm={np.linalg.norm(source_descriptor):.4f}, "
            f"min={source_descriptor.min():.4f}, max={source_descriptor.max():.4f}"
        )

        # Get target features
        target_features = self.embed_service.get_image_features(
            target_image,
            image_id=target_image_id,
        )

        # Compute similarity map
        num_patches = self.embed_service._img_size // self.embed_service._patch_size

        # Reshape features and compute similarity
        feats = target_features.squeeze(0)  # (num_patches, embed_dim)
        feats = F.normalize(feats, dim=-1)

        desc_tensor = torch.from_numpy(source_descriptor).to(feats.device).float()
        similarity = torch.mv(feats, desc_tensor)  # (num_patches,)
        similarity_map = (
            similarity.reshape(num_patches, num_patches).float().cpu().numpy()
        )

        # Find top-k peaks in similarity map
        flat_sim = similarity_map.flatten()
        peak_indices = np.argsort(flat_sim)[::-1]  # Descending order

        # Set target image for SAM
        self.segment_service.set_image(target_image, target_image_id)

        # Try top peaks until we get a good match
        tried_points = []

        # Store results: size-acceptable results and all valid results for fallback
        size_acceptable_results: list[
            tuple[list[float], np.ndarray, float, float, float]
        ] = []  # bbox, mask, confidence, similarity, area_ratio
        all_valid_results: list[
            tuple[list[float], np.ndarray, float, float, float]
        ] = []

        for i in range(min(top_k * 3, len(peak_indices))):  # Try more peaks
            peak_idx = peak_indices[i]
            py, px = divmod(peak_idx, num_patches)

            # Convert patch coords to pixel coords (center of patch)
            point_x = (px + 0.5) / num_patches * img_w
            point_y = (py + 0.5) / num_patches * img_h

            # Skip if too close to already tried points
            too_close = False
            for tx, ty in tried_points:
                if (
                    abs(point_x - tx) < img_w * 0.05
                    and abs(point_y - ty) < img_h * 0.05
                ):
                    too_close = True
                    break
            if too_close:
                continue

            tried_points.append((point_x, point_y))

            if len(tried_points) > top_k:
                break

            peak_sim = flat_sim[peak_idx]
            logger.info(
                f"Trying peak #{len(tried_points)}: ({point_x:.1f}, {point_y:.1f}), sim={peak_sim:.3f}"
            )

            try:
                bbox_hint = _translate_bbox_hint(
                    source_bbox,
                    point_x,
                    point_y,
                    img_w,
                    img_h,
                    scale=bbox_hint_scale,
                )

                # Use point to prompt SAM
                mask, sam_score, refined_bbox = self.segment_service.segment_with_point(
                    point_x,
                    point_y,
                    bbox_hint=bbox_hint,
                )

                # Verify with embedding - compute descriptor of result
                target_descriptor = self.embed_service.get_object_descriptor(
                    target_image,
                    refined_bbox,
                    mask=mask,
                    image_id=target_image_id,
                )

                embed_similarity = self.embed_service.compute_similarity(
                    source_descriptor, target_descriptor
                )

                # Combined confidence
                confidence = 0.6 * embed_similarity + 0.4 * sam_score

                # Check size consistency
                result_area = (refined_bbox[2] - refined_bbox[0]) * (
                    refined_bbox[3] - refined_bbox[1]
                )
                area_ratio = result_area / source_area if source_area > 0 else 1.0
                size_acceptable = size_min_ratio <= area_ratio <= size_max_ratio

                logger.info(
                    f"  Result: sim={embed_similarity:.3f}, sam={sam_score:.3f}, "
                    f"conf={confidence:.3f}, area_ratio={area_ratio:.2f}, size_ok={size_acceptable}, "
                    f"bbox={[int(x) for x in refined_bbox]}"
                )

                if embed_similarity >= min_score:
                    # Store as valid result
                    all_valid_results.append(
                        (refined_bbox, mask, confidence, embed_similarity, area_ratio)
                    )

                    if size_acceptable:
                        # Store as size-acceptable result
                        size_acceptable_results.append(
                            (
                                refined_bbox,
                                mask,
                                confidence,
                                embed_similarity,
                                area_ratio,
                            )
                        )
                        logger.debug(
                            f"  Added to size-acceptable results (now {len(size_acceptable_results)})"
                        )

            except Exception as e:
                logger.warning(f"SAM failed for peak point: {e}")
                continue
        # Log all candidates for debugging
        logger.info("=== Propagation candidates summary ===")
        logger.info(f"Size-acceptable results ({len(size_acceptable_results)}):")
        for i, (bbox, _, conf, sim, ratio) in enumerate(size_acceptable_results):
            logger.info(
                f"  {i + 1}. sim={sim:.3f}, ratio={ratio:.2f}, conf={conf:.3f}, bbox={[int(x) for x in bbox]}"
            )
        logger.info(f"All valid results ({len(all_valid_results)}):")
        for i, (bbox, _, conf, sim, ratio) in enumerate(all_valid_results):
            marker = " <-- size OK" if size_min_ratio <= ratio <= size_max_ratio else ""
            logger.info(
                f"  {i + 1}. sim={sim:.3f}, ratio={ratio:.2f}, conf={conf:.3f}, bbox={[int(x) for x in bbox]}{marker}"
            )
        logger.info("======================================")

        # Select best result
        if size_acceptable_results:
            # Return best similarity among size-acceptable results
            best = max(
                size_acceptable_results, key=lambda x: x[3]
            )  # Sort by similarity
            logger.info(
                f"Propagation succeeded with size-acceptable result: sim={best[3]:.3f}, ratio={best[4]:.2f}"
            )
            return (
                best[0],
                best[1],
                best[2],
                False,
                best[4],
            )  # fallback_used = False, area_ratio

        if all_valid_results and not stop_on_size_mismatch:
            # Fallback: return best similarity overall (only if allowed)
            best = max(all_valid_results, key=lambda x: x[3])
            logger.info(
                f"Propagation using fallback (size mismatch): sim={best[3]:.3f}, ratio={best[4]:.2f}"
            )
            return (
                best[0],
                best[1],
                best[2],
                True,
                best[4],
            )  # fallback_used = True, area_ratio

        if all_valid_results and stop_on_size_mismatch:
            best = max(all_valid_results, key=lambda x: x[3])
            logger.warning(
                f"Size mismatch detected, stopping propagation (stop_on_size_mismatch=True). "
                f"Best ratio was {best[4]:.2f}, allowed: {size_min_ratio}-{size_max_ratio}"
            )
            raise PropagationSizeMismatchError(
                f"Object size changed too much (ratio: {best[4]:.1f}x, allowed: {size_min_ratio:.1f}x-{size_max_ratio:.1f}x)"
            )

        logger.warning(
            f"All {len(tried_points)} peak points failed, similarity below {min_score}"
        )
        raise PropagationNotFoundError("Could not find object in target image")

    def propagate_sequence(
        self,
        images: list[tuple[np.ndarray, str]],  # List of (image, image_id)
        start_bbox: list[float],
        start_mask: Optional[np.ndarray],
        start_idx: int = 0,
        count: int = -1,  # -1 = all remaining
        annotation_id: Optional[int] = None,
        stop_threshold: float = 0.3,
        callback: Optional[callable] = None,
    ) -> list[tuple[int, list[float], np.ndarray, float]]:
        """
        Propagate annotation through a sequence of images.

        Args:
            images: List of (image_array, image_id) tuples
            start_bbox: Starting bbox
            start_mask: Starting mask (optional)
            start_idx: Index of starting image
            count: Number of frames to propagate (-1 = all)
            annotation_id: Annotation ID for logging/caching
            stop_threshold: Stop if confidence drops below this
            callback: Optional callback(idx, bbox, mask, score) for progress

        Returns:
            List of (image_idx, bbox, mask, confidence) for successful propagations
        """
        results = []

        current_bbox = start_bbox
        current_mask = start_mask
        current_idx = start_idx

        # Determine end index
        if count < 0:
            end_idx = len(images)
        else:
            end_idx = min(start_idx + count + 1, len(images))

        # Get source image
        source_image, source_image_id = images[current_idx]

        for target_idx in range(current_idx + 1, end_idx):
            target_image, target_image_id = images[target_idx]

            # Propagate
            result = self.propagate_annotation(
                source_image=source_image,
                source_bbox=current_bbox,
                source_mask=current_mask,
                target_image=target_image,
                source_image_id=source_image_id,
                target_image_id=target_image_id,
                annotation_id=annotation_id,
            )

            if result is None:
                logger.info(f"Propagation stopped at frame {target_idx}: no result")
                break

            new_bbox, new_mask, confidence, _, _ = result  # Unpack all 5 elements

            if confidence < stop_threshold:
                logger.info(
                    f"Propagation stopped at frame {target_idx}: "
                    f"confidence {confidence:.3f} < {stop_threshold}"
                )
                break

            # Store result
            results.append((target_idx, new_bbox, new_mask, confidence))

            # Callback for progress
            if callback:
                callback(target_idx, new_bbox, new_mask, confidence)

            # Update for next iteration
            source_image = target_image
            source_image_id = target_image_id
            current_bbox = new_bbox
            current_mask = new_mask

        return results

    def find_all_instances(
        self,
        reference_image: np.ndarray,
        reference_bbox: list[float],
        reference_mask: Optional[np.ndarray],
        target_image: np.ndarray,
        reference_image_id: Optional[str] = None,
        target_image_id: Optional[str] = None,
        annotation_id: Optional[int] = None,
        min_similarity: float = 0.6,
        max_instances: int = 20,
        min_iou_between_instances: float = 0.0,
        max_iou_between_instances: float = 0.5,
        use_cached_masks: bool = True,
        size_tolerance: float = 0.5,
    ) -> list[PropagationResult]:
        """
        Find ALL instances of a class in the target image.

        Uses the reference annotation to define what the class looks like,
        then finds all similar objects in the target image.

        This is useful for:
        - Per-class auto-labeling (like legacy FastSAM everything mode)
        - Finding multiple instances of the same object type
        - Semi-automatic annotation of repeated objects

        Args:
            reference_image: Reference image with known instance
            reference_bbox: Bbox of reference instance [x1, y1, x2, y2]
            reference_mask: Mask of reference instance (optional but recommended)
            target_image: Image to search for instances
            reference_image_id: ID for caching reference features
            target_image_id: ID for caching target features
            annotation_id: Annotation ID for caching
            min_similarity: Minimum embedding similarity to consider a match
            max_instances: Maximum number of instances to return
            min_iou_between_instances: Min IoU between instances (for overlapping objects)
            max_iou_between_instances: Max IoU between instances (to avoid duplicates)
            use_cached_masks: Whether to use SAM everything cache
            size_tolerance: How much size can vary from reference (0.5 = 50%-200%)

        Returns:
            List of PropagationResult for each found instance
        """
        from core.masks import mask_iou

        img_h, img_w = target_image.shape[:2]
        results = []

        # Calculate reference area for size filtering
        # size_tolerance=0.5 means 50%-200% of reference size (multiplicative)
        ref_area = (reference_bbox[2] - reference_bbox[0]) * (
            reference_bbox[3] - reference_bbox[1]
        )
        min_area = ref_area / (1 + size_tolerance)
        max_area = ref_area * (1 + size_tolerance)

        # Get reference descriptor
        ref_descriptor = self.embed_service.get_object_descriptor(
            reference_image,
            reference_bbox,
            mask=reference_mask,
            image_id=reference_image_id,
            annotation_id=annotation_id,
        )

        # Get target features
        target_features = self.embed_service.get_image_features(
            target_image, image_id=target_image_id
        )

        # Compute similarity map
        num_patches = self.embed_service._img_size // self.embed_service._patch_size
        feats = target_features.squeeze(0)  # (num_patches^2, embed_dim)
        feats = F.normalize(feats, dim=-1)

        desc_tensor = torch.from_numpy(ref_descriptor).to(feats.device).float()
        similarity = torch.mv(feats, desc_tensor)
        similarity_map = similarity.reshape(num_patches, num_patches).cpu().numpy()

        # Set target image for SAM
        self.segment_service.set_image(target_image, target_image_id)

        # Strategy 1: Use cached masks if available
        if use_cached_masks:
            cached_masks = self.segment_service.get_cached_masks(target_image_id)
            if cached_masks is None:
                # Run segment everything to populate cache
                logger.info("Running segment_everything to find all masks...")
                cached_masks = self.segment_service.segment_everything(
                    target_image, target_image_id
                )

            if cached_masks:
                logger.info(
                    f"Evaluating {len(cached_masks)} cached masks for class matching..."
                )

                # Score each cached mask by its average similarity
                scored_masks = []
                for cached_mask in cached_masks:
                    # Check size constraint
                    if not (min_area <= cached_mask.area <= max_area):
                        continue

                    # Get mask's position in patch coords
                    bbox = cached_mask.bbox
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    px = int(cx / img_w * num_patches)
                    py = int(cy / img_h * num_patches)
                    px = min(max(px, 0), num_patches - 1)
                    py = min(max(py, 0), num_patches - 1)

                    # Get similarity at center
                    center_sim = float(similarity_map[py, px])

                    # Also compute descriptor similarity
                    mask_desc = self.embed_service.get_object_descriptor(
                        target_image,
                        bbox,
                        mask=cached_mask.mask,
                        image_id=target_image_id,
                    )
                    desc_sim = self.embed_service.compute_similarity(
                        ref_descriptor, mask_desc
                    )

                    # Combined score
                    score = 0.5 * center_sim + 0.5 * desc_sim

                    if score >= min_similarity:
                        scored_masks.append((cached_mask, score, desc_sim))

                # Sort by score (highest first)
                scored_masks.sort(key=lambda x: x[1], reverse=True)

                # Apply NMS to avoid duplicates
                for cached_mask, score, desc_sim in scored_masks:
                    if len(results) >= max_instances:
                        break

                    # Check IoU with existing results
                    is_duplicate = False
                    for existing in results:
                        iou = mask_iou(cached_mask.mask, existing.mask)
                        if iou > max_iou_between_instances:
                            is_duplicate = True
                            break
                        if iou > 0 and iou < min_iou_between_instances:
                            # Too much overlap but not enough - probably not the same instance
                            continue

                    if not is_duplicate:
                        area_ratio = (
                            cached_mask.area / ref_area if ref_area > 0 else 1.0
                        )
                        results.append(
                            PropagationResult(
                                bbox=cached_mask.bbox,
                                mask=cached_mask.mask,
                                confidence=score,
                                fallback_used=False,
                                area_ratio=area_ratio,
                                method="class_match",
                                iou_score=None,
                            )
                        )

                logger.info(f"Found {len(results)} instances via cached mask matching")

        # Strategy 2: Find peaks in similarity map and prompt SAM
        if len(results) < max_instances:
            logger.info("Finding additional instances via similarity peaks...")

            # Find peaks above threshold
            flat_sim = similarity_map.flatten()
            peak_indices = np.argsort(flat_sim)[::-1]

            tried_points = []
            for peak_idx in peak_indices:
                if len(results) >= max_instances:
                    break

                peak_sim = flat_sim[peak_idx]
                if peak_sim < min_similarity:
                    break

                py, px = divmod(peak_idx, num_patches)
                point_x = (px + 0.5) / num_patches * img_w
                point_y = (py + 0.5) / num_patches * img_h

                # Skip if too close to already tried points
                too_close = False
                for tx, ty in tried_points:
                    if (
                        abs(point_x - tx) < img_w * 0.05
                        and abs(point_y - ty) < img_h * 0.05
                    ):
                        too_close = True
                        break
                if too_close:
                    continue

                tried_points.append((point_x, point_y))

                try:
                    mask, sam_score, bbox = self.segment_service.segment_with_point(
                        point_x, point_y
                    )

                    # Check size constraint
                    mask_area = mask.sum()
                    if not (min_area <= mask_area <= max_area):
                        continue

                    # Verify with descriptor
                    mask_desc = self.embed_service.get_object_descriptor(
                        target_image, bbox, mask=mask, image_id=target_image_id
                    )
                    desc_sim = self.embed_service.compute_similarity(
                        ref_descriptor, mask_desc
                    )

                    if desc_sim < min_similarity:
                        continue

                    # Check IoU with existing results
                    is_duplicate = False
                    for existing in results:
                        iou = mask_iou(mask, existing.mask)
                        if iou > max_iou_between_instances:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        confidence = 0.5 * desc_sim + 0.5 * sam_score
                        area_ratio = mask_area / ref_area if ref_area > 0 else 1.0

                        results.append(
                            PropagationResult(
                                bbox=bbox,
                                mask=mask,
                                confidence=confidence,
                                fallback_used=False,
                                area_ratio=area_ratio,
                                method="peak_match",
                                iou_score=None,
                            )
                        )

                except Exception as e:
                    logger.debug(f"Peak point failed: {e}")
                    continue

        # Sort results by confidence
        results.sort(key=lambda r: r.confidence, reverse=True)

        logger.info(f"find_all_instances complete: found {len(results)} instances")
        return results

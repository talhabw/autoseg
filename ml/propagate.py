"""
Propagation module - Track objects across frames using embeddings + SAM.
"""

import numpy as np
import torch
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BboxCandidate:
    """A candidate bounding box with score."""

    bbox_xyxy: list[float]
    score: float
    source: str = "propagated"


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
    import torch
    import torch.nn.functional as F

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

    def propagate_annotation(
        self,
        source_image: np.ndarray,
        source_bbox: list[float],
        source_mask: Optional[np.ndarray],
        target_image: np.ndarray,
        source_image_id: Optional[str] = None,
        target_image_id: Optional[str] = None,
        annotation_id: Optional[int] = None,
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
                # Use point to prompt SAM
                mask, sam_score, refined_bbox = self.segment_service.segment_with_point(
                    point_x,
                    point_y,
                    bbox_hint=None,  # Let SAM figure out the object
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
        logger.info(f"=== Propagation candidates summary ===")
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
        logger.info(f"======================================")

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

            new_bbox, new_mask, confidence = result

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

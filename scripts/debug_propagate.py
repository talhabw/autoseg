#!/usr/bin/env python3
"""
Debug script to trace exactly what happens in the PropagateService.

This replicates the propagate_annotation flow step-by-step with visualization.

Usage:
    python scripts/debug_propagate.py <image1> <image2> <x1> <y1> <x2> <y2>

Example:
    python scripts/debug_propagate.py ./tbw/images/match-raw-1.png ./tbw/images/match-raw-2.png 162 275 170 383
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, ".")

from ml.embed import EmbedService
from ml.segment import SegmentService


def debug_propagate(
    img1_path: str,
    img2_path: str,
    bbox_xyxy: list[float],
    output_path: str = "debug_propagate.png",
    top_k: int = 4,
    min_score: float = 0.5,
):
    """
    Debug propagation step by step, matching PropagateService.propagate_annotation.
    """
    print("=" * 60)
    print("DEBUG PROPAGATION")
    print("=" * 60)

    # Load images
    print("\n1. Loading images...")
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    img1_np = np.array(img1)
    img2_np = np.array(img2)

    img_h, img_w = img2_np.shape[:2]
    print(f"   Source image: {img1_np.shape}")
    print(f"   Target image: {img2_np.shape}")
    print(f"   Source bbox: {bbox_xyxy}")

    # Calculate source area
    source_area = (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1])
    print(f"   Source area: {source_area:.1f} pixels")

    # Load models
    print("\n2. Loading DINOv3 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_service = EmbedService(device=device)
    embed_service.load_model()

    print("\n3. Loading SAM model...")
    segment_service = SegmentService(device=device)
    segment_service.load_model()

    # Get source descriptor (exactly as PropagateService does)
    print("\n4. Computing source descriptor...")
    source_descriptor = embed_service.get_object_descriptor(
        img1_np,
        bbox_xyxy,
        mask=None,  # No mask in this test
        image_id="source",
        annotation_id=1,
    )
    print(f"   Source descriptor shape: {source_descriptor.shape}")
    print(f"   Source descriptor norm: {np.linalg.norm(source_descriptor):.4f}")

    # Get target features
    print("\n5. Computing target features...")
    target_features = embed_service.get_image_features(
        img2_np,
        image_id="target",
    )

    # Compute similarity map (exactly as PropagateService does)
    print("\n6. Computing similarity map...")
    num_patches = embed_service._img_size // embed_service._patch_size
    print(f"   Num patches: {num_patches}x{num_patches}")

    feats = target_features.squeeze(0)  # (num_patches, embed_dim)
    feats = F.normalize(feats, dim=-1)

    desc_tensor = torch.from_numpy(source_descriptor).to(feats.device).float()
    similarity = torch.mv(feats, desc_tensor)  # (num_patches,)
    similarity_map = similarity.reshape(num_patches, num_patches).float().cpu().numpy()

    print(f"   Similarity map shape: {similarity_map.shape}")
    print(f"   Similarity min: {similarity_map.min():.4f}")
    print(f"   Similarity max: {similarity_map.max():.4f}")
    print(f"   Similarity mean: {similarity_map.mean():.4f}")

    # Find top-k peaks (exactly as PropagateService does)
    print(f"\n7. Finding top {top_k} peaks...")
    flat_sim = similarity_map.flatten()
    peak_indices = np.argsort(flat_sim)[::-1]  # Descending order

    # Set target image for SAM
    segment_service.set_image(img2_np, "target")

    tried_points = []
    results = []

    for i in range(min(top_k * 3, len(peak_indices))):
        peak_idx = peak_indices[i]
        py, px = divmod(peak_idx, num_patches)

        # Convert patch coords to pixel coords (center of patch)
        point_x = (px + 0.5) / num_patches * img_w
        point_y = (py + 0.5) / num_patches * img_h

        # Skip if too close to already tried points
        too_close = False
        for tx, ty in tried_points:
            if abs(point_x - tx) < img_w * 0.05 and abs(point_y - ty) < img_h * 0.05:
                too_close = True
                break
        if too_close:
            continue

        tried_points.append((point_x, point_y))

        if len(tried_points) > top_k:
            break

        peak_sim = flat_sim[peak_idx]
        print(
            f"\n   Peak #{len(tried_points)}: patch=({px}, {py}), pixel=({point_x:.1f}, {point_y:.1f}), peak_sim={peak_sim:.4f}"
        )

        try:
            # Use point to prompt SAM (exactly as PropagateService does)
            mask, sam_score, refined_bbox = segment_service.segment_with_point(
                point_x,
                point_y,
                bbox_hint=None,  # Let SAM figure out the object
            )

            print(f"      SAM result: score={sam_score:.4f}, bbox={refined_bbox}")

            # Verify with embedding - compute descriptor of result
            # NOTE: Don't pass annotation_id to avoid caching issues between iterations
            target_descriptor = embed_service.get_object_descriptor(
                img2_np,
                refined_bbox,
                mask=mask,
                image_id="target_result",  # Use different image_id to avoid cache issues
                use_cache=False,  # Force recompute for each SAM result
            )

            # Debug: check descriptor values
            print(
                f"      Target descriptor: norm={np.linalg.norm(target_descriptor):.4f}, "
                f"min={target_descriptor.min():.4f}, max={target_descriptor.max():.4f}, "
                f"mean={target_descriptor.mean():.6f}"
            )
            print(
                f"      Source descriptor: min={source_descriptor.min():.4f}, max={source_descriptor.max():.4f}"
            )

            # Manual dot product check
            raw_dot = np.dot(source_descriptor, target_descriptor)
            print(f"      Raw dot product: {raw_dot:.6f}")

            embed_similarity = embed_service.compute_similarity(
                source_descriptor, target_descriptor
            )

            # Combined confidence
            confidence = 0.6 * embed_similarity + 0.4 * sam_score

            # Check size
            result_area = (refined_bbox[2] - refined_bbox[0]) * (
                refined_bbox[3] - refined_bbox[1]
            )
            area_ratio = result_area / source_area if source_area > 0 else 1.0

            print(f"      Embed similarity: {embed_similarity:.4f}")
            print(f"      Combined confidence: {confidence:.4f}")
            print(f"      Area ratio: {area_ratio:.2f}x")
            print(
                f"      PASS? embed_sim >= {min_score}: {embed_similarity >= min_score}"
            )

            results.append(
                {
                    "peak_num": len(tried_points),
                    "point": (point_x, point_y),
                    "peak_sim": peak_sim,
                    "mask": mask,
                    "bbox": refined_bbox,
                    "sam_score": sam_score,
                    "embed_sim": embed_similarity,
                    "confidence": confidence,
                    "area_ratio": area_ratio,
                    "passed": embed_similarity >= min_score,
                }
            )

        except Exception as e:
            print(f"      SAM failed: {e}")
            results.append(
                {
                    "peak_num": len(tried_points),
                    "point": (point_x, point_y),
                    "peak_sim": peak_sim,
                    "error": str(e),
                }
            )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = [r for r in results if r.get("passed", False)]
    failed = [r for r in results if not r.get("passed", False) and "embed_sim" in r]
    errors = [r for r in results if "error" in r]

    print(f"Total tried: {len(results)}")
    print(f"Passed (embed_sim >= {min_score}): {len(passed)}")
    print(f"Failed (embed_sim < {min_score}): {len(failed)}")
    print(f"SAM errors: {len(errors)}")

    if passed:
        best = max(passed, key=lambda x: x["embed_sim"])
        print(
            f"\nBest passing result: embed_sim={best['embed_sim']:.4f}, bbox={best['bbox']}"
        )
    elif failed:
        best = max(failed, key=lambda x: x["embed_sim"])
        print(
            f"\nBest failing result: embed_sim={best['embed_sim']:.4f}, bbox={best['bbox']}"
        )
        print(
            "   This is WHY you see 'similarity below 0.5' - the SAM mask doesn't match the source!"
        )

    # Visualize
    print(f"\n8. Creating visualization: {output_path}")
    create_visualization(
        img1_np,
        img2_np,
        bbox_xyxy,
        similarity_map,
        results,
        output_path,
    )

    return results


def create_visualization(
    img1: np.ndarray,
    img2: np.ndarray,
    source_bbox: list[float],
    similarity_map: np.ndarray,
    results: list[dict],
    output_path: str,
):
    """Create multi-panel visualization."""
    n_results = len([r for r in results if "mask" in r])
    n_cols = 3 + min(n_results, 4)  # source, sim map, target, up to 4 SAM results

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))

    # Row 1: Overview
    # Source image with bbox
    axes[0, 0].imshow(img1)
    x1, y1, x2, y2 = source_bbox
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor="lime", facecolor="none"
    )
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title(f"Source + BBox\n{source_bbox}", fontsize=10)
    axes[0, 0].axis("off")

    # Similarity map
    im = axes[0, 1].imshow(similarity_map, cmap="hot", vmin=0, vmax=1)
    axes[0, 1].set_title(
        f"Similarity Map\nmin={similarity_map.min():.3f}, max={similarity_map.max():.3f}",
        fontsize=10,
    )
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    # Target image with peak points
    axes[0, 2].imshow(img2)
    colors = ["lime", "cyan", "yellow", "magenta", "orange", "red", "blue", "white"]
    for i, r in enumerate(results):
        px, py = r["point"]
        color = colors[i % len(colors)]
        axes[0, 2].plot(
            px, py, "o", color=color, markersize=10, markeredgecolor="black"
        )
        axes[0, 2].text(
            px + 10,
            py,
            f"#{r['peak_num']}\n{r['peak_sim']:.3f}",
            color=color,
            fontsize=8,
            fontweight="bold",
        )
    axes[0, 2].set_title("Target + Peak Points\n(peak similarity shown)", fontsize=10)
    axes[0, 2].axis("off")

    # SAM results
    for i, r in enumerate([r for r in results if "mask" in r][:4]):
        col = 3 + i
        axes[0, col].imshow(img2)

        # Overlay mask
        mask_overlay = np.zeros((*r["mask"].shape, 4))
        if r["passed"]:
            mask_overlay[r["mask"] > 0] = [0, 1, 0, 0.4]  # Green for passed
        else:
            mask_overlay[r["mask"] > 0] = [1, 0, 0, 0.4]  # Red for failed
        axes[0, col].imshow(mask_overlay)

        # Draw bbox
        x1, y1, x2, y2 = r["bbox"]
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="lime" if r["passed"] else "red",
            facecolor="none",
        )
        axes[0, col].add_patch(rect)

        status = "PASS ✓" if r["passed"] else "FAIL ✗"
        axes[0, col].set_title(
            f"Peak #{r['peak_num']} SAM Result\n"
            f"embed_sim={r['embed_sim']:.3f} ({status})\n"
            f"area_ratio={r['area_ratio']:.2f}x",
            fontsize=10,
        )
        axes[0, col].axis("off")

    # Row 2: Zoomed views
    # Zoom on source bbox
    pad = 50
    x1, y1, x2, y2 = [int(v) for v in source_bbox]
    x1z, y1z = max(0, x1 - pad), max(0, y1 - pad)
    x2z, y2z = min(img1.shape[1], x2 + pad), min(img1.shape[0], y2 + pad)
    axes[1, 0].imshow(img1[y1z:y2z, x1z:x2z])
    rect = patches.Rectangle(
        (x1 - x1z, y1 - y1z),
        x2 - x1,
        y2 - y1,
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    axes[1, 0].add_patch(rect)
    axes[1, 0].set_title("Source BBox (zoomed)", fontsize=10)
    axes[1, 0].axis("off")

    # Hide unused panels in row 2
    for col in range(1, n_cols):
        axes[1, col].axis("off")

    # Add zoomed SAM results in row 2
    for i, r in enumerate([r for r in results if "mask" in r][:4]):
        col = 1 + i
        if col < n_cols:
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            x1z, y1z = max(0, x1 - pad), max(0, y1 - pad)
            x2z, y2z = min(img2.shape[1], x2 + pad), min(img2.shape[0], y2 + pad)

            crop = img2[y1z:y2z, x1z:x2z].copy()
            mask_crop = r["mask"][y1z:y2z, x1z:x2z]

            axes[1, col].imshow(crop)

            mask_overlay = np.zeros((*mask_crop.shape, 4))
            if r["passed"]:
                mask_overlay[mask_crop > 0] = [0, 1, 0, 0.5]
            else:
                mask_overlay[mask_crop > 0] = [1, 0, 0, 0.5]
            axes[1, col].imshow(mask_overlay)

            axes[1, col].set_title(f"Peak #{r['peak_num']} Zoomed", fontsize=10)
            axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to: {output_path}")


def main():
    if len(sys.argv) < 7:
        print(__doc__)
        print("\nError: Not enough arguments")
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    x1, y1, x2, y2 = map(float, sys.argv[3:7])

    output_path = sys.argv[7] if len(sys.argv) > 7 else "debug_propagate.png"

    debug_propagate(
        img1_path,
        img2_path,
        [x1, y1, x2, y2],
        output_path,
    )


if __name__ == "__main__":
    main()

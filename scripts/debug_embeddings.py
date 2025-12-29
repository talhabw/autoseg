#!/usr/bin/env python3
"""
Debug script to visualize DINOv3 embedding matching.

Usage:
    python scripts/debug_embeddings.py <image1> <image2> <x1> <y1> <x2> <y2>
    
Example:
    python scripts/debug_embeddings.py frame001.jpg frame002.jpg 100 100 200 200
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, '.')

from ml.embed import EmbedService


def visualize_embedding_match(
    img1_path: str,
    img2_path: str,
    bbox_xyxy: list[float],
    output_path: str = "debug_embed.png",
):
    """
    Visualize embedding matching between two images.
    
    Shows:
    - Image 1 with bbox
    - Image 2 with similarity heatmap
    - Top-k matching locations
    """
    print("Loading images...")
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    
    print("Loading DINOv3 model...")
    embed_service = EmbedService(device="cuda" if torch.cuda.is_available() else "cpu")
    embed_service.load_model()
    
    print("Extracting features...")
    # Get features for both images
    feat1 = embed_service.get_image_features(img1_np, image_id="img1")
    feat2 = embed_service.get_image_features(img2_np, image_id="img2")
    
    # Get object descriptor from bbox in image 1
    print(f"Computing descriptor for bbox {bbox_xyxy}...")
    descriptor = embed_service.get_object_descriptor(
        img1_np, 
        bbox_xyxy, 
        mask=None,
        image_id="img1"
    )
    
    # Compute similarity map
    print("Computing similarity map...")
    similarity_map = compute_similarity_map(
        descriptor, 
        feat2, 
        embed_service._img_size,
        embed_service._patch_size,
    )
    
    # Find top matches
    print("Finding top matches...")
    top_matches = find_top_matches(
        similarity_map,
        bbox_xyxy,
        img1.size[0], img1.size[1],
        img2.size[0], img2.size[1],
        embed_service._img_size,
        embed_service._patch_size,
        top_k=5,
    )
    
    # Visualize
    print(f"Saving visualization to {output_path}...")
    visualize_results(
        img1_np, img2_np,
        bbox_xyxy,
        similarity_map,
        top_matches,
        output_path,
    )
    
    # Print match info
    print("\nTop matches:")
    for i, (score, bbox) in enumerate(top_matches):
        print(f"  {i+1}. Score: {score:.4f}, BBox: {bbox}")
    
    print(f"\nVisualization saved to: {output_path}")
    return similarity_map, top_matches


def compute_similarity_map(
    descriptor: np.ndarray,
    features: torch.Tensor,
    img_size: int,
    patch_size: int,
) -> np.ndarray:
    """Compute per-patch similarity to the descriptor."""
    num_patches = img_size // patch_size
    
    # Reshape features to (num_patches, embed_dim)
    feats = features.squeeze(0)  # (num_patches, embed_dim)
    
    # Normalize features
    feats = F.normalize(feats, dim=-1)
    
    # Convert descriptor to tensor
    desc_tensor = torch.from_numpy(descriptor).to(feats.device).float()
    
    # Compute cosine similarity
    similarity = torch.mv(feats, desc_tensor)  # (num_patches,)
    
    # Reshape to spatial
    similarity_map = similarity.reshape(num_patches, num_patches).cpu().numpy()
    
    return similarity_map


def find_top_matches(
    similarity_map: np.ndarray,
    source_bbox: list[float],
    src_w: int, src_h: int,
    tgt_w: int, tgt_h: int,
    img_size: int,
    patch_size: int,
    top_k: int = 5,
) -> list[tuple[float, list[float]]]:
    """Find top-k matching locations."""
    num_patches = img_size // patch_size
    
    # Get source bbox size in patches
    x1, y1, x2, y2 = source_bbox
    bw_patches = int(np.ceil((x2 - x1) / src_w * num_patches))
    bh_patches = int(np.ceil((y2 - y1) / src_h * num_patches))
    
    # Ensure at least 1 patch
    bw_patches = max(1, bw_patches)
    bh_patches = max(1, bh_patches)
    
    # Compute windowed scores
    matches = []
    for py in range(num_patches - bh_patches + 1):
        for px in range(num_patches - bw_patches + 1):
            # Average similarity in this window
            window = similarity_map[py:py+bh_patches, px:px+bw_patches]
            score = window.mean()
            
            # Convert to pixel coordinates in target image
            x1_tgt = px / num_patches * tgt_w
            y1_tgt = py / num_patches * tgt_h
            x2_tgt = (px + bw_patches) / num_patches * tgt_w
            y2_tgt = (py + bh_patches) / num_patches * tgt_h
            
            matches.append((score, [x1_tgt, y1_tgt, x2_tgt, y2_tgt]))
    
    # Sort by score
    matches.sort(key=lambda x: x[0], reverse=True)
    
    # Non-max suppression to get diverse results
    selected = []
    for score, bbox in matches:
        if len(selected) >= top_k:
            break
        
        # Check overlap with already selected
        overlaps = False
        for _, sel_bbox in selected:
            if bbox_iou(bbox, sel_bbox) > 0.3:
                overlaps = True
                break
        
        if not overlaps:
            selected.append((score, bbox))
    
    return selected


def bbox_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def visualize_results(
    img1: np.ndarray,
    img2: np.ndarray,
    source_bbox: list[float],
    similarity_map: np.ndarray,
    top_matches: list[tuple[float, list[float]]],
    output_path: str,
):
    """Create visualization figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Image 1 with source bbox
    axes[0].imshow(img1)
    x1, y1, x2, y2 = source_bbox
    rect = patches.Rectangle(
        (x1, y1), x2-x1, y2-y1,
        linewidth=3, edgecolor='lime', facecolor='none'
    )
    axes[0].add_patch(rect)
    axes[0].set_title("Source Image + BBox", fontsize=14)
    axes[0].axis('off')
    
    # Similarity heatmap
    im = axes[1].imshow(similarity_map, cmap='hot', vmin=-1, vmax=1)
    axes[1].set_title(f"Similarity Map\n(min={similarity_map.min():.3f}, max={similarity_map.max():.3f})", fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Image 2 with top matches
    axes[2].imshow(img2)
    colors = ['lime', 'cyan', 'yellow', 'magenta', 'orange']
    for i, (score, bbox) in enumerate(top_matches):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=colors[i % len(colors)], facecolor='none'
        )
        axes[2].add_patch(rect)
        axes[2].text(x1, y1-5, f"#{i+1}: {score:.3f}", 
                    color=colors[i % len(colors)], fontsize=10, fontweight='bold')
    axes[2].set_title("Target Image + Top Matches", fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    if len(sys.argv) < 7:
        print(__doc__)
        print("\nError: Not enough arguments")
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    x1, y1, x2, y2 = map(float, sys.argv[3:7])
    
    output_path = sys.argv[7] if len(sys.argv) > 7 else "debug_embed.png"
    
    visualize_embedding_match(
        img1_path, img2_path,
        [x1, y1, x2, y2],
        output_path
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Export a project to YOLO-seg format.

Usage:
    python scripts/export_yolo.py <project_dir> <output_dir> [--train-split 0.8]
"""

import sys
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.export_yolo import export_yolo_seg, verify_yolo_seg_export


def main():
    parser = argparse.ArgumentParser(description="Export project to YOLO-seg format")
    parser.add_argument("project_dir", help="Path to the project directory")
    parser.add_argument("output_dir", help="Output directory for export")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--include-unapproved", action="store_true", help="Include unapproved annotations")
    
    args = parser.parse_args()
    
    val_split = 1.0 - args.train_split
    
    print(f"Exporting project: {args.project_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split: train={args.train_split:.0%}, val={val_split:.0%}")
    print("-" * 50)
    
    try:
        report = export_yolo_seg(
            project_dir=args.project_dir,
            out_dir=args.output_dir,
            split={"train": args.train_split, "val": val_split},
            seed=args.seed,
            approved_only=not args.include_unapproved
        )
    except Exception as e:
        print(f"✗ Export failed: {e}")
        sys.exit(1)
    
    print(f"\nExport Report:")
    print(f"  Total images: {report.total_images}")
    print(f"  Train images: {report.train_images}")
    print(f"  Val images: {report.val_images}")
    print(f"  Skipped images: {report.skipped_images}")
    print(f"  Total annotations: {report.total_annotations}")
    print(f"  With polygon: {report.annotations_with_polygon}")
    print(f"  Without polygon: {report.annotations_without_polygon}")
    print(f"  Labels: {', '.join(report.labels)}")
    
    if report.warnings:
        print(f"\nWarnings:")
        for warning in report.warnings:
            print(f"  ⚠ {warning}")
    
    # Verify export
    print("\nVerifying export...")
    is_valid, errors = verify_yolo_seg_export(args.output_dir)
    
    if is_valid:
        print("✓ Export verified successfully!")
    else:
        print(f"✗ Verification found {len(errors)} error(s)")
        for error in errors[:10]:
            print(f"  - {error}")
        sys.exit(1)
    
    print(f"\n✓ Export complete: {args.output_dir}")


if __name__ == "__main__":
    main()

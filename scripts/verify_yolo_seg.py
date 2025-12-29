#!/usr/bin/env python
"""
Verify a YOLO-seg export is valid.

Usage:
    python scripts/verify_yolo_seg.py <export_dir>
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.export_yolo import verify_yolo_seg_export


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_yolo_seg.py <export_dir>")
        sys.exit(1)
    
    export_dir = sys.argv[1]
    
    print(f"Verifying YOLO-seg export: {export_dir}")
    print("-" * 50)
    
    is_valid, errors = verify_yolo_seg_export(export_dir)
    
    if is_valid:
        print("✓ Export is valid!")
        sys.exit(0)
    else:
        print(f"✗ Found {len(errors)} error(s):\n")
        for error in errors[:20]:  # Limit output
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        sys.exit(1)


if __name__ == "__main__":
    main()

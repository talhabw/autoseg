"""Tests for the offline review bundle generator helpers."""

from pathlib import Path

from PIL import Image

from offline_review.generate_bundle import parse_yolo_annotation_line, scan_dataset


def test_parse_yolo_bbox_annotation():
    annotation = parse_yolo_annotation_line("3 0.5 0.25 0.2 0.5", (100, 200))

    assert annotation is not None
    assert annotation.kind == "bbox"
    assert annotation.class_id == 3
    assert annotation.label == "3"
    assert annotation.bbox == (40.0, 0.0, 60.0, 100.0)


def test_parse_yolo_polygon_annotation_with_class_name():
    annotation = parse_yolo_annotation_line(
        "1 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8",
        (200, 100),
        {1: "crate"},
    )

    assert annotation is not None
    assert annotation.kind == "polygon"
    assert annotation.label == "crate"
    assert annotation.points[0] == (20.0, 20.0)
    assert annotation.bbox == (20.0, 20.0, 100.0, 80.0)


def test_scan_dataset_recursively_only_returns_labeled_images(tmp_path: Path):
    root = tmp_path / "dataset"
    nested = root / "nested"
    nested.mkdir(parents=True)

    labeled_image = nested / "frame_001.jpg"
    Image.new("RGB", (16, 16), color="white").save(labeled_image)
    labeled_image.with_suffix(".txt").write_text("0 0.5 0.5 0.5 0.5\n")

    unlabeled_image = root / "frame_002.png"
    Image.new("RGB", (16, 16), color="white").save(unlabeled_image)

    entries = scan_dataset(root)

    assert len(entries) == 1
    assert entries[0].relative_image_path == "nested/frame_001.jpg"
    assert entries[0].relative_label_path == "nested/frame_001.txt"

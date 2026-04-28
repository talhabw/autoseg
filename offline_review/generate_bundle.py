"""Generate a static offline review app for YOLO-style datasets.

The input directory is scanned recursively for images with same-stem `.txt`
label files. The script renders compressed preview images with bounding boxes
or segmentation polygons baked in, then writes a self-contained HTML/CSS/JS
bundle that can be opened directly in a browser.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_MAX_DIM = 1600
DEFAULT_QUALITY = 82
COLOR_PALETTE: list[tuple[int, int, int]] = [
    (239, 68, 68),
    (255, 103, 0),
    (255, 235, 59),
    (255, 255, 255),
    (255, 105, 180),
    (0, 0, 0),
]


@dataclass(frozen=True)
class DatasetEntry:
    image_path: Path
    label_path: Path
    relative_image_path: str
    relative_label_path: str


@dataclass(frozen=True)
class ParsedAnnotation:
    kind: str
    class_id: int
    label: str
    bbox: tuple[float, float, float, float]
    points: list[tuple[float, float]]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _find_class_names(input_dir: Path) -> dict[int, str]:
    for candidate in ("data.yaml", "dataset.yaml"):
        yaml_path = input_dir / candidate
        if yaml_path.is_file():
            names = _parse_names_from_yaml(yaml_path)
            if names:
                return names

    for candidate in ("classes.txt", "labels.txt"):
        txt_path = input_dir / candidate
        if txt_path.is_file():
            lines = [line.strip() for line in txt_path.read_text().splitlines()]
            return {idx: line for idx, line in enumerate(lines) if line}

    return {}


def _parse_names_from_yaml(yaml_path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    lines = yaml_path.read_text().splitlines()
    in_names_block = False

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not in_names_block:
            if stripped.startswith("names:"):
                inline_value = stripped.partition(":")[2].strip()
                if inline_value:
                    try:
                        parsed = ast.literal_eval(inline_value)
                    except (SyntaxError, ValueError):
                        return {}

                    if isinstance(parsed, dict):
                        return {int(key): str(value) for key, value in parsed.items()}
                    if isinstance(parsed, list):
                        return {idx: str(value) for idx, value in enumerate(parsed)}
                    return {}
                in_names_block = True
            continue

        if raw_line[:1] not in {" ", "\t", "-"} and ":" in stripped:
            break

        if stripped.startswith("-"):
            value = stripped[1:].strip()
            names[len(names)] = value.strip("\"'")
            continue

        key, sep, value = stripped.partition(":")
        if not sep:
            break
        try:
            names[int(key.strip())] = value.strip().strip("\"'")
        except ValueError:
            continue

    return names


def parse_yolo_annotation_line(
    line: str,
    image_size: tuple[int, int],
    class_names: dict[int, str] | None = None,
) -> ParsedAnnotation | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    parts = stripped.split()
    if len(parts) < 5:
        raise ValueError(f"expected at least 5 fields, got {len(parts)}")

    try:
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"failed to parse numeric values: {exc}") from exc

    image_width, image_height = image_size
    label = (class_names or {}).get(class_id, str(class_id))

    if len(coords) == 4:
        cx, cy, box_w, box_h = coords
        x1 = _clamp((cx - box_w / 2) * image_width, 0.0, float(image_width))
        y1 = _clamp((cy - box_h / 2) * image_height, 0.0, float(image_height))
        x2 = _clamp((cx + box_w / 2) * image_width, 0.0, float(image_width))
        y2 = _clamp((cy + box_h / 2) * image_height, 0.0, float(image_height))
        return ParsedAnnotation(
            kind="bbox",
            class_id=class_id,
            label=label,
            bbox=(x1, y1, x2, y2),
            points=[],
        )

    if len(coords) >= 6 and len(coords) % 2 == 0:
        points: list[tuple[float, float]] = []
        xs: list[float] = []
        ys: list[float] = []
        for idx in range(0, len(coords), 2):
            x = _clamp(coords[idx] * image_width, 0.0, float(image_width))
            y = _clamp(coords[idx + 1] * image_height, 0.0, float(image_height))
            points.append((x, y))
            xs.append(x)
            ys.append(y)

        return ParsedAnnotation(
            kind="polygon",
            class_id=class_id,
            label=label,
            bbox=(min(xs), min(ys), max(xs), max(ys)),
            points=points,
        )

    raise ValueError(
        "unsupported label line: expected YOLO bbox (4 coords) or polygon (even coord count >= 6)"
    )


def _parse_label_file(
    label_path: Path,
    image_size: tuple[int, int],
    class_names: dict[int, str],
) -> tuple[list[ParsedAnnotation], list[str]]:
    annotations: list[ParsedAnnotation] = []
    warnings: list[str] = []

    for line_no, raw_line in enumerate(label_path.read_text().splitlines(), start=1):
        try:
            parsed = parse_yolo_annotation_line(raw_line, image_size, class_names)
        except ValueError as exc:
            warnings.append(f"{label_path}:{line_no}: {exc}")
            continue

        if parsed is not None:
            annotations.append(parsed)

    return annotations, warnings


def scan_dataset(input_dir: Path, output_dir: Path | None = None) -> list[DatasetEntry]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve() if output_dir is not None else None
    entries: list[DatasetEntry] = []
    skip_generated_subtree = output_dir is not None and output_dir.is_relative_to(
        input_dir
    )

    for image_path in sorted(input_dir.rglob("*")):
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in IMAGE_EXTENSIONS
        ):
            continue
        if skip_generated_subtree and image_path.is_relative_to(output_dir):
            continue

        label_path = image_path.with_suffix(".txt")
        if not label_path.is_file():
            continue

        entries.append(
            DatasetEntry(
                image_path=image_path,
                label_path=label_path,
                relative_image_path=image_path.relative_to(input_dir).as_posix(),
                relative_label_path=label_path.relative_to(input_dir).as_posix(),
            )
        )

    return entries


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    return COLOR_PALETTE[class_id % len(COLOR_PALETTE)]


def _scale_bbox(
    bbox: tuple[float, float, float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return x1 * scale, y1 * scale, x2 * scale, y2 * scale


def _scale_points(
    points: list[tuple[float, float]], scale: float
) -> list[tuple[float, float]]:
    return [(x * scale, y * scale) for x, y in points]


def _draw_label_tag(
    draw: ImageDraw.ImageDraw,
    text: str,
    anchor: tuple[float, float],
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    canvas_size: tuple[int, int],
) -> None:
    canvas_width, canvas_height = canvas_size
    pad_x = 6
    pad_y = 4
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x = max(0.0, min(anchor[0], canvas_width - text_width - 2 * pad_x))
    if anchor[1] > text_height + 2 * pad_y + 4:
        y = anchor[1] - text_height - 2 * pad_y - 2
    else:
        y = anchor[1] + 2
    y = max(0.0, min(y, canvas_height - text_height - 2 * pad_y))

    draw.rounded_rectangle(
        [x, y, x + text_width + 2 * pad_x, y + text_height + 2 * pad_y],
        radius=6,
        fill=(*color, 230),
    )
    draw.text((x + pad_x, y + pad_y), text, fill=(255, 255, 255, 255), font=font)


def _render_preview_image(
    image_path: Path,
    label_path: Path,
    preview_path: Path,
    class_names: dict[int, str],
    max_dim: int,
    quality: int,
) -> tuple[dict[str, Any], list[str]]:
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    original_width, original_height = image.size
    annotations, warnings = _parse_label_file(label_path, image.size, class_names)

    scale = 1.0
    if max_dim > 0 and max(original_width, original_height) > max_dim:
        scale = max_dim / float(max(original_width, original_height))
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    canvas = image.convert("RGBA")
    fill_overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_overlay, "RGBA")

    for annotation in annotations:
        color = _color_for_class(annotation.class_id)
        if annotation.kind == "polygon":
            scaled_points = _scale_points(annotation.points, scale)
            if len(scaled_points) >= 3:
                fill_draw.polygon(scaled_points, fill=(*color, 84))

    canvas = Image.alpha_composite(canvas, fill_overlay)
    outline_width = max(2, int(round(max(canvas.size) / 480)))
    font = _load_font(max(14, int(round(max(canvas.size) / 48))))
    draw = ImageDraw.Draw(canvas, "RGBA")

    for annotation in annotations:
        color = _color_for_class(annotation.class_id)
        if annotation.kind == "polygon":
            scaled_points = _scale_points(annotation.points, scale)
            if len(scaled_points) >= 2:
                draw.line(
                    scaled_points + [scaled_points[0]],
                    fill=(*color, 255),
                    width=outline_width,
                )
            anchor_x = min(point[0] for point in scaled_points)
            anchor_y = min(point[1] for point in scaled_points)
        else:
            scaled_bbox = _scale_bbox(annotation.bbox, scale)
            draw.rectangle(scaled_bbox, outline=(*color, 255), width=outline_width)
            anchor_x = scaled_bbox[0]
            anchor_y = scaled_bbox[1]

        _draw_label_tag(
            draw,
            annotation.label,
            (anchor_x, anchor_y),
            color,
            font,
            canvas.size,
        )

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(
        preview_path,
        format="JPEG",
        quality=max(30, min(95, quality)),
        optimize=True,
    )

    metadata = {
        "width": canvas.width,
        "height": canvas.height,
        "annotationCount": len(annotations),
        "annotationKinds": sorted({annotation.kind for annotation in annotations}),
        "classIds": sorted({annotation.class_id for annotation in annotations}),
        "classLabels": sorted({annotation.label for annotation in annotations}),
    }
    return metadata, warnings


def _copy_web_assets(output_dir: Path) -> None:
    template_dir = Path(__file__).resolve().parent / "webapp"
    for asset_name in ("index.html", "app.js", "styles.css"):
        shutil.copy2(template_dir / asset_name, output_dir / asset_name)


def build_bundle(
    input_dir: Path,
    output_dir: Path,
    max_dim: int = DEFAULT_MAX_DIM,
    quality: int = DEFAULT_QUALITY,
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_entries = scan_dataset(input_dir, output_dir)
    if not dataset_entries:
        raise ValueError(f"No labeled images found under {input_dir}")

    class_names = _find_class_names(input_dir)
    dataset_key = hashlib.sha1(str(input_dir).encode("utf-8")).hexdigest()[:16]
    warnings: list[str] = []
    manifest_items: list[dict[str, Any]] = []

    for index, entry in enumerate(dataset_entries, start=1):
        preview_relpath = (
            Path("assets") / "previews" / Path(entry.relative_image_path)
        ).with_suffix(".jpg")
        preview_path = output_dir / preview_relpath
        preview_metadata, item_warnings = _render_preview_image(
            entry.image_path,
            entry.label_path,
            preview_path,
            class_names,
            max_dim=max_dim,
            quality=quality,
        )
        warnings.extend(item_warnings)

        manifest_items.append(
            {
                "id": entry.relative_image_path,
                "index": index - 1,
                "sourcePath": entry.relative_image_path,
                "labelPath": entry.relative_label_path,
                "previewPath": preview_relpath.as_posix(),
                **preview_metadata,
            }
        )

    _copy_web_assets(output_dir)

    manifest = {
        "datasetName": input_dir.name,
        "datasetKey": dataset_key,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "imageCount": len(manifest_items),
        "classNames": {str(key): value for key, value in class_names.items()},
        "warningCount": len(warnings),
        "warnings": warnings[:50],
        "images": manifest_items,
    }
    (output_dir / "manifest.js").write_text(
        "window.REVIEW_DATA = " + json.dumps(manifest, separators=(",", ":")) + ";\n"
    )

    return {
        "datasetName": input_dir.name,
        "datasetKey": dataset_key,
        "imageCount": len(manifest_items),
        "warningCount": len(warnings),
        "outputDir": str(output_dir),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a static offline review app for YOLO bbox/seg datasets"
    )
    parser.add_argument("input_dir", type=Path, help="Dataset root to scan recursively")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the offline review bundle will be written",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=DEFAULT_MAX_DIM,
        help=f"Maximum preview width/height in pixels (default: {DEFAULT_MAX_DIM})",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=DEFAULT_QUALITY,
        help=f"JPEG preview quality from 30-95 (default: {DEFAULT_QUALITY})",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    result = build_bundle(
        args.input_dir,
        args.output_dir,
        max_dim=args.max_dim,
        quality=args.quality,
    )

    print(f"Generated offline review bundle for {result['imageCount']} images")
    print(f"Output: {result['outputDir']}")
    if result["warningCount"] > 0:
        print(f"Warnings: {result['warningCount']} label lines were skipped")


if __name__ == "__main__":
    main()

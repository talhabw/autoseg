# Offline Review Bundle

This folder contains a standalone generator plus a zero-build static webapp for reviewing YOLO-style labels offline.

## What It Does

- Scans a dataset root recursively for `.png`, `.jpg`, and `.jpeg` files with same-stem `.txt` labels.
- Accepts YOLO bbox labels and YOLO polygon/segmentation labels.
- Renders compressed preview images with the labels baked in.
- Writes a self-contained static review app that can be opened directly in a browser.

## Generate A Bundle

```bash
uv run python offline_review/generate_bundle.py /path/to/dataset -o /path/to/review_bundle
```

Optional flags:

- `--max-dim 1600`
- `--quality 82`

## Open The Review App

Open `/path/to/review_bundle/index.html` in a browser.

## Review Controls

- `A` / `Left Arrow`: previous image
- `D` / `Right Arrow`: next image
- `Q`: mark or unmark the current image for removal

The removal list is saved in local storage. Stale entries are automatically removed when the generated image set changes.

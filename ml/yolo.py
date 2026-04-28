"""
YOLO detection service for assisted annotation.

The service is intentionally thin around Ultralytics so the rest of the
application deals with stable detection records instead of model-specific
result objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class YoloDetection:
    """A single YOLO detection in original-image coordinates."""

    bbox_xyxy: list[float]
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[np.ndarray] = None


class YoloService:
    """Lazy-loaded Ultralytics YOLO model wrapper."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = str(Path(model_path).expanduser())
        self.device = device
        self.model = None

    def is_loaded(self) -> bool:
        """Check whether the YOLO model is loaded."""
        return self.model is not None

    def load_model(self) -> None:
        """Load the configured YOLO model."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run `uv sync` after updating dependencies."
            ) from exc

        logger.info("Loading YOLO model from %s", self.model_path)
        self.model = YOLO(self.model_path)

    def predict(
        self,
        image_path: str,
        confidence: float = 0.25,
        iou: float = 0.7,
        imgsz: Optional[int] = None,
        max_detections: int = 300,
        class_filter: Optional[list[str]] = None,
    ) -> list[YoloDetection]:
        """Run YOLO inference on an image and return normalized detections."""
        if not self.is_loaded():
            self.load_model()

        assert self.model is not None

        predict_kwargs = {
            "source": image_path,
            "conf": max(0.0, min(1.0, confidence)),
            "iou": max(0.0, min(1.0, iou)),
            "max_det": max(1, int(max_detections)),
            "device": self.device,
            "verbose": False,
        }
        if imgsz is not None:
            predict_kwargs["imgsz"] = max(32, int(imgsz))

        results = self.model.predict(**predict_kwargs)
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)

        names = getattr(result, "names", None) or getattr(self.model, "names", {}) or {}
        orig_h, orig_w = getattr(result, "orig_shape", (None, None))
        if orig_h is None or orig_w is None:
            orig_h = int(np.max(xyxy[:, [1, 3]])) if len(xyxy) else 0
            orig_w = int(np.max(xyxy[:, [0, 2]])) if len(xyxy) else 0

        masks = self._extract_masks(result, int(orig_h), int(orig_w))
        filter_tokens = _normalize_class_filter(class_filter)

        detections: list[YoloDetection] = []
        for idx, bbox in enumerate(xyxy):
            class_id = int(class_ids[idx])
            class_name = _class_name(names, class_id)
            if not _class_allowed(class_id, class_name, filter_tokens):
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox]
            if orig_w > 0:
                x1 = max(0.0, min(float(orig_w), x1))
                x2 = max(0.0, min(float(orig_w), x2))
            if orig_h > 0:
                y1 = max(0.0, min(float(orig_h), y1))
                y2 = max(0.0, min(float(orig_h), y2))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                YoloDetection(
                    bbox_xyxy=[x1, y1, x2, y2],
                    confidence=float(confidences[idx]),
                    class_id=class_id,
                    class_name=class_name,
                    mask=masks[idx] if masks is not None and idx < len(masks) else None,
                )
            )

        return detections

    def _extract_masks(self, result, height: int, width: int) -> Optional[list[np.ndarray]]:
        """Extract binary masks from a YOLO segmentation result, if present."""
        result_masks = getattr(result, "masks", None)
        if result_masks is None or getattr(result_masks, "data", None) is None:
            return None

        import cv2

        mask_data = result_masks.data.detach().cpu().numpy()
        masks: list[np.ndarray] = []
        for mask in mask_data:
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            masks.append((mask > 0.5).astype(np.uint8))

        return masks


_YOLO_SERVICES: dict[tuple[str, str], YoloService] = {}


def get_yolo_service(model_path: str, device: str = "cuda") -> YoloService:
    """Get or create a cached YOLO service for a model path/device pair."""
    key = (str(Path(model_path).expanduser()), device)
    service = _YOLO_SERVICES.get(key)
    if service is None:
        service = YoloService(model_path=model_path, device=device)
        _YOLO_SERVICES[key] = service
    return service


def clear_yolo_services() -> None:
    """Clear cached YOLO services."""
    _YOLO_SERVICES.clear()


def _class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def _normalize_class_filter(class_filter: Optional[list[str]]) -> set[str]:
    if not class_filter:
        return set()
    return {str(item).strip().lower() for item in class_filter if str(item).strip()}


def _class_allowed(class_id: int, class_name: str, filter_tokens: set[str]) -> bool:
    if not filter_tokens:
        return True
    return str(class_id) in filter_tokens or class_name.lower() in filter_tokens

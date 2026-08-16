# -*- coding: utf-8 -*-
"""CPU EasyOCR tool with normalized positioned text observations."""

from __future__ import annotations

import hashlib
import io
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool


_SHARED_READERS: dict[bool, Any] = {}
_READER_LOCK = threading.RLock()
_READ_LOCK = threading.Lock()


@dataclass
class OCRWithPositionTool(BaseTool):
    """Extract positioned text with one process-shared CPU EasyOCR reader.

    EasyOCR is intentionally the only backend.  A reader is initialized lazily
    and shared by tool instances in the process; OCR calls are serialized because
    the same reader is not assumed to be thread-safe.
    """

    name: str = "ocr_with_position"
    description: str = (
        "Extract text from the image with position information through EasyOCR. "
        "Returns text regions, bounding boxes, confidence scores, and language."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {
                    "type": "string",
                    "description": "Local path to the image file.",
                },
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "Optional target region [x1,y1,x2,y2], using normalized "
                        "0-1 coordinates or absolute pixels."
                    ),
                },
                "goal": {
                    "type": "string",
                    "description": "What visible text or property this OCR should check.",
                },
                "source_evidence_id": {
                    "type": "string",
                    "description": "Evidence id that motivated this visual revisit.",
                },
                "expected_property": {
                    "type": "string",
                    "description": "Discriminative text or property expected in the target region.",
                },
                "visual_question_id": {
                    "type": "string",
                    "description": "Pending visual question this regional OCR resolves.",
                },
                "source_discovery_id": {
                    "type": "string",
                    "description": "Discovery id that motivated this visual revisit.",
                },
            },
            "required": ["image_input"],
        }
    )
    _reader: Optional[Any] = field(default=None, repr=False)
    min_confidence: float = 0.5
    use_gpu: Optional[bool] = None

    def _use_gpu(self) -> bool:
        if self.use_gpu is not None:
            return bool(self.use_gpu)
        return (
            os.getenv("EASYOCR_GPU", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )

    def _get_reader(self) -> Any:
        if self._reader is not None:
            return self._reader

        use_gpu = self._use_gpu()
        with _READER_LOCK:
            if use_gpu not in _SHARED_READERS:
                import easyocr

                _SHARED_READERS[use_gpu] = easyocr.Reader(
                    ["ch_sim", "en"],
                    gpu=use_gpu,
                    verbose=False,
                )
            self._reader = _SHARED_READERS[use_gpu]
        return self._reader

    def _ocr_model(self) -> str:
        try:
            import easyocr

            return f"easyocr-{getattr(easyocr, '__version__', 'unknown')}"
        except Exception:
            return "easyocr"

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_path = str(params.get("image_input", "")).strip()
        artifact_bytes = b""
        requested_bbox: List[float] = []
        attempts: List[Dict[str, Any]] = []

        try:
            from PIL import Image
            import numpy as np

            if not image_path:
                raise ValueError("image_input is required")
            with Image.open(image_path).convert("RGB") as image:
                image_width, image_height = image.size
                requested_bbox = self._normalize_bbox(
                    params.get("bbox"),
                    image_width,
                    image_height,
                )
                if requested_bbox:
                    px1 = int(round(requested_bbox[0] * image_width))
                    py1 = int(round(requested_bbox[1] * image_height))
                    px2 = int(round(requested_bbox[2] * image_width))
                    py2 = int(round(requested_bbox[3] * image_height))
                    ocr_image = image.crop((px1, py1, px2, py2))
                    offset_x, offset_y = px1, py1
                else:
                    ocr_image = image.copy()
                    offset_x = offset_y = 0
                ocr_width, ocr_height = ocr_image.size
                artifact_bytes = self._image_bytes(ocr_image)
                ocr_input = np.asarray(ocr_image)

            reader = self._get_reader()
            with _READ_LOCK:
                raw_results = reader.readtext(
                    ocr_input,
                    detail=1,
                    paragraph=False,
                )
            attempts.append(
                {
                    "backend": "easyocr",
                    "status": "success",
                    "device": "gpu" if self._use_gpu() else "cpu",
                }
            )

            text_regions: List[Dict[str, Any]] = []
            rejected_text_regions: List[Dict[str, Any]] = []
            full_text_parts: List[str] = []
            for index, raw in enumerate(raw_results or []):
                bbox, text, confidence = self._parse_result(raw, index=index)
                quad = self._normalize_quad(bbox, index=index)
                global_quad = [
                    [point[0] + offset_x, point[1] + offset_y]
                    for point in quad
                ]
                xs = [point[0] for point in global_quad]
                ys = [point[1] for point in global_quad]
                region = {
                    "text": text,
                    "bbox_quad": [
                        [
                            round(float(point[0]) / image_width, 4),
                            round(float(point[1]) / image_height, 4),
                        ]
                        for point in global_quad
                    ],
                    "bbox": [
                        round(min(xs) / image_width, 4),
                        round(min(ys) / image_height, 4),
                        round(max(xs) / image_width, 4),
                        round(max(ys) / image_height, 4),
                    ],
                    "confidence": round(float(confidence), 3),
                    "language": (
                        "zh"
                        if any(
                            0x4E00 <= ord(char) <= 0x9FFF
                            for char in text
                        )
                        else "en"
                    ),
                }
                if (
                    text.strip()
                    and float(confidence) >= float(self.min_confidence)
                ):
                    text_regions.append(region)
                    full_text_parts.append(text)
                else:
                    rejected_text_regions.append(region)

            return self._success(
                params,
                requested_bbox=requested_bbox,
                artifact_bytes=artifact_bytes,
                text_regions=text_regions,
                rejected_text_regions=rejected_text_regions,
                full_text=" ".join(full_text_parts),
                attempts=attempts,
            )
        except Exception as exc:
            if not attempts:
                attempts.append(
                    {
                        "backend": "easyocr",
                        "status": "error",
                        "device": "gpu" if self._use_gpu() else "cpu",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            return {
                "status": "error",
                "error": f"EasyOCR failed: {exc}",
                "text_regions": [],
                "total_regions": 0,
                "full_text": "",
                "rejected_text_regions": [],
                "requested_bbox": requested_bbox,
                "goal": str(params.get("goal", "")),
                "source_evidence_id": str(
                    params.get("source_evidence_id", "")
                ),
                "expected_property": str(
                    params.get("expected_property", "")
                ),
                "artifact_sha256": (
                    hashlib.sha256(artifact_bytes).hexdigest()
                    if artifact_bytes
                    else ""
                ),
                "ocr_backend": "easyocr",
                "ocr_model": self._ocr_model(),
                "backend_attempts": attempts,
                "subcalls": self._ocr_subcalls(attempts),
            }

    def _success(
        self,
        params: Dict[str, Any],
        *,
        requested_bbox: List[float],
        artifact_bytes: bytes,
        text_regions: List[Dict[str, Any]],
        rejected_text_regions: List[Dict[str, Any]],
        full_text: str,
        attempts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "status": "success",
            "text_regions": text_regions,
            "total_regions": len(text_regions),
            "full_text": full_text,
            "rejected_text_regions": rejected_text_regions,
            "requested_bbox": requested_bbox,
            "goal": str(params.get("goal", "")),
            "source_evidence_id": str(
                params.get("source_evidence_id", "")
            ),
            "expected_property": str(
                params.get("expected_property", "")
            ),
            "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "ocr_backend": "easyocr",
            "ocr_model": self._ocr_model(),
            "backend_attempts": attempts,
            "subcalls": self._ocr_subcalls(attempts),
        }

    @staticmethod
    def _parse_result(raw: Any, *, index: int) -> tuple[Any, str, float]:
        if not isinstance(raw, (list, tuple)) or len(raw) < 3:
            raise ValueError(f"EasyOCR result {index} must be [bbox, text, score]")
        bbox, text, confidence = raw[0], str(raw[1]).strip(), raw[2]
        if isinstance(confidence, bool) or not isinstance(confidence, Real):
            raise ValueError(f"EasyOCR result {index} has invalid confidence")
        return bbox, text, float(confidence)

    @staticmethod
    def _image_bytes(image: Any) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _ocr_subcalls(
        attempts: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [
            {
                "kind": "ocr",
                "provider": str(item.get("backend", "easyocr")),
                "status": str(item.get("status", "error")),
                "request_count": 1,
            }
            for item in attempts
        ]

    @staticmethod
    def _normalize_bbox(
        raw_bbox: Any,
        width: int,
        height: int,
    ) -> List[float]:
        if raw_bbox in (None, []):
            return []
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2].")
        if not all(isinstance(value, (int, float)) for value in raw_bbox):
            raise ValueError("bbox coordinates must be numeric.")
        values = [float(value) for value in raw_bbox]
        if all(0.0 <= value <= 1.0 for value in values):
            x1, y1, x2, y2 = values
        else:
            x1, y1, x2, y2 = (
                values[0] / width,
                values[1] / height,
                values[2] / width,
                values[3] / height,
            )
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError("bbox must be ordered and remain inside the image")
        if x2 - x1 <= 0.001 or y2 - y1 <= 0.001:
            raise ValueError("bbox must describe a non-empty region inside the image.")
        return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]

    @staticmethod
    def _normalize_quad(value: Any, *, index: int) -> List[List[float]]:
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            normalized_points: List[Any] = []
            for point in value:
                point_tolist = getattr(point, "tolist", None)
                if callable(point_tolist):
                    point = point_tolist()
                if isinstance(point, tuple):
                    point = list(point)
                normalized_points.append(point)
            value = normalized_points
        if (
            isinstance(value, list)
            and len(value) == 4
            and all(
                isinstance(point, list)
                and len(point) == 2
                and all(
                    isinstance(item, Real)
                    and not isinstance(item, bool)
                    for item in point
                )
                for point in value
            )
        ):
            return [[float(point[0]), float(point[1])] for point in value]
        if (
            isinstance(value, list)
            and len(value) == 4
            and all(
                isinstance(item, Real)
                and not isinstance(item, bool)
                for item in value
            )
        ):
            x1, y1, x2, y2 = [float(item) for item in value]
            if x1 < x2 and y1 < y2:
                return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        raise ValueError(f"EasyOCR result {index} requires a quadrilateral or bbox")

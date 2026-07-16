# -*- coding: utf-8 -*-
"""Layered positioned OCR with optional PP-OCR service and EasyOCR fallback.

Returns text regions with quadrilateral bounding box coordinates,
confidence scores, and language detection.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from src.tools.base import BaseTool


@dataclass
class OCRWithPositionTool(BaseTool):
    """OCR tool that returns text with position information.

    Uses a configured PP-OCR/PP-Structure-compatible HTTP service when present,
    with a local EasyOCR fallback.
    Supports Chinese + English. Returns both:
    - bbox_quad: normalized 4-corner quadrilateral for pipeline contracts
    - bbox: normalized axis-aligned box for crop-oriented tools
    """

    name: str = "ocr_with_position"
    description: str = (
        "Extract text from the image with position information. "
        "Returns each text region with its bounding box (normalized 0-1), "
        "the recognized text, confidence score, and detected language. "
        "Use when you need to know what text is in the image and where it is."
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
                    "description": "What visible text or property this regional OCR should check.",
                },
                "source_evidence_id": {
                    "type": "string",
                    "description": "Evidence id that motivated this visual revisit, when applicable.",
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
                    "description": "Discovery id that motivated this revisit, when applicable.",
                },
            },
            "required": ["image_input"],
        }
    )

    _reader: Optional[Any] = field(default=None, repr=False)
    min_confidence: float = 0.5
    use_gpu: Optional[bool] = None
    ppocr_service_url: str = ""
    ppocr_timeout: float = 45.0

    def __post_init__(self) -> None:
        self.ppocr_service_url = (
            self.ppocr_service_url
            or os.getenv("PPOCR_SERVICE_URL", "").strip()
        )

    def _get_reader(self):
        """Lazy initialization of EasyOCR reader."""
        if self._reader is None:
            import easyocr

            gpu = (
                self.use_gpu
                if self.use_gpu is not None
                else os.getenv(
                    "EASYOCR_GPU",
                    "false",
                ).strip().lower()
                in {"1", "true", "yes", "on"}
            )
            self._reader = easyocr.Reader(
                ["ch_sim", "en"],
                gpu=gpu,
                verbose=False,
            )
        return self._reader

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run OCR on the image and return structured results."""
        image_path = params["image_input"]
        artifact_bytes = b""
        requested_bbox: List[float] = []

        try:
            from PIL import Image

            img = Image.open(image_path).convert("RGB")
            img_w, img_h = img.size
            requested_bbox = self._normalize_bbox(params.get("bbox"), img_w, img_h)
            if requested_bbox:
                px1 = int(round(requested_bbox[0] * img_w))
                py1 = int(round(requested_bbox[1] * img_h))
                px2 = int(round(requested_bbox[2] * img_w))
                py2 = int(round(requested_bbox[3] * img_h))
                cropped = img.crop((px1, py1, px2, py2))
                ocr_input = __import__("numpy").asarray(cropped)
                offset_x, offset_y = px1, py1
                artifact_bytes = self._image_bytes(cropped)
            else:
                ocr_input = image_path
                offset_x = offset_y = 0
                artifact_bytes = self._image_bytes(img)

            results, backend, backend_attempts = self._run_layered_ocr(
                ocr_input,
                artifact_bytes,
            )

            if not results:
                return {
                    "status": "success",
                    "text_regions": [],
                    "total_regions": 0,
                    "full_text": "",
                    "rejected_text_regions": [],
                    "requested_bbox": requested_bbox,
                    "goal": str(params.get("goal", "")),
                    "source_evidence_id": str(params.get("source_evidence_id", "")),
                    "expected_property": str(params.get("expected_property", "")),
                    "artifact_sha256": hashlib.sha256(
                        artifact_bytes
                    ).hexdigest(),
                    "ocr_backend": backend,
                    "backend_attempts": backend_attempts,
                    "subcalls": self._ocr_subcalls(backend_attempts),
                }

            text_regions: List[Dict[str, Any]] = []
            rejected_text_regions: List[Dict[str, Any]] = []
            full_text_parts: List[str] = []

            for bbox, text, confidence in results:
                global_bbox = [[p[0] + offset_x, p[1] + offset_y] for p in bbox]
                xs = [p[0] for p in global_bbox]
                ys = [p[1] for p in global_bbox]
                x1 = min(xs) / img_w
                y1 = min(ys) / img_h
                x2 = max(xs) / img_w
                y2 = max(ys) / img_h
                bbox_quad = [
                    [round(float(px) / img_w, 4), round(float(py) / img_h, 4)]
                    for px, py in global_bbox
                ]

                has_chinese = any(0x4E00 <= ord(char) <= 0x9FFF for char in text)
                language = "zh" if has_chinese else "en"

                region = {
                    "text": text,
                    "bbox_quad": bbox_quad,
                    "bbox": [
                        round(float(x1), 4),
                        round(float(y1), 4),
                        round(float(x2), 4),
                        round(float(y2), 4),
                    ],
                    "confidence": round(float(confidence), 3),
                    "language": language,
                }
                if (
                    text.strip()
                    and float(confidence) >= float(self.min_confidence)
                ):
                    text_regions.append(region)
                    full_text_parts.append(text)
                else:
                    rejected_text_regions.append(region)

        except Exception as e:
            backend_attempts = list(
                getattr(e, "backend_attempts", []) or []
            )
            return {
                "status": "error",
                "error": f"OCR failed: {str(e)}",
                "text_regions": [],
                "total_regions": 0,
                "full_text": "",
                "rejected_text_regions": [],
                "requested_bbox": requested_bbox,
                "artifact_sha256": (
                    hashlib.sha256(artifact_bytes).hexdigest()
                    if artifact_bytes
                    else ""
                ),
                "backend_attempts": backend_attempts,
                "subcalls": self._ocr_subcalls(backend_attempts),
            }
        return {
            "status": "success",
            "text_regions": text_regions,
            "total_regions": len(text_regions),
            "full_text": " ".join(full_text_parts),
            "rejected_text_regions": rejected_text_regions,
            "requested_bbox": requested_bbox,
            "goal": str(params.get("goal", "")),
            "source_evidence_id": str(params.get("source_evidence_id", "")),
            "expected_property": str(params.get("expected_property", "")),
            "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "ocr_backend": backend,
            "backend_attempts": backend_attempts,
            "subcalls": self._ocr_subcalls(backend_attempts),
        }

    def _run_layered_ocr(
        self,
        ocr_input: Any,
        artifact_bytes: bytes,
    ) -> tuple[List[Any], str, List[Dict[str, str]]]:
        attempts: List[Dict[str, str]] = []
        if self.ppocr_service_url:
            try:
                results = self._call_ppocr_service(artifact_bytes)
                attempts.append(
                    {"backend": "ppocr_service", "status": "success"}
                )
                return results, "ppocr_service", attempts
            except Exception as exc:
                attempts.append(
                    {
                        "backend": "ppocr_service",
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        try:
            reader = self._get_reader()
            results = reader.readtext(ocr_input)
            attempts.append({"backend": "easyocr", "status": "success"})
            return list(results or []), "easyocr", attempts
        except Exception as exc:
            attempts.append(
                {
                    "backend": "easyocr",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            error = RuntimeError("all configured OCR backends failed")
            error.backend_attempts = attempts  # type: ignore[attr-defined]
            raise error from exc

    def _call_ppocr_service(self, image_bytes: bytes) -> List[Any]:
        response = requests.post(
            self.ppocr_service_url,
            json={
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                "response_format": "ifv_positioned_ocr_v1",
            },
            timeout=self.ppocr_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("PP-OCR service response must be an object")
        rows = payload.get("text_regions", payload.get("regions"))
        if not isinstance(rows, list):
            raise ValueError(
                "PP-OCR service response requires text_regions or regions"
            )
        results: List[Any] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"PP-OCR region {index} must be an object")
            text = str(row.get("text", "")).strip()
            confidence = row.get("confidence", row.get("score"))
            bbox = row.get("bbox_quad", row.get("points", row.get("bbox")))
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
            ):
                raise ValueError(
                    f"PP-OCR region {index} requires numeric confidence"
                )
            quad = self._service_quad(bbox, index=index)
            results.append((quad, text, float(confidence)))
        return results

    @staticmethod
    def _service_quad(value: Any, *, index: int) -> List[List[float]]:
        if (
            isinstance(value, list)
            and len(value) == 4
            and all(
                isinstance(point, (list, tuple))
                and len(point) == 2
                and all(
                    isinstance(item, (int, float))
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
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                for item in value
            )
        ):
            x1, y1, x2, y2 = [float(item) for item in value]
            if x1 < x2 and y1 < y2:
                return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        raise ValueError(
            f"PP-OCR region {index} requires bbox_quad or ordered bbox"
        )

    @staticmethod
    def _image_bytes(image: Any) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _ocr_subcalls(
        attempts: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        return [
            {
                "kind": "ocr",
                "provider": str(attempt.get("backend", "unknown")),
                "status": str(attempt.get("status", "error")),
                "request_count": 1,
            }
            for attempt in attempts
        ]

    @staticmethod
    def _normalize_bbox(raw_bbox: Any, width: int, height: int) -> List[float]:
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
            raise ValueError(
                "bbox must be ordered and remain inside the image"
            )
        if x2 - x1 <= 0.001 or y2 - y1 <= 0.001:
            raise ValueError("bbox must describe a non-empty region inside the image.")
        return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]

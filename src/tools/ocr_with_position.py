# -*- coding: utf-8 -*-
"""OCR with position information using EasyOCR.

Returns text regions with quadrilateral bounding box coordinates,
confidence scores, and language detection. Uses PyTorch backend
(no PaddlePaddle dependency).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool


@dataclass
class OCRWithPositionTool(BaseTool):
    """OCR tool that returns text with position information.

    Uses EasyOCR (PyTorch-based) for text detection and recognition.
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

    def _get_reader(self):
        """Lazy initialization of EasyOCR reader."""
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(
                ["ch_sim", "en"],
                gpu=False,
                verbose=False,
            )
        return self._reader

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run OCR on the image and return structured results."""
        image_path = params["image_input"]

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
                ocr_input = __import__("numpy").asarray(img.crop((px1, py1, px2, py2)))
                offset_x, offset_y = px1, py1
            else:
                ocr_input = image_path
                offset_x = offset_y = 0

            reader = self._get_reader()
            results = reader.readtext(ocr_input)

            if not results:
                return {
                    "status": "success",
                    "text_regions": [],
                    "total_regions": 0,
                    "full_text": "",
                    "requested_bbox": requested_bbox,
                    "goal": str(params.get("goal", "")),
                    "source_evidence_id": str(params.get("source_evidence_id", "")),
                    "expected_property": str(params.get("expected_property", "")),
                }

            text_regions: List[Dict[str, Any]] = []
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

                text_regions.append(
                    {
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
                )
                full_text_parts.append(text)

        except Exception as e:
            return {
                "status": "error",
                "error": f"OCR failed: {str(e)}",
                "text_regions": [],
                "total_regions": 0,
                "full_text": "",
            }
        finally:
            self._release_reader()

        return {
            "status": "success",
            "text_regions": text_regions,
            "total_regions": len(text_regions),
            "full_text": " ".join(full_text_parts),
            "requested_bbox": requested_bbox,
            "goal": str(params.get("goal", "")),
            "source_evidence_id": str(params.get("source_evidence_id", "")),
            "expected_property": str(params.get("expected_property", "")),
        }

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
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        if x2 - x1 <= 0.001 or y2 - y1 <= 0.001:
            raise ValueError("bbox must describe a non-empty region inside the image.")
        return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]

    def _release_reader(self):
        """Release EasyOCR reader to free memory."""
        import gc

        self._reader = None
        gc.collect()

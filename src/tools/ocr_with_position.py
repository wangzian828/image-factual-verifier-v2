# -*- coding: utf-8 -*-
"""OCR with position information using EasyOCR.

Returns text regions with bounding box coordinates, confidence scores,
and language detection. Uses PyTorch backend (no PaddlePaddle dependency).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool


@dataclass
class OCRWithPositionTool(BaseTool):
    """OCR tool that returns text with position information.

    Uses EasyOCR (PyTorch-based) for text detection and recognition.
    Supports Chinese + English. Returns bounding boxes as normalized
    coordinates (0-1) for compatibility with crop tools.
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
            },
            "required": ["image_input"],
        }
    )

    _reader: Optional[Any] = field(default=None, repr=False)

    def _get_reader(self):
        """Lazy initialization of EasyOCR reader."""
        if self._reader is None:
            import easyocr
            # Support Chinese (simplified + traditional) and English
            # Use CPU to avoid cuDNN version issues
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

            img = Image.open(image_path)
            img_w, img_h = img.size

            reader = self._get_reader()
            # EasyOCR returns: [[bbox, text, confidence], ...]
            # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] (4 corners)
            results = reader.readtext(image_path)

            if not results:
                return {
                    "status": "success",
                    "text_regions": [],
                    "total_regions": 0,
                    "full_text": "",
                }

            text_regions: List[Dict[str, Any]] = []
            full_text_parts: List[str] = []

            for bbox, text, confidence in results:
                # Convert 4-corner bbox to normalized [x1, y1, x2, y2]
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x1 = min(xs) / img_w
                y1 = min(ys) / img_h
                x2 = max(xs) / img_w
                y2 = max(ys) / img_h

                # Detect language (simple heuristic)
                has_chinese = any('一' <= c <= '鿿' for c in text)
                language = "zh" if has_chinese else "en"

                text_regions.append({
                    "text": text,
                    "bbox": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
                    "confidence": round(float(confidence), 3),
                    "language": language,
                })
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
            # Release model to free memory
            self._release_reader()

        return {
            "status": "success",
            "text_regions": text_regions,
            "total_regions": len(text_regions),
            "full_text": " ".join(full_text_parts),
        }

    def _release_reader(self):
        """Release EasyOCR reader to free memory."""
        import gc
        self._reader = None
        gc.collect()

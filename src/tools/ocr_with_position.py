# -*- coding: utf-8 -*-
"""OCR with position information using PaddleOCR.

Returns text regions with bounding box coordinates, confidence scores,
and detected language. Runs on CPU.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool


def _detect_language(text: str) -> str:
    """Simple heuristic language detection."""
    if not text:
        return "unknown"
    # Count CJK characters
    cjk_count = len(re.findall(r"[一-鿿㐀-䶿]", text))
    if cjk_count > len(text) * 0.3:
        return "zh"
    # Count Korean
    korean_count = len(re.findall(r"[가-힯]", text))
    if korean_count > len(text) * 0.3:
        return "ko"
    # Count Japanese hiragana/katakana
    jp_count = len(re.findall(r"[぀-ゟ゠-ヿ]", text))
    if jp_count > len(text) * 0.2:
        return "ja"
    # Default to English/Latin
    return "en"


@dataclass
class OCRWithPositionTool(BaseTool):
    """OCR tool using PaddleOCR that returns text with bounding box positions.

    Unlike the old VLM-based OCR, this provides:
    - Precise bounding box coordinates (quad points)
    - Per-region confidence scores
    - Language detection
    - Works on CPU with ~0.5-2s per image
    """

    name: str = "ocr_with_position"
    description: str = (
        "Extract text from the image with precise bounding box positions. "
        "Returns a list of text regions, each with the recognized text, "
        "bounding box coordinates (4 corner points), confidence score, and language. "
        "Use this to know WHERE text appears in the image and read it accurately."
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

    _ocr: Optional[Any] = field(default=None, repr=False)

    def _get_ocr(self):
        """Lazy initialization of PaddleOCR (loads model on first call)."""
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="ch",  # Chinese + English
                use_gpu=False,
                show_log=False,
            )
        return self._ocr

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run OCR on the image and return structured results."""
        image_path = params["image_input"]

        try:
            ocr = self._get_ocr()
            result = ocr.ocr(image_path, cls=True)
        except Exception as e:
            return {
                "status": "error",
                "error": f"OCR failed: {str(e)}",
                "text_regions": [],
            }

        text_regions: List[Dict[str, Any]] = []

        if result and result[0]:
            for line in result[0]:
                bbox_quad = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text = line[1][0]  # recognized text
                confidence = float(line[1][1])  # confidence score

                text_regions.append({
                    "text": text,
                    "bbox_quad": bbox_quad,
                    "confidence": round(confidence, 3),
                    "language": _detect_language(text),
                })

        return {
            "status": "success",
            "text_regions": text_regions,
            "total_regions": len(text_regions),
            "full_text": "\n".join(r["text"] for r in text_regions),
        }

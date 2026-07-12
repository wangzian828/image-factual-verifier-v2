# -*- coding: utf-8 -*-
"""Crop a region from the image and inspect it with VLM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.integrations.gemini import RUNTIME_METRICS_KEY, exception_runtime_metrics
from src.tools.base import BaseTool


INSPECT_PROMPT_TEMPLATE = """\
You are inspecting a cropped image region for image verification.
Focus question: {focus_question}

Return exactly one JSON object:
{{
  "description": "literal description of the cropped region",
  "findings": ["finding 1", "finding 2"],
  "anomalies": ["anomaly 1"],
  "answer": "direct answer to the focus question"
}}

Be concrete and conservative. Output JSON only.
"""

INSPECT_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "maxLength": 700},
        "findings": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
            "maxItems": 8,
        },
        "anomalies": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
            "maxItems": 8,
        },
        "answer": {"type": "string", "maxLength": 700},
    },
}


@dataclass
class CropAndInspectTool(BaseTool):
    """Crop a region from the image and analyze it in detail."""

    name: str = "crop_and_inspect"
    description: str = (
        "Crop a specific region from the image and analyze it in detail. "
        "Provide a normalized bounding box [x1, y1, x2, y2] and a focus question."
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
                    "description": "Bounding box [x1, y1, x2, y2] normalized to 0-1.",
                },
                "focus_question": {
                    "type": "string",
                    "description": "What to inspect in this cropped region.",
                },
                "visual_question_id": {"type": "string"},
                "source_evidence_id": {"type": "string"},
                "source_discovery_id": {"type": "string"},
                "expected_property": {"type": "string"},
            },
            "required": ["image_input", "bbox", "focus_question"],
        }
    )

    client: Optional[Any] = field(default=None, repr=False)
    provider: str = "gemini"
    model_name: str = "gemini-3.5-flash"

    def _get_client(self):
        if self.client is None:
            from src.integrations.vlm.factory import build_vlm_client

            self.client = build_vlm_client(
                provider=self.provider,
                model_name=self.model_name,
            )
        return self.client

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_path = params["image_input"]
        bbox = params["bbox"]
        focus_question = params["focus_question"]

        if not bbox or len(bbox) != 4:
            return {
                "status": "error",
                "error": "bbox must be [x1, y1, x2, y2] with 4 normalized values.",
            }

        try:
            import os
            import tempfile
            from PIL import Image

            img = Image.open(image_path)
            w, h = img.size

            x1, y1, x2, y2 = self._bbox_to_pixels(bbox, w, h)

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            cropped = img.crop((x1, y1, x2, y2))
            tmp_path = os.path.join(
                tempfile.gettempdir(),
                f"crop_{os.getpid()}_{id(self)}.png",
            )
            cropped.save(tmp_path)
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Crop failed: {exc}",
            }

        try:
            client = self._get_client()
            parsed = client.create_image_json(
                system_prompt=INSPECT_PROMPT_TEMPLATE.format(focus_question=focus_question),
                user_text=f"Inspect this cropped region. Focus: {focus_question}",
                image_input=tmp_path,
                max_tokens=1000,
                model_name=self.model_name,
                response_schema=INSPECT_SCHEMA,
            )
        except Exception as exc:
            error = {
                "status": "error",
                "error": f"VLM inspection failed: {exc}",
            }
            metrics = exception_runtime_metrics(exc)
            if metrics:
                error[RUNTIME_METRICS_KEY] = metrics
            return error
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        return {
            "status": "success",
            "description": str(parsed.get("description", "")),
            "findings": parsed.get("findings", []),
            "anomalies": parsed.get("anomalies", []),
            "answer": str(parsed.get("answer", "")),
            "crop_bbox": bbox,
            "focus_question": focus_question,
            RUNTIME_METRICS_KEY: parsed.get(RUNTIME_METRICS_KEY, {}),
        }

    @staticmethod
    def _bbox_to_pixels(bbox: Any, width: int, height: int) -> tuple[int, int, int, int]:
        values = [float(v) for v in bbox]
        max_value = max(values)

        if max_value <= 1.5:
            x1 = int(values[0] * width)
            y1 = int(values[1] * height)
            x2 = int(values[2] * width)
            y2 = int(values[3] * height)
        elif max_value <= 1000.0:
            x1 = int(values[0] / 1000.0 * width)
            y1 = int(values[1] / 1000.0 * height)
            x2 = int(values[2] / 1000.0 * width)
            y2 = int(values[3] / 1000.0 * height)
        else:
            x1, y1, x2, y2 = [int(v) for v in values]

        return x1, y1, x2, y2

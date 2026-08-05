# -*- coding: utf-8 -*-
"""Scene perception using VLM with structured output."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool
from src.integrations.gemini import RUNTIME_METRICS_KEY, exception_runtime_metrics


PERCEIVE_SCENE_PROMPT = """\
You are the perception module of an image verification system.
Inspect the image carefully and return exactly one JSON object:
{
  "entities": [
    {
      "name": "literal visible label or generic descriptor",
      "entity_type": "person|object|building|logo|animal|scene_element",
      "bbox": [x_min, y_min, x_max, y_max],
      "confidence": 0.9
    }
  ],
  "scene_description": "one-sentence literal description",
  "image_type": "photo|screenshot|document|illustration|meme"
}

Rules:
1. List at most 8 decision-relevant visible entities.
2. For people, do not assign a proper-name identity from appearance alone;
   use a generic visible descriptor such as "pilot", "man in dark suit", or
   "unidentified person" unless visible text explicitly labels the person.
3. Include visible logos, but do not transcribe text or describe entity attributes;
   a separate OCR stage handles visible text.
4. Use normalized [x_min, y_min, x_max, y_max] bounding boxes in [0,1].
   If no reliable box is available, use [].
5. Keep every entity name under 100 characters.
6. Keep scene_description to one literal, objective sentence under 280 characters.
7. Do not include explanations, hidden-state reasoning, history, biographies, or
   information that is not directly visible in the pixels.
8. Output JSON only.
"""

PERCEIVE_SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 100},
                    "entity_type": {
                        "type": "string",
                        "enum": [
                            "person",
                            "object",
                            "building",
                            "logo",
                            "animal",
                            "scene_element",
                        ],
                    },
                    "bbox": {"type": "array", "items": {"type": "number"}, "maxItems": 4},
                    "confidence": {"type": "number"},
                },
            },
        },
        "scene_description": {"type": "string", "maxLength": 280},
        "image_type": {
            "type": "string",
            "enum": ["photo", "screenshot", "document", "illustration", "meme"],
        },
    },
}


@dataclass
class PerceiveSceneTool(BaseTool):
    """VLM-based scene perception that outputs a structured entity list."""

    name: str = "perceive_scene"
    description: str = (
        "Observe the image and extract a structured list of visible entities "
        "(people, objects, logos, buildings, animals), their types, and approximate "
        "positions. Also determine the image type and provide a "
        "scene description."
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

    client: Optional[Any] = field(default=None, repr=False)
    provider: str = "gemini"
    model_name: str = "gemini-3.6-flash"

    def _get_client(self):
        if self.client is None:
            from src.integrations.vlm.factory import build_vlm_client

            self.client = build_vlm_client(
                provider=self.provider,
                model_name=self.model_name,
            )
        return self.client

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_input = params["image_input"]

        try:
            client = self._get_client()
            parsed = client.create_image_json(
                system_prompt=PERCEIVE_SCENE_PROMPT,
                user_text="Inspect this image and output the structured scene JSON.",
                image_input=image_input,
                max_tokens=2000,
                model_name=self.model_name,
                response_schema=PERCEIVE_SCENE_SCHEMA,
            )
        except Exception as exc:
            error = {
                "status": "error",
                "error": (
                    "Scene perception failed: "
                    f"{type(exc).__name__}: {exc or '<no message>'}"
                ),
                "entities": [],
                "scene_description": "",
                "image_type": "unknown",
            }
            metrics = exception_runtime_metrics(exc)
            if metrics:
                error[RUNTIME_METRICS_KEY] = metrics
            return error

        entities = []
        bbox_warnings = []
        for index, ent in enumerate(parsed.get("entities", [])):
            if not isinstance(ent, dict):
                continue
            try:
                bbox = normalize_entity_bbox(
                    ent.get("bbox", []),
                    thousand_scale_order=(
                        "yxyx" if self.provider.strip().lower() == "gemini" else "xyxy"
                    ),
                )
            except ValueError as exc:
                # A model may return one malformed or mixed-scale region while
                # still producing a useful literal scene report. Never guess or
                # repair that geometry: retain the entity as an unlocalized
                # observation and keep every valid sibling bbox.
                bbox = []
                bbox_warnings.append(
                    {
                        "entity_index": index,
                        "entity_name": str(ent.get("name", "")).strip()[:100],
                        "error": str(exc),
                    }
                )
            entities.append(
                {
                    "name": str(ent.get("name", "")).strip(),
                    "entity_type": str(ent.get("entity_type", "object")).strip(),
                    "bbox": bbox,
                    "confidence": float(ent.get("confidence", 0.8)),
                    "attributes": {},
                }
            )

        return {
            "status": "success",
            "entities": entities[:8],
            "scene_description": str(parsed.get("scene_description", "")).strip(),
            "image_type": str(parsed.get("image_type", "photo")).strip(),
            "total_entities": len(entities[:8]),
            "bbox_warnings": bbox_warnings[:8],
            RUNTIME_METRICS_KEY: parsed.get(RUNTIME_METRICS_KEY, {}),
        }


def normalize_entity_bbox(
    raw_bbox: Any,
    *,
    thousand_scale_order: str = "yxyx",
) -> List[float]:
    """Convert a model bbox to normalized project-order XYXY coordinates.

    The public tool contract is ``[x_min, y_min, x_max, y_max]`` in ``[0, 1]``.
    Gemini vision can nevertheless emit its native ``[y_min, x_min, y_max,
    x_max]`` coordinates on a 0..1000 grid. Non-empty boxes outside these two
    documented forms are rejected so incorrect regions cannot enter ReInspect.
    """

    if raw_bbox in (None, []):
        return []
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise ValueError("bbox must contain exactly four coordinates")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in raw_bbox):
        raise ValueError("bbox coordinates must be numeric")
    values = [float(value) for value in raw_bbox]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox coordinates must be finite")
    if any(0.0 < value < 1.0 for value in values) and any(
        value > 1.0 for value in values
    ):
        raise ValueError(
            "bbox mixes normalized and Gemini 0..1000 coordinate scales"
        )

    if all(0.0 <= value <= 1.0 for value in values):
        x1, y1, x2, y2 = values
    elif all(0.0 <= value <= 1000.0 for value in values) and any(
        value > 1.0 for value in values
    ):
        if thousand_scale_order == "yxyx":
            y1, x1, y2, x2 = (value / 1000.0 for value in values)
        elif thousand_scale_order == "xyxy":
            x1, y1, x2, y2 = (value / 1000.0 for value in values)
        else:
            raise ValueError("thousand_scale_order must be 'xyxy' or 'yxyx'")
    else:
        raise ValueError("bbox must use normalized XYXY or Gemini 0..1000 YXYX coordinates")

    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("bbox must be ordered, normalized, and non-empty")
    return [round(value, 6) for value in (x1, y1, x2, y2)]

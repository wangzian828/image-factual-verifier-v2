# -*- coding: utf-8 -*-
"""Compare the current image with one reference image using Gemini Interactions."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from src.integrations.gemini import (
    extract_text,
    normalize_json_schema,
    validate_interaction_response,
)
from src.tools.base import BaseTool


DIFFERENCE_TYPES = (
    "crop",
    "color_adjust",
    "watermark",
    "perspective",
    "lighting",
    "compression",
    "occlusion",
    "addition",
    "removal",
    "modification",
    "different_capture",
    "unrelated_content",
    "uncertain",
)
EDIT_DIFFERENCE_TYPES = frozenset({"addition", "removal", "modification"})
EDIT_STRENGTHS = ("none", "weak", "moderate", "strong")
SIGNIFICANCE_LEVELS = ("high", "medium", "low")

COMPARE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "same_subject_or_scene": {"type": "boolean"},
        "same_capture_or_near_duplicate": {"type": "boolean"},
        "likely_different_original_capture": {"type": "boolean"},
        "edit_evidence_present": {"type": "boolean"},
        "edit_evidence_strength": {
            "type": "string",
            "enum": list(EDIT_STRENGTHS),
        },
        "differences": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "maxLength": 300},
                    "description": {"type": "string", "maxLength": 800},
                    "type": {"type": "string", "enum": list(DIFFERENCE_TYPES)},
                    "significance": {
                        "type": "string",
                        "enum": list(SIGNIFICANCE_LEVELS),
                    },
                    "is_edit_evidence": {"type": "boolean"},
                },
            },
        },
        "overall_observation": {"type": "string", "maxLength": 1200},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

COMPARE_PROMPT = """\
Compare the two supplied images for evidence extraction, not final fact-check judgment.

Image 1 is a candidate reference image found through search.
Image 2 is the current image being verified.

Focus: {focus}

Describe concrete visual relationships and differences only.
1. Decide whether the images show the same subject, event, object, or scene.
2. Distinguish the same original capture or a near-duplicate from different original captures.
3. Mark edit evidence only for directly visible addition, removal, or modification.
4. Treat crop, resizing, compression, lighting, perspective, watermark, occlusion, and
   color shifts as benign unless they clearly alter factual content.
5. If the images are unrelated, report unrelated content without inferring manipulation.
6. Keep edit evidence false when uncertain and explain the uncertainty.
"""

SYSTEM_INSTRUCTION = (
    "You are an image comparison analyst. Return exactly one JSON object matching "
    "the supplied response schema, without markdown or commentary."
)


@dataclass
class CompareWithReferenceTool(BaseTool):
    """Compare the current image with a reference image for visual evidence."""

    name: str = "compare_with_reference"
    description: str = (
        "Compare the current image with a reference image found via search and return "
        "low-level evidence about whether the images are near-duplicates, different "
        "original captures, or show direct edit evidence."
    )
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "reference_url": {
                "type": "string",
                "description": "URL of the reference image to compare against.",
            },
            "focus": {
                "type": "string",
                "description": (
                    "What to focus on in the comparison. For example: "
                    "'number of windows', 'person on the left', or 'banner text'."
                ),
            },
            "visual_question_id": {"type": "string"},
            "source_evidence_id": {"type": "string"},
            "source_discovery_id": {"type": "string"},
            "expected_property": {"type": "string"},
        },
        "required": ["reference_url"],
    })

    # Kept as the injected backend attribute for compatibility with existing wiring.
    vlm_backend: Any = None
    image_path: str = ""

    def call(self, params: Dict[str, Any]) -> Any:
        """Synchronous compatibility entry point."""
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.call_async(params))

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            future = pool.submit(asyncio.run, self.call_async(params))
            return future.result(timeout=120)

    async def call_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Download the reference and compare both images through Interactions."""
        reference_url = str(params.get("reference_url", "")).strip()
        focus = str(params.get("focus", "general comparison")).strip() or "general comparison"

        if not reference_url:
            return self._error("reference_url is required.")
        if self.vlm_backend is None:
            return self._error("VLM backend not configured for image comparison.")
        if not callable(getattr(self.vlm_backend, "create_interaction", None)):
            return self._error(
                "Gemini Interactions backend not configured for image comparison."
            )

        try:
            reference_data_url = await self._download_reference(reference_url)
            if not reference_data_url:
                return self._error(
                    f"Could not download reference image from {reference_url}"
                )

            from src.tools.vision_utils import image_to_data_url

            current_data_url = image_to_data_url(self.image_path)
            input_payload = [
                {"type": "text", "text": COMPARE_PROMPT.format(focus=focus)},
                self._data_url_to_image_content(reference_data_url),
                self._data_url_to_image_content(current_data_url),
            ]
            schema = normalize_json_schema(
                COMPARE_RESPONSE_SCHEMA,
                require_all_properties=True,
            )
            payload = await self.vlm_backend.create_interaction(
                input_payload=input_payload,
                system_instruction=SYSTEM_INSTRUCTION,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                },
                store=True,
                max_tokens=8192,
                temperature=0.0,
            )
            _, status = validate_interaction_response(payload)
            if status != "completed":
                raise RuntimeError(
                    "Gemini comparison requires status=completed, "
                    f"received status={status}."
                )

            content = extract_text(payload)
            if not content.strip():
                raise ValueError("Gemini Interactions comparison response was empty.")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Gemini Interactions comparison response was not valid JSON."
                ) from exc
            validated = self._validate_response(parsed)
        except Exception as exc:
            return self._error(
                "Gemini Interactions comparison failed: "
                f"{type(exc).__name__}: {exc or '<no message>'}"
            )

        return {"status": "success", "reference_url": reference_url, **validated}

    async def _download_reference(self, url: str) -> Optional[str]:
        """Download a reference image and convert it to a data URL."""
        import base64

        import httpx

        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    return None

                content_type = response.headers.get("content-type", "image/jpeg")
                content_type = content_type.split(";", 1)[0].strip().lower()
                if not content_type.startswith("image/"):
                    return None
                encoded = base64.b64encode(response.content).decode("ascii")
                return f"data:{content_type};base64,{encoded}"
        except Exception:
            return None

    @staticmethod
    def _data_url_to_image_content(data_url: str) -> Dict[str, Any]:
        if not (data_url.startswith("data:") and ";base64," in data_url):
            raise ValueError("Comparison images must be base64 data URLs.")
        header, data = data_url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0].strip().lower()
        if not mime_type.startswith("image/") or not data:
            raise ValueError("Comparison image data URL is invalid.")
        return {"type": "image", "mime_type": mime_type, "data": data}

    @classmethod
    def _validate_response(cls, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("comparison output must be a JSON object")

        expected = set(COMPARE_RESPONSE_SCHEMA["properties"])
        actual = set(value)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            raise ValueError("comparison output is missing fields: " + ", ".join(missing))
        if extra:
            raise ValueError("comparison output has unexpected fields: " + ", ".join(extra))

        boolean_fields = (
            "same_subject_or_scene",
            "same_capture_or_near_duplicate",
            "likely_different_original_capture",
            "edit_evidence_present",
        )
        for name in boolean_fields:
            if not isinstance(value[name], bool):
                raise ValueError(f"comparison output field '{name}' must be boolean")

        edit_strength = value["edit_evidence_strength"]
        if edit_strength not in EDIT_STRENGTHS:
            raise ValueError("comparison output has invalid edit_evidence_strength")

        overall = value["overall_observation"]
        if not isinstance(overall, str) or not overall.strip():
            raise ValueError(
                "comparison output field 'overall_observation' must be a non-empty string"
            )

        confidence = value["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("comparison output field 'confidence' must be a number")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("comparison output field 'confidence' must be between 0 and 1")

        differences = value["differences"]
        if not isinstance(differences, list):
            raise ValueError("comparison output field 'differences' must be an array")
        if len(differences) > 20:
            raise ValueError("comparison output field 'differences' exceeds 20 items")

        validated_differences = []
        for index, item in enumerate(differences):
            validated_differences.append(cls._validate_difference(item, index))

        same_subject = value["same_subject_or_scene"]
        same_capture = value["same_capture_or_near_duplicate"]
        different_capture = value["likely_different_original_capture"]
        edit_present = value["edit_evidence_present"]
        edit_items = [item for item in validated_differences if item["is_edit_evidence"]]

        if same_capture and not same_subject:
            raise ValueError("same capture requires same_subject_or_scene=true")
        if different_capture and not same_subject:
            raise ValueError("different original capture requires same_subject_or_scene=true")
        if same_capture and different_capture:
            raise ValueError("same capture and different original capture cannot both be true")
        if edit_present != bool(edit_items):
            raise ValueError(
                "edit_evidence_present must match differences marked as edit evidence"
            )
        if edit_present and edit_strength == "none":
            raise ValueError("edit evidence cannot have strength 'none'")
        if not edit_present and edit_strength != "none":
            raise ValueError("edit_evidence_strength must be 'none' without edit evidence")

        return {
            "same_subject_or_scene": same_subject,
            "same_capture_or_near_duplicate": same_capture,
            "likely_different_original_capture": different_capture,
            "edit_evidence_present": edit_present,
            "edit_evidence_strength": edit_strength,
            "differences": validated_differences,
            "overall_observation": overall.strip(),
            "confidence": float(confidence),
        }

    @staticmethod
    def _validate_difference(value: Any, index: int) -> Dict[str, Any]:
        path = f"differences[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"comparison output {path} must be an object")

        expected = {"region", "description", "type", "significance", "is_edit_evidence"}
        if set(value) != expected:
            raise ValueError(f"comparison output {path} has invalid fields")
        for name in ("region", "description"):
            if not isinstance(value[name], str) or not value[name].strip():
                raise ValueError(f"comparison output {path}.{name} must be a non-empty string")

        difference_type = value["type"]
        if difference_type not in DIFFERENCE_TYPES:
            raise ValueError(f"comparison output {path}.type is invalid")
        significance = value["significance"]
        if significance not in SIGNIFICANCE_LEVELS:
            raise ValueError(f"comparison output {path}.significance is invalid")
        is_edit = value["is_edit_evidence"]
        if not isinstance(is_edit, bool):
            raise ValueError(f"comparison output {path}.is_edit_evidence must be boolean")
        if is_edit != (difference_type in EDIT_DIFFERENCE_TYPES):
            raise ValueError(
                f"comparison output {path}.is_edit_evidence does not match its type"
            )

        return {
            "region": value["region"].strip(),
            "description": value["description"].strip(),
            "type": difference_type,
            "significance": significance,
            "is_edit_evidence": is_edit,
        }

    @staticmethod
    def _error(message: str) -> Dict[str, str]:
        return {"status": "error", "error": message.strip() or "Image comparison failed."}

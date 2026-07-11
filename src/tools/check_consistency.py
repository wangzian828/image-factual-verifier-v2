# -*- coding: utf-8 -*-
"""Check visual consistency using VLM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.tools.base import BaseTool


CONSISTENCY_PROMPT_TEMPLATE = """\
You are a visual consistency checker for image verification.
Inspect the image for {aspect} consistency and return exactly one JSON object:
{{
  "consistent": true,
  "details": "brief explanation",
  "inconsistencies": [
    {{
      "description": "what is inconsistent",
      "severity": "high|medium|low",
      "location": "where in the image"
    }}
  ]
}}

Consider:
- shadow direction and lighting
- perspective and geometry
- relative scale and proportions
- edges, seams, or compositing artifacts
- physical plausibility

Be conservative. Output JSON only.
"""

CONSISTENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "consistent": {"type": "boolean"},
        "details": {"type": "string", "maxLength": 800},
        "inconsistencies": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "maxLength": 300},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "location": {"type": "string", "maxLength": 200},
                },
            },
        },
    },
}


@dataclass
class CheckConsistencyTool(BaseTool):
    """VLM-based visual consistency checker."""

    name: str = "check_consistency"
    description: str = (
        "Check the visual or physical consistency of the image for signs of manipulation. "
        "Specify an aspect to check: shadow, perspective, scale, lighting, edges, physics, or all."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {
                    "type": "string",
                    "description": "Local path to the image file.",
                },
                "aspect": {
                    "type": "string",
                    "enum": ["shadow", "perspective", "scale", "lighting", "edges", "physics", "all"],
                    "description": "Which aspect of consistency to check.",
                },
            },
            "required": ["image_input", "aspect"],
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
        image_input = params["image_input"]
        aspect = params.get("aspect", "all")

        try:
            client = self._get_client()
            prompt = CONSISTENCY_PROMPT_TEMPLATE.format(aspect=aspect)
            parsed = client.create_image_json(
                system_prompt=prompt,
                user_text=f"Check this image for {aspect} consistency.",
                image_input=image_input,
                max_tokens=1000,
                model_name=self.model_name,
                response_schema=CONSISTENCY_SCHEMA,
            )
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Consistency check failed: {exc}",
            }

        inconsistencies = parsed.get("inconsistencies", [])
        if not isinstance(inconsistencies, list):
            return {
                "status": "error",
                "error": "Consistency check returned invalid inconsistencies data.",
            }
        if not isinstance(parsed.get("consistent"), bool):
            return {
                "status": "error",
                "error": "Consistency check response is missing boolean 'consistent'.",
            }
        details = parsed.get("details")
        if not isinstance(details, str) or not details.strip():
            return {
                "status": "error",
                "error": "Consistency check response is missing explanatory details.",
            }

        return {
            "status": "success",
            "consistent": parsed["consistent"],
            "aspect_checked": aspect,
            "details": details.strip(),
            "inconsistencies": inconsistencies,
        }

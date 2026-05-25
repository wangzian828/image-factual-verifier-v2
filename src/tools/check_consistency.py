# -*- coding: utf-8 -*-
"""Check visual consistency using VLM.

Analyzes shadow directions, perspective, scale relationships,
and other physical consistency aspects of the image.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.tools.base import BaseTool


CONSISTENCY_PROMPT_TEMPLATE = """\
你是视觉一致性检查模块。检查这张图片中的物理/视觉一致性。

检查方面：{aspect}

请仔细分析，输出 JSON：
{{
  "consistent": true/false,
  "details": "具体分析说明",
  "inconsistencies": [
    {{
      "description": "不一致的具体描述",
      "severity": "high|medium|low",
      "location": "图中位置描述"
    }}
  ]
}}

分析要点：
- shadow: 检查所有物体的阴影方向是否一致（同一光源）
- perspective: 检查透视关系是否合理（消失点、近大远小）
- scale: 检查物体间的比例关系是否合理
- lighting: 检查光照方向和强度是否一致
- edges: 检查物体边缘是否有拼接痕迹（锯齿、模糊边界、色差）
- physics: 检查物理规律是否合理（重力、反射、遮挡关系）

只输出 JSON。"""


@dataclass
class CheckConsistencyTool(BaseTool):
    """VLM-based visual consistency checker.

    Checks whether the image has consistent shadows, perspective, scale,
    lighting, and physics — indicators of manipulation or AI generation.
    """

    name: str = "check_consistency"
    description: str = (
        "Check the visual/physical consistency of the image for signs of manipulation. "
        "Specify an aspect to check: 'shadow' (shadow directions), 'perspective' (vanishing points), "
        "'scale' (object proportions), 'lighting' (illumination consistency), "
        "'edges' (splicing artifacts), 'physics' (physical plausibility), or 'all'. "
        "Use when you suspect the image may be composited or AI-generated."
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
    provider: str = "necodex"
    model_name: str = "gpt-5.5"

    def _get_client(self):
        if self.client is None:
            from src.integrations.vlm.factory import build_vlm_client

            self.client = build_vlm_client(
                provider=self.provider, model_name=self.model_name
            )
        return self.client

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check visual consistency of the specified aspect."""
        image_input = params["image_input"]
        aspect = params.get("aspect", "all")

        try:
            client = self._get_client()
            prompt = CONSISTENCY_PROMPT_TEMPLATE.format(aspect=aspect)
            parsed = client.create_image_json(
                system_prompt=prompt,
                user_text=f"请检查这张图片的{aspect}一致性。",
                image_input=image_input,
                max_tokens=1000,
                model_name=self.model_name,
            )
        except Exception as e:
            return {
                "status": "error",
                "error": f"Consistency check failed: {str(e)}",
            }

        inconsistencies = parsed.get("inconsistencies", [])
        if not isinstance(inconsistencies, list):
            inconsistencies = []

        return {
            "status": "success",
            "consistent": bool(parsed.get("consistent", True)),
            "aspect_checked": aspect,
            "details": str(parsed.get("details", "")),
            "inconsistencies": inconsistencies,
        }

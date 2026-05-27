# -*- coding: utf-8 -*-
"""Object counting using VLM with structured prompt.

VLM-based counting for now. P2: integrate GroundingDINO for precise detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.tools.base import BaseTool


COUNT_PROMPT_TEMPLATE = """\
你是物体计数模块。仔细数一下图中指定物体的数量。

要数的目标：{target_object}

计数规则：
1. 仔细逐个数，不要估算
2. 被遮挡但明显存在的也要数
3. 如果目标物体不存在，count 为 0
4. 对于人体部位（手指、腿等），要特别仔细

输出 JSON：
{{
  "target": "{target_object}",
  "count": 数量,
  "confidence": 0.0-1.0,
  "details": "计数过程说明（如：左边3个，右边2个）",
  "locations": ["位置1描述", "位置2描述", ...]
}}

只输出 JSON。"""


@dataclass
class CountObjectsTool(BaseTool):
    """Count specific objects in the image using VLM.

    VLM counting is known to be unreliable (Blink paper), but with
    structured prompting and chain-of-thought it's acceptable for
    small counts. For precise counting, GroundingDINO will be added later.
    """

    name: str = "count_objects"
    description: str = (
        "Count the number of specific objects in the image or a region. "
        "Specify what to count (e.g., 'fingers on the left hand', 'spires on the tower', "
        "'people in the background'). Returns count with confidence and location details. "
        "Note: for counts > 10, accuracy decreases."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {
                    "type": "string",
                    "description": "Local path to the image file.",
                },
                "target_object": {
                    "type": "string",
                    "description": "What to count. Be specific: 'fingers on left hand', not just 'fingers'.",
                },
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional: restrict counting to this region [x1,y1,x2,y2] normalized 0-1.",
                },
            },
            "required": ["image_input", "target_object"],
        }
    )

    client: Optional[Any] = field(default=None, repr=False)
    provider: str = "lmdeploy"
    model_name: str = "/gsdata/home/wza/models/Qwen3-VL-8B-Thinking"

    def _get_client(self):
        if self.client is None:
            from src.integrations.vlm.factory import build_vlm_client

            self.client = build_vlm_client(
                provider=self.provider, model_name=self.model_name
            )
        return self.client

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Count objects in the image."""
        image_input = params["image_input"]
        target_object = params["target_object"]
        bbox = params.get("bbox")

        # If bbox provided, crop first
        actual_image = image_input
        tmp_path = None

        if bbox and len(bbox) == 4:
            try:
                from PIL import Image
                import tempfile
                import os

                img = Image.open(image_input)
                w, h = img.size
                x1 = int(bbox[0] * w)
                y1 = int(bbox[1] * h)
                x2 = int(bbox[2] * w)
                y2 = int(bbox[3] * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                cropped = img.crop((x1, y1, x2, y2))
                tmp_path = os.path.join(
                    tempfile.gettempdir(), f"count_{os.getpid()}_{id(self)}.png"
                )
                cropped.save(tmp_path)
                actual_image = tmp_path
            except Exception:
                pass  # Fall back to full image

        try:
            client = self._get_client()
            prompt = COUNT_PROMPT_TEMPLATE.format(target_object=target_object)
            parsed = client.create_image_json(
                system_prompt=prompt,
                user_text=f"请数一下图中的{target_object}。",
                image_input=actual_image,
                max_tokens=500,
                model_name=self.model_name,
            )
        except Exception as e:
            return {
                "status": "error",
                "error": f"Counting failed: {str(e)}",
            }
        finally:
            if tmp_path:
                try:
                    import os
                    os.remove(tmp_path)
                except OSError:
                    pass

        return {
            "status": "success",
            "target": target_object,
            "count": int(parsed.get("count", 0)),
            "confidence": float(parsed.get("confidence", 0.5)),
            "details": str(parsed.get("details", "")),
            "locations": parsed.get("locations", []),
        }

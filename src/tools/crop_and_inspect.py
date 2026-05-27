# -*- coding: utf-8 -*-
"""Crop and inspect tool: crop a region from the image and analyze it with VLM.

Used in the verification stage to examine specific areas in detail,
guided by entity bboxes from the perception stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.tools.base import BaseTool


INSPECT_PROMPT_TEMPLATE = """\
你正在检查一张图片中的一个裁剪区域。

检查重点：{focus_question}

请仔细观察这个区域，输出 JSON：
{{
  "description": "这个区域的详细描述",
  "findings": ["发现1", "发现2", ...],
  "anomalies": ["异常1", ...],  // 如果没有异常则为空列表
  "answer": "针对检查重点的直接回答"
}}

只输出 JSON。"""


@dataclass
class CropAndInspectTool(BaseTool):
    """Crop a region from the image and perform detailed VLM analysis.

    Requires entity bboxes from the perception stage. Allows the verification
    agent to zoom into specific areas for detailed examination.
    """

    name: str = "crop_and_inspect"
    description: str = (
        "Crop a specific region from the image and analyze it in detail. "
        "Provide a bounding box [x1, y1, x2, y2] (normalized 0-1) and a focus question. "
        "The tool will crop that region and use VLM to examine it closely. "
        "Use when you need to verify details in a specific area of the image."
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
                    "description": "Bounding box [x1, y1, x2, y2] normalized 0-1.",
                },
                "focus_question": {
                    "type": "string",
                    "description": "What to look for in this region. E.g., 'How many fingers does this person have?'",
                },
            },
            "required": ["image_input", "bbox", "focus_question"],
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
        """Crop the region and analyze it."""
        image_path = params["image_input"]
        bbox = params["bbox"]
        focus_question = params["focus_question"]

        # Validate bbox
        if not bbox or len(bbox) != 4:
            return {
                "status": "error",
                "error": "bbox must be [x1, y1, x2, y2] with 4 values normalized 0-1.",
            }

        try:
            from PIL import Image

            img = Image.open(image_path)
            w, h = img.size

            # Convert normalized coords to pixel coords
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)

            # Clamp to image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # Crop
            cropped = img.crop((x1, y1, x2, y2))

            # Save to temp file for VLM
            import tempfile
            import os

            tmp_path = os.path.join(
                tempfile.gettempdir(), f"crop_{os.getpid()}_{id(self)}.png"
            )
            cropped.save(tmp_path)

        except Exception as e:
            return {
                "status": "error",
                "error": f"Crop failed: {str(e)}",
            }

        try:
            client = self._get_client()
            prompt = INSPECT_PROMPT_TEMPLATE.format(focus_question=focus_question)
            parsed = client.create_image_json(
                system_prompt=prompt,
                user_text=f"检查重点：{focus_question}",
                image_input=tmp_path,
                max_tokens=1000,
                model_name=self.model_name,
            )
        except Exception as e:
            return {
                "status": "error",
                "error": f"VLM inspection failed: {str(e)}",
            }
        finally:
            # Clean up temp file
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
        }

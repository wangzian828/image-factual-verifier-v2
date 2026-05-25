# -*- coding: utf-8 -*-
"""Scene perception using VLM with structured output.

Extracts entities, scene type, and description from the image
using a carefully designed prompt that forces structured output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool


PERCEIVE_SCENE_PROMPT = """\
你是图像感知模块。仔细观察这张图片，提取所有可见的实体和场景信息。

输出要求（严格 JSON）：
{
  "entities": [
    {
      "name": "实体名称（具体描述，如'穿蓝色西装的中年男性'）",
      "entity_type": "person|object|building|text|logo|animal|scene_element",
      "bbox": [x1, y1, x2, y2],  // 归一化坐标 0-1，左上角为原点。如果无法确定精确位置则留空 []
      "confidence": 0.9,  // 你对这个实体存在的确信度
      "attributes": {"key": "value"}  // 相关属性，如颜色、大小、状态等
    }
  ],
  "scene_description": "一句话描述整个场景",
  "image_type": "photo|screenshot|document|illustration|meme"
}

规则：
1. 列出所有重要实体（人物、物体、文字、logo、建筑等），最多 15 个
2. 人物要描述外貌特征（衣着、年龄段、性别）
3. 文字内容单独作为 entity_type="text" 的实体
4. logo/品牌标识单独列出
5. bbox 用归一化坐标 [x1, y1, x2, y2]，范围 0-1。不确定就留空 []
6. scene_description 要客观描述，不做判断
7. image_type 判断图片类型

只输出 JSON，不要其他文字。"""


@dataclass
class PerceiveSceneTool(BaseTool):
    """VLM-based scene perception that outputs structured entity list.

    This is the primary perception tool — it uses the VLM's strong semantic
    understanding to identify all entities, their types, and relationships.
    """

    name: str = "perceive_scene"
    description: str = (
        "Observe the image and extract a structured list of all visible entities "
        "(people, objects, text, logos, buildings, animals), their types, approximate "
        "positions, and attributes. Also determines the image type and provides a "
        "scene description. Use this FIRST to understand what's in the image."
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

    # VLM client (injected at construction or built lazily)
    client: Optional[Any] = field(default=None, repr=False)
    provider: str = "necodex"
    model_name: str = "gpt-5.5"

    def _get_client(self):
        """Lazy initialization of VLM client."""
        if self.client is None:
            from src.integrations.vlm.factory import build_vlm_client

            self.client = build_vlm_client(
                provider=self.provider, model_name=self.model_name
            )
        return self.client

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run scene perception on the image."""
        image_input = params["image_input"]

        last_error = None
        for attempt in range(2):  # Retry once on failure
            try:
                client = self._get_client()
                parsed = client.create_image_json(
                    system_prompt=PERCEIVE_SCENE_PROMPT,
                    user_text="请仔细观察这张图片，输出结构化的实体列表和场景信息。",
                    image_input=image_input,
                    max_tokens=2000,
                    model_name=self.model_name,
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt == 0:
                    import time
                    time.sleep(2)

        if last_error is not None:
            return {
                "status": "error",
                "error": f"Scene perception failed: {str(last_error)}",
                "entities": [],
                "scene_description": "",
                "image_type": "unknown",
            }

        # Normalize output
        entities = []
        for ent in parsed.get("entities", []):
            if not isinstance(ent, dict):
                continue
            entities.append({
                "name": str(ent.get("name", "")).strip(),
                "entity_type": str(ent.get("entity_type", "object")).strip(),
                "bbox": ent.get("bbox", []) if isinstance(ent.get("bbox"), list) else [],
                "confidence": float(ent.get("confidence", 0.8)),
                "attributes": ent.get("attributes", {}) if isinstance(ent.get("attributes"), dict) else {},
            })

        return {
            "status": "success",
            "entities": entities[:15],
            "scene_description": str(parsed.get("scene_description", "")).strip(),
            "image_type": str(parsed.get("image_type", "photo")).strip(),
            "total_entities": len(entities[:15]),
        }

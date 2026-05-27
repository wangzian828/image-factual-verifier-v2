from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.integrations.vlm.factory import build_vlm_client
from src.tools.base import BaseTool


OCR_PROMPT = """请读取图像中的可见文字。
要求：
1. 尽量逐字忠实转写
2. 如果没有可见文字，返回空字符串
3. 只输出一个 JSON 对象，格式为：
{
  "text": "识别出的文字",
  "language": "主要语言，未知则写 unknown"
}
"""


@dataclass
class OCRTool(BaseTool):
    """OCR extraction using a configurable vision-language model."""

    client: Optional[Any] = None
    provider: str = "lmdeploy"
    model_name: str = "/gsdata/home/wza/models/Qwen3-VL-8B-Thinking"
    max_tokens: int = 500
    name: str = "ocr"
    description: str = "Read visible text from an input image."
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {"type": "string", "description": "Local path, URL, or data URL of the image."},
            },
            "required": ["image_input"],
        }
    )

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = build_vlm_client(provider=self.provider, model_name=self.model_name)

    def extract(self, image_input: str) -> Dict[str, Any]:
        parsed = self.client.create_image_json(
            system_prompt=OCR_PROMPT,
            user_text="提取这张图像中的所有可见文字。",
            image_input=image_input,
            max_tokens=self.max_tokens,
            model_name=self.model_name,
        )
        return {
            "text": str(parsed.get("text", "")),
            "language": str(parsed.get("language", "unknown")),
        }

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.extract(params["image_input"])

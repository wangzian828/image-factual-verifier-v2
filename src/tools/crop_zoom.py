from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from src.tools.base import BaseTool


@dataclass
class CropZoomTool(BaseTool):
    """Crop a local image region for localized OCR or inspection."""

    output_dir: str = "outputs/crops"
    name: str = "crop_zoom"
    description: str = "Crop a local image region and return the cropped file path plus coordinates."
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {"type": "string", "description": "Local path, URL, or data URL of the image."},
                "region": {
                    "type": "object",
                    "properties": {
                        "x1": {"type": "number"},
                        "y1": {"type": "number"},
                        "x2": {"type": "number"},
                        "y2": {"type": "number"},
                    },
                    "required": ["x1", "y1", "x2", "y2"],
                },
                "pad_ratio": {"type": "number", "description": "Optional padding ratio around the crop box."},
            },
            "required": ["image_input", "region"],
        }
    )

    def crop(self, image_input: str, region: Dict[str, float], pad_ratio: float = 0.0) -> Dict[str, Any]:
        image = self._load_image(image_input)
        width, height = image.size
        x1 = float(region["x1"]) * width
        y1 = float(region["y1"]) * height
        x2 = float(region["x2"]) * width
        y2 = float(region["y2"]) * height

        if pad_ratio:
            pad_x = (x2 - x1) * pad_ratio
            pad_y = (y2 - y1) * pad_ratio
            x1 -= pad_x
            y1 -= pad_y
            x2 += pad_x
            y2 += pad_y

        x1_i = max(0, int(round(min(x1, x2))))
        y1_i = max(0, int(round(min(y1, y2))))
        x2_i = min(width, int(round(max(x1, x2))))
        y2_i = min(height, int(round(max(y1, y2))))

        if x2_i <= x1_i or y2_i <= y1_i:
            raise ValueError("Invalid crop region after clamping.")

        cropped = image.crop((x1_i, y1_i, x2_i, y2_i))
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(self._source_name(image_input)).stem}_{x1_i}_{y1_i}_{x2_i}_{y2_i}.png"
        cropped.save(out_path)

        return {
            "image_path": str(out_path),
            "source_image": self._source_name(image_input),
            "region": {
                "x1": x1_i,
                "y1": y1_i,
                "x2": x2_i,
                "y2": y2_i,
            },
            "normalized_region": {
                "x1": round(x1_i / width, 4),
                "y1": round(y1_i / height, 4),
                "x2": round(x2_i / width, 4),
                "y2": round(y2_i / height, 4),
            },
            "size": {"width": x2_i - x1_i, "height": y2_i - y1_i},
        }

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.crop(
            image_input=params["image_input"],
            region=params["region"],
            pad_ratio=float(params.get("pad_ratio", 0.0) or 0.0),
        )

    @staticmethod
    def _source_name(image_input: str) -> str:
        if image_input.startswith("data:"):
            return "data_image"
        if image_input.startswith("http://") or image_input.startswith("https://"):
            return Path(image_input.split("?")[0]).name or "remote_image"
        return Path(image_input).name

    @staticmethod
    def _load_image(image_input: str) -> Image.Image:
        if image_input.startswith("data:"):
            header, b64_data = image_input.split(",", 1)
            data = base64.b64decode(b64_data)
            return Image.open(BytesIO(data)).convert("RGB")
        if image_input.startswith("http://") or image_input.startswith("https://"):
            from src.tools.vision_utils import image_to_data_url

            data_url = image_to_data_url(image_input)
            _, b64_data = data_url.split(",", 1)
            data = base64.b64decode(b64_data)
            return Image.open(BytesIO(data)).convert("RGB")
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_input}")
        return Image.open(path).convert("RGB")

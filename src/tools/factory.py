from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from src.integrations.browse.jina_reader import JinaReaderClient
from src.integrations.clock.system_clock import SystemClockClient
from src.integrations.search.serper import SerperImageSearchClient, SerperNewsSearchClient, SerperTextSearchClient
from src.integrations.vlm.factory import build_vlm_client
from src.tools.current_time import CurrentTimeTool
from src.tools.crop_and_reverse_search import CropAndReverseSearchTool
from src.tools.crop_zoom import CropZoomTool
from src.tools.image_search import ImageSearchTool
from src.tools.ocr import OCRTool
from src.tools.query_guidelines import QueryGuidelinesTool
from src.tools.reverse_image_search import ReverseImageSearchTool
from src.tools.text_search import TextSearchTool
from src.tools.news_search import NewsSearchTool
from src.tools.visit import VisitTool


def load_tool_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Tool config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_external_tools(
    config_path: str | Path = "configs/tools.yaml",
    *,
    vlm_provider_override: str | None = None,
    vlm_model_override: str | None = None,
) -> Dict[str, object]:
    config = load_tool_config(config_path)

    text_search_cfg = config.get("search", {})
    image_search_cfg = config.get("image_search", {})
    visit_cfg = config.get("visit", {})
    reverse_cfg = config.get("reverse_image_search", {})
    crop_reverse_cfg = config.get("crop_and_reverse_search", {})
    ocr_cfg = config.get("ocr", {})
    vlm_cfg = config.get("vlm", {})

    default_vlm_provider = vlm_provider_override or str(vlm_cfg.get("provider", "lmdeploy"))
    default_vlm_model = vlm_model_override or str(vlm_cfg.get("model_name", "/gsdata/home/wza/models/Qwen3-VL-8B-Thinking"))

    text_search = TextSearchTool(
        client=SerperTextSearchClient(),
        top_k=int(text_search_cfg.get("top_k", 10)),
    )
    news_search = NewsSearchTool(
        client=SerperNewsSearchClient(),
        top_k=int(text_search_cfg.get("top_k", 10)),
    )
    image_search = ImageSearchTool(
        client=SerperImageSearchClient(),
        top_k=int(image_search_cfg.get("top_k", 5)),
    )
    visit_tool = VisitTool(
        client=JinaReaderClient(
            max_chars=int(visit_cfg.get("max_chars", 12000)),
            snippet_chars=int(visit_cfg.get("snippet_chars", 2000)),
        )
    )
    ocr_tool = OCRTool(
        client=build_vlm_client(
            provider=str(ocr_cfg.get("provider", default_vlm_provider)),
            model_name=str(ocr_cfg.get("model_name", default_vlm_model)),
        ),
        provider=str(ocr_cfg.get("provider", default_vlm_provider)),
        model_name=str(ocr_cfg.get("model_name", default_vlm_model)),
    )
    reverse_tool = ReverseImageSearchTool(
        vlm_client=build_vlm_client(
            provider=str(reverse_cfg.get("provider", default_vlm_provider)),
            model_name=str(reverse_cfg.get("model_name", default_vlm_model)),
        ),
        image_search_client=SerperImageSearchClient(),
        provider=str(reverse_cfg.get("provider", default_vlm_provider)),
        model_name=str(reverse_cfg.get("model_name", default_vlm_model)),
        qwen_model_name=str(reverse_cfg.get("model_name", default_vlm_model)),
        top_k=int(reverse_cfg.get("top_k", 5)),
    )
    current_time = CurrentTimeTool(client=SystemClockClient())
    crop_zoom = CropZoomTool()
    crop_and_reverse_search = CropAndReverseSearchTool(
        top_k=int(crop_reverse_cfg.get("top_k", reverse_cfg.get("top_k", 5))),
    )
    query_guidelines = QueryGuidelinesTool()

    return {
        "current_time": current_time,
        "crop_zoom": crop_zoom,
        "crop_and_reverse_search": crop_and_reverse_search,
        "text_search": text_search,
        "news_search": news_search,
        "image_search": image_search,
        "visit": visit_tool,
        "ocr": ocr_tool,
        "reverse_image_search": reverse_tool,
        "query_guidelines": query_guidelines,
    }

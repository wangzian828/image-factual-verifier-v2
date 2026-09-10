# -*- coding: utf-8 -*-
"""Tool registry for the active orchestrator pipeline."""
from __future__ import annotations

import os
import inspect
from typing import Any, Dict, Optional, Tuple

from src.integrations.vlm.factory import build_vlm_client
from src.orchestrator.tool_health import ToolHealth
from src.tools.base import BaseTool


REQUIRED_TOOLS = (
    "perceive_scene",
    "focused_visual_inspection",
    "ocr_with_position",
    "text_search",
    "text_image_search",
    "visit",
    "reverse_image_search",
)
def build_all_tools_with_health(
    vlm_provider: str = "gemini",
    vlm_model: str = "gemini-3.7-flash",
    vlm_wire_api: Optional[str] = None,
    vlm_base_url: Optional[str] = None,
) -> Tuple[Dict[str, BaseTool], Dict[str, ToolHealth]]:
    tools: Dict[str, BaseTool] = {}
    health: Dict[str, ToolHealth] = {}

    from src.integrations.browse.jina_reader import JinaReaderClient
    from src.integrations.search.serper import (
        SerperImageSearchClient,
        SerperLensSearchClient,
    )
    from src.integrations.search.visual_search import VisualReverseSearchClient

    request_timeout = float(
        os.getenv("VLM_TOOL_REQUEST_TIMEOUT_SECONDS", "90")
    )
    request_max_retries = int(
        os.getenv("VLM_TOOL_REQUEST_MAX_RETRIES", "3")
    )
    def sync_vlm_client() -> Any:
        client = build_vlm_client(
            provider=vlm_provider,
            model_name=vlm_model,
            wire_api=vlm_wire_api,
            base_url=vlm_base_url,
            timeout=request_timeout,
            max_retries=request_max_retries,
        )
        _validate_structured_vision_client(
            client,
            provider=vlm_provider,
        )
        return client

    shared_vlm_client: Any = None
    shared_vlm_error = ""
    try:
        shared_vlm_client = sync_vlm_client()
    except Exception as exc:
        shared_vlm_error = f"{type(exc).__name__}: {exc}"

    def shared_sync_vlm_client() -> Any:
        if shared_vlm_client is not None:
            return shared_vlm_client
        raise RuntimeError(shared_vlm_error or "shared VLM client is unavailable")

    # These clients are stateless at the tool boundary and already protect
    # mutable caches/sessions internally. Sharing them lets visit and
    # crop_and_search reuse page content, HTTP keep-alive sessions, and the
    # persistent Gemini extraction transport without serializing page visits.
    shared_browse_client = JinaReaderClient()
    shared_lens_client = SerperLensSearchClient()
    shared_image_search_client = SerperImageSearchClient()
    shared_visual_search_client = VisualReverseSearchClient(
        serper_lens_client=shared_lens_client,
    )

    def register(name: str, builder) -> None:
        try:
            tool = builder()
            if tool is None:
                raise RuntimeError("builder returned None")
            tools[name] = tool
            health[name] = ToolHealth(available=True)
        except Exception as exc:
            health[name] = ToolHealth(available=False, error=f"{type(exc).__name__}: {exc}")

    register(
        "perceive_scene",
        lambda: __import__("src.tools.perceive_scene", fromlist=["PerceiveSceneTool"]).PerceiveSceneTool(
            client=shared_sync_vlm_client(),
            provider=vlm_provider,
            model_name=vlm_model,
        ),
    )
    register(
        "ocr_with_position",
        lambda: __import__("src.tools.ocr_with_position", fromlist=["OCRWithPositionTool"]).OCRWithPositionTool(),
    )
    register(
        "current_time",
        lambda: __import__("src.tools.current_time", fromlist=["CurrentTimeTool"]).CurrentTimeTool(),
    )
    register(
        "recall_evidence",
        lambda: __import__(
            "src.tools.context_memory",
            fromlist=["RecallEvidenceTool"],
        ).RecallEvidenceTool(),
    )
    register(
        "read_evidence",
        lambda: __import__(
            "src.tools.context_memory",
            fromlist=["ReadEvidenceTool"],
        ).ReadEvidenceTool(),
    )
    register(
        "crop_and_inspect",
        lambda: __import__("src.tools.crop_and_inspect", fromlist=["CropAndInspectTool"]).CropAndInspectTool(
            client=shared_sync_vlm_client(),
            provider=vlm_provider,
            model_name=vlm_model,
        ),
    )
    register(
        "focused_visual_inspection",
        lambda: __import__(
            "src.tools.focused_visual_inspection",
            fromlist=["FocusedVisualInspectionTool"],
        ).FocusedVisualInspectionTool(
            client=shared_sync_vlm_client(),
            provider=vlm_provider,
            model_name=vlm_model,
        ),
    )
    register(
        "crop_and_search",
        lambda: __import__("src.tools.crop_and_search", fromlist=["CropAndSearchTool"]).CropAndSearchTool(
            vlm_client=shared_sync_vlm_client(),
            lens_client=shared_lens_client,
            image_search_client=shared_image_search_client,
            browse_client=shared_browse_client,
            visual_search_client=shared_visual_search_client,
            provider=vlm_provider,
            model_name=vlm_model,
        ),
    )
    register(
        "count_objects",
        lambda: __import__("src.tools.count_objects", fromlist=["CountObjectsTool"]).CountObjectsTool(
            client=shared_sync_vlm_client(),
            provider=vlm_provider,
            model_name=vlm_model,
        ),
    )
    register(
        "check_consistency",
        lambda: __import__("src.tools.check_consistency", fromlist=["CheckConsistencyTool"]).CheckConsistencyTool(
            client=shared_sync_vlm_client(),
            provider=vlm_provider,
            model_name=vlm_model,
        ),
    )
    register(
        "text_search",
        lambda: _build_text_search_tool(),
    )
    register(
        "text_image_search",
        lambda: __import__(
            "src.tools.text_image_search",
            fromlist=["TextImageSearchTool"],
        ).TextImageSearchTool(client=shared_image_search_client),
    )
    register(
        "reverse_image_search",
        lambda: __import__("src.tools.reverse_image_search", fromlist=["ReverseImageSearchTool"]).ReverseImageSearchTool(
            vlm_client=shared_sync_vlm_client(),
            image_search_client=shared_image_search_client,
            lens_client=shared_lens_client,
            visual_search_client=shared_visual_search_client,
            provider=vlm_provider,
            model_name=vlm_model,
        ),
    )
    register(
        "visit",
        lambda: __import__("src.tools.visit", fromlist=["VisitTool"]).VisitTool(
            client=shared_browse_client,
        ),
    )
    register(
        "analyze_visual_anomalies",
        lambda: __import__("src.tools.visual_anomaly", fromlist=["VisualAnomalyTool"]).VisualAnomalyTool(
            client=shared_sync_vlm_client(),
            provider=vlm_provider,
            model_name=vlm_model,
        ),
    )
    register(
        "compare_with_reference",
        lambda: __import__("src.tools.compare_reference", fromlist=["CompareWithReferenceTool"]).CompareWithReferenceTool(
            client=shared_sync_vlm_client(),
            provider=vlm_provider,
            model_name=vlm_model,
        ),
    )

    return tools, health


def _build_text_search_tool() -> BaseTool:
    from src.tools.text_search import TextSearchTool

    reranker = None
    if str(
        os.getenv("JINA_API_KEY")
        or os.getenv("JINA_API_KEYS")
        or ""
    ).strip():
        from src.integrations.search.jina_reranker import JinaRerankerClient

        reranker = JinaRerankerClient()
    return TextSearchTool(candidate_reranker=reranker)


def _validate_structured_vision_client(
    client: Any,
    *,
    provider: str,
) -> None:
    method = getattr(client, "create_image_json", None)
    if not callable(method):
        raise RuntimeError(
            f"{provider} vision client does not implement create_image_json."
        )
    parameters = inspect.signature(method).parameters
    required = {
        "system_prompt",
        "user_text",
        "image_input",
        "max_tokens",
        "response_schema",
    }
    missing = sorted(required - set(parameters))
    if missing:
        raise RuntimeError(
            f"{provider} vision client lacks structured-output parameters: "
            + ", ".join(missing)
        )
    multi_view = getattr(client, "create_images_json", None)
    if not callable(multi_view):
        raise RuntimeError(
            f"{provider} vision client does not implement create_images_json."
        )
    multi_parameters = inspect.signature(multi_view).parameters
    multi_required = {
        "system_prompt",
        "user_text",
        "image_inputs",
        "max_tokens",
        "response_schema",
    }
    multi_missing = sorted(multi_required - set(multi_parameters))
    if multi_missing:
        raise RuntimeError(
            f"{provider} vision client lacks multi-view structured-output "
            "parameters: "
            + ", ".join(multi_missing)
        )
    if not str(getattr(client, "api_key", "") or "").strip():
        env_name = (
            "QWEN_API_KEY or DASHSCOPE_API_KEY"
            if str(provider).strip().lower() == "qwen"
            else f"{str(provider).upper()} API key"
        )
        raise RuntimeError(f"{env_name} is required for structured vision tools.")

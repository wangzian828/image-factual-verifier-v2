# -*- coding: utf-8 -*-
"""Stage-aware tool registry: builds tool subsets for each pipeline stage.

Each stage only sees its designated tools. Dependencies between tools
are checked and friendly errors returned when prerequisites aren't met.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool


# Stage → tool names mapping
STAGE_TOOLS: Dict[str, List[str]] = {
    "perception": [
        "perceive_scene",
        "ocr_with_position",
        "face_detect",
    ],
    "verification": [
        "crop_and_inspect",
        "count_objects",
        "text_search",
        "news_search",
        "reverse_image_search",
        "visit",
        "compare_with_reference",
        "check_consistency",
        "analyze_visual_anomalies",
        "verify_face_identity",
    ],
}

# Tools that have no stage (pure reasoning stages: planning, judgment)
# These stages don't use tools.


def build_stage_tools(
    stage_name: str,
    all_tools: Dict[str, BaseTool],
) -> List[BaseTool]:
    """Build the tool subset for a given stage.

    Args:
        stage_name: "perception" or "verification"
        all_tools: Dict of all available tool instances keyed by name.

    Returns:
        List of BaseTool instances for this stage.
    """
    tool_names = STAGE_TOOLS.get(stage_name, [])
    tools = []
    for name in tool_names:
        if name in all_tools:
            tools.append(all_tools[name])
    return tools


def build_all_tools(
    vlm_provider: str = "lmdeploy",
    vlm_model: str = "/gsdata/home/wza/models/Qwen3-VL-8B-Thinking",
) -> Dict[str, BaseTool]:
    """Build all tool instances for the pipeline.

    Lazily imports and constructs each tool. Tools that require
    external services (PaddleOCR, InsightFace) are initialized
    lazily on first call.

    Args:
        vlm_provider: Provider for VLM-based tools.
        vlm_model: Model name for VLM-based tools.

    Returns:
        Dict of tool_name → BaseTool instance.
    """
    tools: Dict[str, BaseTool] = {}

    # Build a shared LLMBackend for tools that need async VLM calls
    from src.orchestrator.llm_backend import APIBackend
    vlm_backend = APIBackend(
        provider=vlm_provider,
        model_name=vlm_model,
        temperature=0.0,
        max_tokens=4096,
    )

    # --- Perception tools ---
    from src.tools.perceive_scene import PerceiveSceneTool
    tools["perceive_scene"] = PerceiveSceneTool(
        provider=vlm_provider, model_name=vlm_model
    )

    from src.tools.ocr_with_position import OCRWithPositionTool
    tools["ocr_with_position"] = OCRWithPositionTool()

    from src.tools.face_detect import FaceDetectTool
    face_detect = FaceDetectTool()
    tools["face_detect"] = face_detect

    # --- Verification tools ---
    from src.tools.crop_and_inspect import CropAndInspectTool
    tools["crop_and_inspect"] = CropAndInspectTool(
        provider=vlm_provider, model_name=vlm_model
    )

    from src.tools.count_objects import CountObjectsTool
    tools["count_objects"] = CountObjectsTool(
        provider=vlm_provider, model_name=vlm_model
    )

    from src.tools.check_consistency import CheckConsistencyTool
    tools["check_consistency"] = CheckConsistencyTool(
        provider=vlm_provider, model_name=vlm_model
    )

    from src.tools.verify_face_identity import VerifyFaceIdentityTool
    tools["verify_face_identity"] = VerifyFaceIdentityTool(
        face_detect_tool=face_detect
    )

    # External search tools (require API keys)
    try:
        from src.tools.text_search import TextSearchTool
        tools["text_search"] = TextSearchTool()
    except Exception:
        pass

    try:
        from src.tools.news_search import NewsSearchTool
        tools["news_search"] = NewsSearchTool()
    except Exception:
        pass

    try:
        from src.tools.reverse_image_search import ReverseImageSearchTool
        tools["reverse_image_search"] = ReverseImageSearchTool()
    except Exception:
        pass

    try:
        from src.tools.visit import VisitTool
        tools["visit"] = VisitTool()
    except Exception:
        pass

    # Tools that need vlm_backend (async LLMBackend) + image_path (set per-run)
    try:
        from src.tools.visual_anomaly import VisualAnomalyTool
        tools["analyze_visual_anomalies"] = VisualAnomalyTool(
            vlm_backend=vlm_backend,
        )
    except Exception:
        pass

    try:
        from src.tools.compare_reference import CompareWithReferenceTool
        tools["compare_with_reference"] = CompareWithReferenceTool(
            vlm_backend=vlm_backend,
        )
    except Exception:
        pass

    return tools

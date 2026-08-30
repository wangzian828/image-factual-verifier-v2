from __future__ import annotations

from typing import Any, Dict

from src.orchestrator.unified_prompts import UNIFIED_REACT_SYSTEM_PROMPT
from src.orchestrator.unified_react import UnifiedReactToolAdapter
from src.eval.private_gold_metrics import (
    agent_private_gold_category,
    private_gold_category,
)
from src.tools.base import BaseTool
from src.tools.compare_reference import CompareWithReferenceTool
from src.tools.crop_and_search import CropAndSearchTool
from src.tools.reverse_image_search import ReverseImageSearchTool


class _RecordingCompareTool(BaseTool):
    name = "compare_with_reference"
    description = "fixture compare"
    parameters = {
        "type": "object",
        "properties": {
            "reference_url": {"type": "string"},
            "source_page_url": {"type": "string"},
        },
        "required": ["reference_url"],
    }

    def __init__(self) -> None:
        self.received: Dict[str, Any] = {}

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.received = dict(params)
        return {"status": "success"}


def test_compare_adapter_injects_paired_source_page_only_when_missing() -> None:
    delegate = _RecordingCompareTool()
    adapter = UnifiedReactToolAdapter(
        delegate=delegate,
        reference_source_page_urls={
            "https://cdn.example.test/opaque-image?id=7": (
                "https://publisher.example.test/story"
            )
        },
    )

    adapter.call({"reference_url": "https://cdn.example.test/opaque-image?id=7"})
    assert delegate.received["source_page_url"] == "https://publisher.example.test/story"

    adapter.call(
        {
            "reference_url": "https://cdn.example.test/opaque-image?id=7",
            "source_page_url": "https://explicit.example.test/page",
        }
    )
    assert delegate.received["source_page_url"] == "https://explicit.example.test/page"


def test_extensionless_provider_image_candidates_are_kept_for_content_type_validation() -> None:
    opaque = "https://cdn.example.test/image?resize=1200"
    assert ReverseImageSearchTool._is_image_url(opaque) is True
    assert CropAndSearchTool._is_image_candidate_url(opaque) is True
    assert ReverseImageSearchTool._is_image_url("ftp://cdn.example.test/image") is False
    assert CropAndSearchTool._is_image_candidate_url("not-a-url") is False


def test_reference_download_subcalls_keep_stage_and_failure_detail() -> None:
    subcalls = CompareWithReferenceTool._download_subcalls(
        {
            "download_diagnostics": [
                {
                    "stage": "direct",
                    "outcome": "http_error",
                    "http_status": 403,
                },
                {
                    "stage": "source_page",
                    "outcome": "page_images_extracted",
                },
                {
                    "stage": "page_image",
                    "outcome": "image_downloaded",
                },
            ]
        }
    )

    assert [item["stage"] for item in subcalls] == [
        "direct",
        "source_page",
        "page_image",
    ]
    assert subcalls[0]["http_status"] == 403
    assert subcalls[0]["status"] == "error"
    assert subcalls[-1]["status"] == "success"


def test_react_prompt_marks_reverse_results_as_unverified_candidates() -> None:
    assert "未验证候选" in UNIFIED_REACT_SYSTEM_PROMPT
    assert "每次搜索都要回答一个具体问题" in UNIFIED_REACT_SYSTEM_PROMPT


def test_agent_primary_category_uses_evidence_grounding_not_legacy_quality_bucket() -> None:
    row = {
        "status": "completed",
        "private_gold_auditable": True,
        "verdict_matches_gold": True,
        "quality_bucket": "rejected",
        "reason_quality": "decisive_and_grounded",
    }
    assert agent_private_gold_category(row) == "correct_point_with_strong_evidence"

    row["quality_bucket"] = "strong"
    row["reason_quality"] = "artifact_based"
    assert agent_private_gold_category(row) == "correct_verdict_insufficient_evidence"
    assert private_gold_category(row) == "correct_point_with_strong_evidence"

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.trajectory.reference_chain import (
    SemanticMatchDecision,
    score_reference_chain_trace,
)
from test_trajectory_export import _trace


def _gold_for_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    investigation = trace["state"]["investigation_state"]
    runtime_fact = investigation["facts"][0]
    return {
        "case_id": trace["image_id"],
        "factual_status": "supported",
        "decisive_facts": [
            {
                "fact_id": "rf-ship",
                "kind": runtime_fact["kind"],
                "statement": runtime_fact["statement"],
                "expected_status": "supported",
                "visual_anchor": {
                    "type": "scene",
                    "description": "The marked vessel and surrounding water.",
                },
                "acceptable_evidence": [
                    {
                        "source_id": "source-noaa",
                        "canonical_url": (
                            "https://www.noaa.gov/news/example-source-page"
                        ),
                        "exact_span": (
                            "The NOAA research vessel is shown on the water."
                        ),
                        "stance": "support",
                        "source_family": "noaa",
                        "artifact_sha256": "a" * 64,
                    }
                ],
            }
        ],
    }


def test_semantic_recovery_accepts_same_source_capture_without_exact_snapshot(
    tmp_path: Path,
) -> None:
    trace = _trace(tmp_path)
    metrics = asyncio.run(
        score_reference_chain_trace(trace, _gold_for_trace(trace))
    )

    assert metrics["metrics"] == {
        "fact_recovery_recall": 1.0,
        "chain_recovery_recall": 1.0,
        "evidence_recovery_recall": 1.0,
        "basis_reference_precision": 1.0,
    }
    match = metrics["reference_facts"][0]["evidence_matches"][0]
    assert match["method"] == "same_source_same_capture"
    assert metrics["semantic_matcher"]["enabled"] is False


def test_semantic_recovery_accepts_same_source_page_version(
    tmp_path: Path,
) -> None:
    trace = _trace(tmp_path)
    investigation = trace["state"]["investigation_state"]
    runtime_fact = investigation["facts"][0]
    runtime_fact["status"] = "refuted"
    evidence = investigation["evidence"][0]
    evidence.update(
        {
            "evidence_kind": "web_span",
            "source_url": "https://www.nasa.gov/image-article/new-page-version/",
            "source_family": "domain:nasa.gov",
            "source_class": "official",
            "exact_text": (
                "NASA states that the event occurred at Johnson Space Center. "
                "The crew participated in the Apollo 14 Moon Tree dedication. "
                "The page also explains the tree species, seed history, forest "
                "service germination program, educational use, ceremony date, "
                "crew visit schedule, and later distribution of seedlings."
            ),
            "stance": "refute",
            "quality": "strong",
            "claim_binding": "source_assertion",
            "same_subject_or_scene": None,
            "same_capture_or_near_duplicate": None,
            "likely_different_original_capture": None,
        }
    )
    gold = _gold_for_trace(trace)
    gold_fact = gold["decisive_facts"][0]
    gold_fact["expected_status"] = "refuted"
    gold_fact["acceptable_evidence"][0].update(
        {
            "canonical_url": "https://www.nasa.gov/image-detail/old-page/",
            "exact_span": (
                "The event occurred at Johnson Space Center, where the crew "
                "participated in the Apollo 14 Moon Tree dedication."
            ),
            "stance": "refute",
            "source_family": "nasa",
        }
    )

    metrics = asyncio.run(score_reference_chain_trace(trace, gold))

    assert metrics["metrics"]["evidence_recovery_recall"] == 1.0
    assert metrics["metrics"]["chain_recovery_recall"] == 1.0
    assert metrics["reference_facts"][0]["evidence_matches"][0]["method"] == (
        "same_source_text_equivalent"
    )


def test_off_chain_evidence_only_hurts_when_added_to_basis(
    tmp_path: Path,
) -> None:
    trace = _trace(tmp_path)
    investigation = trace["state"]["investigation_state"]
    runtime_fact = investigation["facts"][0]
    task = investigation["tasks"][0]
    extra_evidence = copy.deepcopy(investigation["evidence"][0])
    extra_evidence.update(
        {
            "evidence_id": "evidence-off-chain",
            "source_url": "https://example.com/generic-vessel-page",
            "source_family": "domain:example.com",
            "source_class": "news",
            "exact_text": "A generic page about research vessels.",
            "artifact_sha256": "b" * 64,
            "claim_binding": "source_assertion",
            "same_subject_or_scene": None,
            "same_capture_or_near_duplicate": None,
            "confidence": 0.8,
        }
    )
    investigation["evidence"].append(extra_evidence)
    investigation["findings"].append(
        {
            "finding_id": "finding-off-chain",
            "task_id": task["task_id"],
            "fact_ids": [runtime_fact["fact_id"]],
            "statement": "A generic page discusses research vessels.",
            "stance": "support",
            "evidence_ids": ["evidence-off-chain"],
            "source_family_ids": ["domain:example.com"],
            "quality": "contextual",
        }
    )
    gold = _gold_for_trace(trace)

    without_basis_pollution = asyncio.run(
        score_reference_chain_trace(trace, gold)
    )
    assert (
        without_basis_pollution["metrics"]["basis_reference_precision"] == 1.0
    )
    assert (
        without_basis_pollution["off_reference_chain_basis_evidence_ids"] == []
    )

    trace["verdict_basis"]["evidence_ids"].append("evidence-off-chain")
    with_basis_pollution = asyncio.run(score_reference_chain_trace(trace, gold))
    assert with_basis_pollution["metrics"]["basis_reference_precision"] == 0.5
    assert with_basis_pollution[
        "off_reference_chain_basis_evidence_ids"
    ] == ["evidence-off-chain"]


class _AcceptingMatcher:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {"provider": "fixture", "model": "fixture", "calls": self.calls}

    async def match(
        self,
        *,
        gold_fact: Mapping[str, Any],
        runtime_fact: Mapping[str, Any],
        evidence: Mapping[str, Any],
        references: Sequence[Mapping[str, Any]],
    ) -> SemanticMatchDecision:
        self.calls += 1
        assert gold_fact["fact_id"] == "rf-ship"
        assert runtime_fact["status"] == "supported"
        assert evidence["evidence_id"]
        assert len(references) == 1
        return SemanticMatchDecision(
            match=True,
            confidence=0.92,
            reference_index=0,
            reason="The alternate official evidence recovers the same vessel claim.",
        )


def test_llm_matcher_only_handles_unresolved_qualified_edge(
    tmp_path: Path,
) -> None:
    trace = _trace(tmp_path)
    evidence = trace["state"]["investigation_state"]["evidence"][0]
    evidence.update(
        {
            "source_url": "https://catalog.archives.gov/alternate.jpg",
            "source_family": "domain:archives.gov",
            "exact_text": (
                "This official catalog image is the accepted NOAA vessel capture."
            ),
        }
    )
    matcher = _AcceptingMatcher()

    metrics = asyncio.run(
        score_reference_chain_trace(
            trace,
            _gold_for_trace(trace),
            semantic_matcher=matcher,
        )
    )

    assert matcher.calls == 1
    assert metrics["metrics"]["evidence_recovery_recall"] == 1.0
    assert metrics["metrics"]["chain_recovery_recall"] == 1.0
    assert metrics["metrics"]["basis_reference_precision"] == 1.0
    assert metrics["reference_facts"][0]["evidence_matches"][0]["method"] == (
        "llm_semantic"
    )

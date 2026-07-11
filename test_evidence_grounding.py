from __future__ import annotations

import json

from src.orchestrator.ledger import build_verification_case, compile_runtime_ledgers
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.stage_runner import StageStep
from src.orchestrator.state import (
    EvidenceItem,
    InvestigationQuestion,
    VerificationPlan,
    VerificationResult,
    VisualAnomaly,
)


def _orchestrator() -> Orchestrator:
    return object.__new__(Orchestrator)


def _plan() -> VerificationPlan:
    return VerificationPlan(
        questions=[
            InvestigationQuestion(
                question_id="q0",
                question="Did Reuters publish the flood image?",
                related_entities=["Reuters", "flood"],
                priority=1,
            )
        ]
    )


def _successful_step() -> StageStep:
    return StageStep(
        round=1,
        stage_name="verification",
        action_type="tool_call",
        tool_name="visit",
        tool_args={"__question_id": "q0"},
        tool_result=(
            '{"status":"success","selected_url":"https://reuters.example/flood",'
            '"evidence":"Reuters published the flood image on 10 July.",'
            '"summary":"Reuters published the flood image on 10 July.",'
            '"relevance":"high","stance":"support",'
            '"artifact_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
            '"evidence_span":{"start":0,"end":45},'
            '"retrieved_at":"2026-07-11T00:00:00+00:00",'
            '"injection_flags":[],"directness":"direct","evidence_eligible":true}'
        ),
        metadata={"tool_success": True, "function_call_id": "call-visit-1"},
    )


def _evidence(**updates) -> EvidenceItem:
    values = {
        "function_call_id": "call-visit-1",
        "source": "https://reuters.example/flood",
        "summary": "Reuters published the flood image on 10 July.",
        "raw_excerpt": "Reuters published the flood image on 10 July.",
        "direction": "supports",
        "quality": "strong",
        "tool_used": "visit",
        "related_question": "q0",
    }
    values.update(updates)
    return EvidenceItem(**values)


def test_evidence_requires_exact_successful_function_call_id() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    evidence = _evidence(function_call_id="call-other")

    assert orchestrator._evidence_item_is_grounded(evidence, [step]) is False


def test_evidence_rejects_fabricated_excerpt_even_with_real_call_id() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    evidence = _evidence(raw_excerpt="Reuters confirmed this was an AI-generated hoax.")

    assert orchestrator._evidence_item_is_grounded(evidence, [step]) is False


def test_coverage_rejects_real_but_question_irrelevant_excerpt() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    step.tool_result = (
        '{"status":"success","selected_url":"https://reuters.example/weather",'
        '"evidence":"Temperatures reached 30 degrees on 10 July.",'
        '"summary":"Temperatures reached 30 degrees on 10 July."}'
    )
    evidence = _evidence(
        source="https://reuters.example/weather",
        summary="Temperatures reached 30 degrees on 10 July.",
        raw_excerpt="Temperatures reached 30 degrees on 10 July.",
    )
    parsed = VerificationResult(evidence=[evidence])

    accepted, reason = orchestrator._validate_verification_output(parsed, [step], _plan())

    assert accepted is False
    assert "priority questions lack grounded evidence" in reason


def test_model_cannot_insert_anomaly_without_anomaly_tool_result() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    anomaly = VisualAnomaly(
        function_call_id="call-visit-1",
        related_question="q0",
        name="Fabricated anomaly",
        region="center",
        phenomenon="An invented visual defect.",
        reasoning="This text was not returned by a visual tool.",
        severity=90,
        type="manipulation",
        entities_involved=["image"],
    )
    parsed = VerificationResult(evidence=[_evidence()], visual_anomalies=[anomaly])

    accepted, reason = orchestrator._validate_verification_output(parsed, [step], _plan())

    assert accepted is False
    assert "visual anomalies must copy" in reason


def test_canonicalization_does_not_trust_model_summary_or_direction() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    model_item = _evidence(
        summary="The image is definitely fake.",
        direction="refutes",
    )

    canonical = orchestrator._canonicalize_model_evidence(model_item, [step], _plan())

    assert canonical is not None
    assert canonical.summary == canonical.raw_excerpt
    assert canonical.direction == "supports"
    assert canonical.function_call_id == "call-visit-1"


def test_canonicalization_uses_refuting_tool_stance_not_model_direction() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    step.tool_result = step.tool_result.replace('"stance":"support"', '"stance":"refute"')
    model_item = _evidence(direction="supports")

    canonical = orchestrator._canonicalize_model_evidence(model_item, [step], _plan())

    assert canonical is not None
    assert canonical.direction == "refutes"


def test_nested_text_search_stance_is_bound_to_exact_excerpt() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    step.tool_name = "text_search"
    step.tool_result = (
        '{"status":"success","queries":['
        '{"query":"flood","selected_url":"https://reuters.example/flood",'
        '"evidence":"Reuters published the flood image on 10 July.",'
        '"summary":"Reuters published the flood image on 10 July.",'
        '"relevance":"high","stance":"support",'
        '"artifact_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"evidence_span":{"start":0,"end":45},'
        '"retrieved_at":"2026-07-11T00:00:00+00:00",'
        '"injection_flags":[],"directness":"direct","evidence_eligible":true},'
        '{"query":"hoax","selected_url":"https://example.test/hoax",'
        '"evidence":"An unrelated page calls another image a hoax.",'
        '"summary":"Unrelated claim.","relevance":"low","stance":"refute",'
        '"artifact_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
        '"evidence_span":{"start":0,"end":45},'
        '"retrieved_at":"2026-07-11T00:00:00+00:00",'
        '"injection_flags":[],"directness":"direct","evidence_eligible":true}]}'
    )
    model_item = _evidence(tool_used="text_search", direction="refutes")

    canonical = orchestrator._canonicalize_model_evidence(model_item, [step], _plan())

    assert canonical is not None
    assert canonical.direction == "supports"


def test_browse_evidence_without_explicit_stance_is_rejected() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    step.tool_result = step.tool_result.replace(',"relevance":"high","stance":"support"', "")

    assert orchestrator._canonicalize_model_evidence(_evidence(), [step], _plan()) is None


def test_duplicate_search_candidates_merge_before_ledger_insertion(tmp_path) -> None:
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"bounded-ledger-fixture")
    step = StageStep(
        round=1,
        stage_name="verification",
        action_type="tool_call",
        tool_name="reverse_image_search",
        tool_args={"__question_id": "q0"},
        tool_result=json.dumps(
            {
                "status": "success",
                "lens_results": [
                    {
                        "url": "https://example.test/reference",
                        "title": "Reference image",
                        "snippet": "Short snippet.",
                    }
                ],
                "semantic_results": [
                    {
                        "url": "https://example.test/reference",
                        "title": "Reference image",
                        "snippet": "A longer and more informative reference snippet.",
                    }
                ],
            }
        ),
        metadata={"tool_success": True, "function_call_id": "call-reverse-1"},
    )

    ledgers = compile_runtime_ledgers(
        build_verification_case(
            str(image_path),
            user_claim="Does the image match the reference?",
        ),
        _plan(),
        VerificationResult(),
        [step],
    )

    assert len(ledgers.discoveries) == 1
    assert ledgers.discoveries[0].snippet == (
        "A longer and more informative reference snippet."
    )


def test_visual_ledger_compilation_is_idempotent(tmp_path) -> None:
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"idempotent-visual-fixture")
    case = build_verification_case(
        str(image_path),
        user_claim="Does the image contain a reference object?",
    )
    steps = [
        StageStep(
            round=index,
            stage_name="verification",
            action_type="tool_call",
            tool_name="crop_and_inspect",
            tool_args={"__question_id": "q0", "bbox": bbox},
            tool_result=json.dumps({"status": "success", "summary": excerpt}),
            metadata={
                "tool_success": True,
                "function_call_id": call_id,
                "observed_at": observed_at,
            },
        )
        for index, (call_id, bbox, excerpt, observed_at) in enumerate(
            [
                (
                    "call-visual-1",
                    [0.1, 0.1, 0.5, 0.5],
                    "The first inspected region contains a reference object.",
                    "2026-07-11T08:00:00+00:00",
                ),
                (
                    "call-visual-2",
                    [0.5, 0.5, 0.9, 0.9],
                    "The second inspected region contains a matching object.",
                    "2026-07-11T08:01:00+00:00",
                ),
            ],
            start=1,
        )
    ]
    result = VerificationResult(
        evidence=[
            EvidenceItem(
                function_call_id=step.metadata["function_call_id"],
                source="",
                summary=json.loads(step.tool_result)["summary"],
                raw_excerpt=json.loads(step.tool_result)["summary"],
                direction="supports",
                quality="moderate",
                tool_used="crop_and_inspect",
                related_question="q0",
            )
            for step in steps
        ]
    )

    first = compile_runtime_ledgers(case, _plan(), result, steps)
    second = compile_runtime_ledgers(case, _plan(), result, steps)

    assert first == second
    assert len(first.sources) == 1
    assert first.sources[0].retrieved_at == case.created_at
    assert [item.retrieved_at for item in first.evidence] == [
        "2026-07-11T08:00:00+00:00",
        "2026-07-11T08:01:00+00:00",
    ]


def test_current_time_is_recorded_as_runtime_anchor(tmp_path) -> None:
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"runtime-anchor-fixture")
    case = build_verification_case(
        str(image_path),
        user_claim="Was this claim posted before today?",
    )
    excerpt = "2026-07-11T16:00:00+08:00"
    step = StageStep(
        round=1,
        stage_name="verification",
        action_type="tool_call",
        tool_name="current_time",
        tool_args={"__question_id": "q0"},
        tool_result=json.dumps({"status": "success", "datetime": excerpt}),
        metadata={
            "tool_success": True,
            "function_call_id": "call-time-1",
            "observed_at": "2026-07-11T08:00:01+00:00",
        },
    )
    result = VerificationResult(
        evidence=[
            EvidenceItem(
                function_call_id="call-time-1",
                source="current_time",
                summary=excerpt,
                raw_excerpt=excerpt,
                direction="neutral",
                quality="strong",
                tool_used="current_time",
                related_question="q0",
            )
        ]
    )

    ledgers = compile_runtime_ledgers(case, _plan(), result, [step])

    assert len(ledgers.sources) == 1
    assert ledgers.sources[0].source_class == "runtime"
    assert ledgers.evidence[0].evidence_kind == "runtime_anchor"
    assert ledgers.evidence[0].image_region is None

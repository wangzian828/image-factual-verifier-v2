from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from src.orchestrator.bootstrap import build_bootstrap_investigation
from src.orchestrator.coverage import (
    activate_initial_decisive_facts,
    audit_coverage,
    compile_verdict_basis,
)
from src.orchestrator.investigation_models import (
    InvestigationSegmentOutput,
    ReflectionOutput,
    TaskUpdate,
)
from src.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    TextRegion,
)
from src.orchestrator.task_store import (
    apply_reflection,
    apply_segment_output,
    record_tool_observation,
    state_from_bootstrap,
)


def _runtime_state():
    case = ImageOnlyRuntimeCase(
        case_id="case-state-machine",
        image_path="fixture.jpg",
        image_sha256="a" * 64,
    )
    perception = PerceptionReport(
        scene_description="A marked research vessel is visible on the water.",
        entities=[
            Entity(
                name="NOAA Ship Henry B. Bigelow",
                entity_type="object",
                bbox=[0.1, 0.2, 0.9, 0.9],
                confidence=0.98,
            )
        ],
        text_regions=[
            TextRegion(
                text="HENRY B. BIGELOW",
                bbox_quad=[[0.2, 0.4], [0.7, 0.4], [0.7, 0.5], [0.2, 0.5]],
                confidence=0.98,
            ),
            TextRegion(
                text="R 225",
                bbox_quad=[[0.7, 0.5], [0.82, 0.5], [0.82, 0.58], [0.7, 0.58]],
                confidence=0.96,
            ),
        ],
    )
    state = state_from_bootstrap(
        build_bootstrap_investigation(case, perception)
    )
    activate_initial_decisive_facts(state)
    return case, state


def _step(
    *,
    task_id: str,
    call_id: str,
    tool_name: str,
    result: str,
):
    return SimpleNamespace(
        action_type="tool_call",
        tool_name=tool_name,
        tool_args={"__question_id": task_id},
        tool_result=result,
        metadata={
            "function_call_id": call_id,
            "observed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def test_discovery_is_not_evidence_and_reflection_requires_real_basis() -> None:
    case, state = _runtime_state()
    provenance = state.tasks[0]
    step = _step(
        task_id=provenance.task_id,
        call_id="call-ris",
        tool_name="reverse_image_search",
        result=(
            '{"status":"success","candidate_page_urls":["https://example.org/page"],'
            '"reference_image_candidates":["https://example.org/ref.jpg"],'
            '"lens_results":[{"title":"candidate","url":"https://example.org/page",'
            '"snippet":"search snippet","image_url":"https://example.org/ref.jpg"}],'
            '"semantic_results":[]}'
        ),
    )

    update = record_tool_observation(
        state,
        step,
        image_sha256=case.image_sha256,
    )

    assert update["created_discovery_ids"]
    assert not update["created_evidence_ids"]
    assert not state.evidence
    assert not state.findings
    invalid = ReflectionOutput(
        task_updates=[
            TaskUpdate(
                task_id=provenance.task_id,
                status="resolved",
                basis_ids=update["created_discovery_ids"],
                reason="A discovery is not a Finding.",
            )
        ]
    )
    record = apply_reflection(
        state,
        invalid,
        evidence_gain=False,
        decision_gain=False,
    )
    assert not record.accepted_task_update_ids
    assert "resolved without Finding" in record.rejected_reasons[0]


def test_official_direct_evidence_can_resolve_decisive_fact_and_compile_real() -> None:
    case, state = _runtime_state()
    for task in state.tasks:
        for fact_id in task.fact_ids:
            if fact_id in state.decisive_fact_ids:
                target_task = task
                break
        else:
            continue
        break
    else:
        raise AssertionError("no task owns a decisive fact")

    statement = "NOAA identifies the vessel as NOAA Ship Henry B. Bigelow."
    result = {
        "status": "success",
        "selected_url": "https://www.noaa.gov/ship-henry-bigelow",
        "url": "https://www.noaa.gov/ship-henry-bigelow",
        "evidence": statement,
        "summary": statement,
        "relevance": "high",
        "stance": "support",
        "directness": "direct",
        "temporal_alignment": "not_applicable",
        "artifact_sha256": "b" * 64,
        "evidence_span": {"start": 0, "end": len(statement)},
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "injection_flags": [],
        "evidence_eligible": True,
    }
    import json

    update = record_tool_observation(
        state,
        _step(
            task_id=target_task.task_id,
            call_id="call-visit",
            tool_name="visit",
            result=json.dumps(result),
        ),
        image_sha256=case.image_sha256,
    )

    assert update["created_evidence_ids"]
    assert update["created_finding_ids"]
    assert target_task.status == "resolved"
    segment = apply_segment_output(
        state,
        InvestigationSegmentOutput(
            segment_summary="Official evidence was recorded.",
            finding_proposals=[],
        ),
    )
    assert not segment["rejected_reasons"]

    for fact_id in state.decisive_fact_ids:
        if next(fact for fact in state.facts if fact.fact_id == fact_id).status != "supported":
            next(
                fact for fact in state.facts if fact.fact_id == fact_id
            ).decision_relevance = "supporting"
    state.decisive_fact_ids = [
        fact_id
        for fact_id in state.decisive_fact_ids
        if next(fact for fact in state.facts if fact.fact_id == fact_id).status
        == "supported"
    ]
    coverage = audit_coverage(state)
    verdict, basis = compile_verdict_basis(state)

    assert coverage.complete is True
    assert verdict == "real"
    assert basis.fact_ids == state.decisive_fact_ids
    assert basis.evidence_ids == update["created_evidence_ids"]

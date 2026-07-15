from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from src.orchestrator.bootstrap import build_bootstrap_investigation
from src.orchestrator.coverage import (
    activate_initial_decisive_facts,
    audit_coverage,
    compile_verdict_basis,
)
from src.orchestrator.evidence_adjudication import assess_fact
from src.orchestrator.investigation_models import (
    FactOrigin,
    Finding,
    InvestigationEvidence,
    InvestigationSegmentOutput,
    ReflectionOutput,
    TaskUpdate,
    VisualFact,
)
from src.orchestrator.image_only_prompts import render_react_context
from src.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    TextRegion,
)
from src.orchestrator.task_store import (
    apply_reflection,
    next_action_boundary,
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


def test_next_action_boundary_advances_after_reflection() -> None:
    assert next_action_boundary(0) == 4
    assert next_action_boundary(1) == 4
    assert next_action_boundary(4) == 8
    assert next_action_boundary(8) == 12
    assert next_action_boundary(23) == 24
    assert next_action_boundary(24) == 24


def test_discovery_is_not_evidence_and_reflection_only_reprioritizes() -> None:
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
    reflection = ReflectionOutput(
        task_updates=[
            TaskUpdate(
                task_id=provenance.task_id,
                priority=2,
                reason="Reprioritize without changing reducer-owned status.",
            )
        ]
    )
    record = apply_reflection(
        state,
        reflection,
        evidence_gain=False,
        decision_gain=False,
    )
    assert record.accepted_task_update_ids == [provenance.task_id]
    assert provenance.priority == 2
    assert provenance.status == "active"
    assert "status" not in TaskUpdate.model_json_schema()["properties"]
    assert "basis_ids" not in TaskUpdate.model_json_schema()["properties"]


def test_semantic_reverse_match_preserves_reference_and_is_rendered() -> None:
    case, state = _runtime_state()
    provenance = state.tasks[0]
    reference_url = "https://www.noaa.gov/media/ship.jpg"
    result = {
        "status": "success",
        "reference_image_candidates": [reference_url],
        "lens_results": [],
        "semantic_results": [
            {
                "title": "Official ship image",
                "url": "https://www.noaa.gov/ship",
                "snippet": "",
                "image_url": reference_url,
            }
        ],
    }
    import json

    record_tool_observation(
        state,
        _step(
            task_id=provenance.task_id,
            call_id="call-semantic-ris",
            tool_name="reverse_image_search",
            result=json.dumps(result),
        ),
        image_sha256=case.image_sha256,
    )

    discovery = next(
        item for item in state.discoveries if item.candidate_url.endswith("/ship")
    )
    assert discovery.reference_image_url == reference_url
    rendered = render_react_context(state)
    assert "Untested reference images" in rendered
    assert reference_url in rendered
    assert '"source_class": "official"' in rendered


def test_supporting_finding_does_not_close_unresolved_decisive_task() -> None:
    case, state = _runtime_state()
    target_task = next(
        task
        for task in state.tasks
        if any(fact_id in state.decisive_fact_ids for fact_id in task.fact_ids)
    )
    statement = "An independent page identifies a possible vessel context."
    result = {
        "status": "success",
        "selected_url": "https://example.org/possible-context",
        "url": "https://example.org/possible-context",
        "evidence": statement,
        "summary": statement,
        "relevance": "high",
        "stance": "support",
        "directness": "direct",
        "temporal_alignment": "not_applicable",
        "artifact_sha256": "c" * 64,
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
            call_id="call-supporting-visit",
            tool_name="visit",
            result=json.dumps(result),
        ),
        image_sha256=case.image_sha256,
    )

    fact = next(item for item in state.facts if item.fact_id in target_task.fact_ids)
    assert update["created_finding_ids"]
    assert fact.status == "active"
    assert target_task.status == "active"
    assert target_task.finding_ids == update["created_finding_ids"]


def test_scene_reference_requires_near_duplicate_not_only_same_subject() -> None:
    case, state = _runtime_state()
    scene_task = next(
        task
        for task in state.tasks
        if any(
            fact.fact_id in task.fact_ids and fact.predicate == "appears_to_depict"
            for fact in state.facts
        )
    )
    result = {
        "status": "success",
        "reference_url": "https://www.noaa.gov/media/different-capture.jpg",
        "same_subject_or_scene": True,
        "same_capture_or_near_duplicate": False,
        "likely_different_original_capture": True,
        "edit_evidence_present": False,
        "edit_evidence_strength": "none",
        "differences": [],
        "overall_observation": (
            "The same vessel appears in a different original capture and setting."
        ),
        "confidence": 0.95,
    }
    import json

    update = record_tool_observation(
        state,
        _step(
            task_id=scene_task.task_id,
            call_id="call-different-capture",
            tool_name="compare_with_reference",
            result=json.dumps(result),
        ),
        image_sha256=case.image_sha256,
    )

    scene_fact = next(
        fact for fact in state.facts if fact.fact_id in scene_task.fact_ids
    )
    assert update["created_evidence_ids"]
    assert not update["created_finding_ids"]
    assert scene_fact.status == "active"
    assert scene_task.status == "active"


def test_scene_support_requires_near_duplicate_and_source_assertion() -> None:
    case, state = _runtime_state()
    scene_task = next(
        task
        for task in state.tasks
        if any(
            fact.fact_id in task.fact_ids and fact.predicate == "appears_to_depict"
            for fact in state.facts
        )
    )
    result = {
        "status": "success",
        "reference_url": "https://www.noaa.gov/media/matching-capture.jpg",
        "same_subject_or_scene": True,
        "same_capture_or_near_duplicate": True,
        "likely_different_original_capture": False,
        "edit_evidence_present": False,
        "edit_evidence_strength": "none",
        "differences": [],
        "overall_observation": "The images are the same original capture.",
        "confidence": 0.99,
    }
    import json

    update = record_tool_observation(
        state,
        _step(
            task_id=scene_task.task_id,
            call_id="call-near-duplicate",
            tool_name="compare_with_reference",
            result=json.dumps(result),
        ),
        image_sha256=case.image_sha256,
    )

    scene_fact = next(
        fact for fact in state.facts if fact.fact_id in scene_task.fact_ids
    )
    assert update["created_finding_ids"]
    assert scene_fact.status == "active"
    assert scene_task.status == "active"

    statement = (
        "NOAA's source page identifies this exact image as NOAA Ship Henry B. "
        "Bigelow underway."
    )
    source_update = record_tool_observation(
        state,
        _step(
            task_id=scene_task.task_id,
            call_id="call-source-assertion",
            tool_name="visit",
            result=json.dumps(
                {
                    "status": "success",
                    "selected_url": "https://www.noaa.gov/media/ship-page",
                    "url": "https://www.noaa.gov/media/ship-page",
                    "evidence": statement,
                    "summary": statement,
                    "relevance": "high",
                    "stance": "support",
                    "directness": "direct",
                    "temporal_alignment": "not_applicable",
                    "artifact_sha256": "a" * 64,
                    "evidence_span": {
                        "start": 0,
                        "end": len(statement),
                    },
                    "retrieved_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "injection_flags": [],
                    "evidence_eligible": True,
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )

    assert scene_fact.status == "supported"
    assert scene_task.status == "resolved"
    coverage = audit_coverage(state)
    verdict, basis = compile_verdict_basis(state)
    assert verdict == "real"
    assert set(basis.evidence_ids) == set(
        update["created_evidence_ids"]
        + source_update["created_evidence_ids"]
    )
    assert coverage.facts[0].winning_evidence_ids == basis.evidence_ids


def test_generic_official_support_cannot_resolve_scene_without_visual_bridge() -> None:
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
    assert target_task.status == "active"
    segment = InvestigationSegmentOutput(
        segment_summary="Official evidence was recorded.",
    )
    assert segment.ready_for_reflection is True

    coverage = audit_coverage(state)
    verdict, basis = compile_verdict_basis(state)

    assert coverage.complete is False
    assert verdict == "unverifiable"
    assert basis.fact_ids == state.decisive_fact_ids
    assert basis.evidence_ids == update["created_evidence_ids"]


def test_direct_official_refutation_resolves_scene_and_compiles_fake() -> None:
    case, state = _runtime_state()
    target_task = next(
        task
        for task in state.tasks
        if any(
            fact_id in state.decisive_fact_ids
            for fact_id in task.fact_ids
        )
    )
    statement = (
        "NASA's event record states that this ceremony took place at Johnson "
        "Space Center, not Kennedy Space Center."
    )
    result = {
        "status": "success",
        "selected_url": "https://www.nasa.gov/official-event-record",
        "url": "https://www.nasa.gov/official-event-record",
        "evidence": statement,
        "summary": statement,
        "relevance": "high",
        "stance": "refute",
        "directness": "direct",
        "temporal_alignment": "not_applicable",
        "artifact_sha256": "d" * 64,
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
            call_id="call-official-refute",
            tool_name="visit",
            result=json.dumps(result),
        ),
        image_sha256=case.image_sha256,
    )

    assert target_task.status == "resolved"
    coverage = audit_coverage(state)
    verdict, basis = compile_verdict_basis(state)

    assert coverage.complete is True
    assert verdict == "fake"
    assert basis.fact_ids == state.decisive_fact_ids
    assert basis.evidence_ids == update["created_evidence_ids"]


def test_conflict_requires_discriminating_evidence_before_resolution() -> None:
    fact = VisualFact(
        fact_id="fact-conflict",
        kind="attribute",
        statement="The input image visibly contains the marked object.",
        subject_entity_id="entity-1",
        predicate="visible_in",
        status="active",
        basis_ids=["entity-1"],
        decision_relevance="decisive",
        origin=FactOrigin(type="input_image", origin_ids=["entity-1"]),
    )
    common = {
        "task_id": "task-conflict",
        "fact_ids": [fact.fact_id],
        "function_call_id": "call-conflict",
        "tool_name": "crop_and_inspect",
        "evidence_kind": "image_region",
        "source_url": "",
        "source_class": "visual",
        "exact_text": "A direct pixel observation.",
        "image_region": [0.0, 0.0, 1.0, 1.0],
        "artifact_sha256": "e" * 64,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "quality": "moderate",
        "directness": "direct",
        "claim_binding": "pixel_observation",
    }
    support = InvestigationEvidence(
        evidence_id="evidence-support",
        source_family="visual:support",
        stance="support",
        **common,
    )
    refute = InvestigationEvidence(
        evidence_id="evidence-refute",
        source_family="visual:refute",
        stance="refute",
        **common,
    )
    findings = [
        Finding(
            finding_id="finding-support",
            task_id="task-conflict",
            fact_ids=[fact.fact_id],
            statement="Pixel evidence supports the object.",
            stance="support",
            evidence_ids=[support.evidence_id],
            source_family_ids=[support.source_family],
        ),
        Finding(
            finding_id="finding-refute",
            task_id="task-conflict",
            fact_ids=[fact.fact_id],
            statement="Pixel evidence refutes the object.",
            stance="refute",
            evidence_ids=[refute.evidence_id],
            source_family_ids=[refute.source_family],
        ),
    ]

    tied = assess_fact(
        fact,
        findings,
        {
            support.evidence_id: support,
            refute.evidence_id: refute,
        },
        all_fact_evidence=[support, refute],
    )

    assert tied.status == "conflicted"
    assert (
        tied.conflict_resolution
        == "needs_discriminating_evidence"
    )

    decisive_refute = refute.model_copy(
        update={
            "evidence_id": "evidence-refute-official",
            "source_family": "domain:official.example",
            "source_class": "official",
            "quality": "strong",
        }
    )
    findings.append(
        Finding(
            finding_id="finding-refute-official",
            task_id="task-conflict",
            fact_ids=[fact.fact_id],
            statement="An original direct observation refutes the object.",
            stance="refute",
            evidence_ids=[decisive_refute.evidence_id],
            source_family_ids=[decisive_refute.source_family],
            quality="decisive",
        )
    )
    resolved = assess_fact(
        fact,
        findings,
        {
            support.evidence_id: support,
            refute.evidence_id: refute,
            decisive_refute.evidence_id: decisive_refute,
        },
        all_fact_evidence=[support, refute, decisive_refute],
    )

    assert resolved.status == "refuted"
    assert resolved.conflict_resolution == "refute_wins"

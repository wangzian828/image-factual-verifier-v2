from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from src.orchestrator.investigation_models import (
    FactOrigin,
    Finding,
    ImageOnlyInvestigationState,
    InvestigationBrief,
    InvestigationEvidence,
    RetrievalAnchor,
    VisualEntity,
    VisualFact,
)
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.image_only_prompts import (
    render_image_account_planning_context,
    render_discrepancy_decision_context,
    render_discrepancy_react_context,
    select_discrepancy_react_tasks,
)
from src.orchestrator.stage_runner import InteractionSession
from src.orchestrator.task_store import remaining_claim_hypothesis_routes
from src.orchestrator.runtime_case import image_sha256
from src.orchestrator.source_access import SourceAccessPolicy
from src.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    VerificationState,
)
from src.tools.base import BaseTool
from src.workflow import VerificationWorkflow, WorkflowConfig
from scripts.audit_real_trace import audit_trace


class StaticToolFixture(BaseTool):
    def __init__(
        self,
        name: str,
        result: dict[str, Any],
        parameters: dict[str, Any],
    ) -> None:
        self.name = name
        self.description = f"Controlled {name} fixture."
        self.result = result
        self.parameters = parameters

    def call(self, _params: dict[str, Any]) -> dict[str, Any]:
        return dict(self.result)


class TextSearchToolFixture(BaseTool):
    name = "text_search"
    description = "Run one bounded web query."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def call(self, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(params))
        query = str(params["query"])
        return {
            "status": "success",
            "queries": [
                {
                    "query": query,
                    "provider": "fixture",
                    "results": [
                        {
                            "title": "Candidate source",
                            "url": "https://example.org/source",
                            "snippet": "A discovery lead, not Evidence.",
                        }
                    ],
                    "search_error": "",
                    "timings": {},
                }
            ],
            "subcalls": [],
        }


class VisitToolFixture(BaseTool):
    name = "visit"
    description = "Fetch one candidate source page."
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def call(self, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(params))
        text = "The source photograph shows the person holding a microphone."
        return {
            "status": "success",
            "url": params["url"],
            "selected_url": params["url"],
            "provider": "fixture",
            "evidence_records": [
                {
                    "url": params["url"],
                    "selected_url": params["url"],
                    "evidence": text,
                    "relevance": "high",
                    "stance": "refute",
                    "relation_scope": "same_relation",
                    "relation_stance": "contradicts",
                    "directness": "direct",
                    "context_only": False,
                    "temporal_alignment": "not_applicable",
                    "artifact_sha256": "b" * 64,
                    "evidence_span": {"start": 0, "end": len(text)},
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "injection_flags": [],
                    "evidence_eligible": True,
                }
            ],
            "subcalls": [],
        }


class ImageAccountPlanningBackend:
    provider = "gemini"
    wire_api = "interactions"

    def __init__(self, *, unknown_anchor: bool = False) -> None:
        self.unknown_anchor = unknown_anchor
        self.requests: list[dict[str, Any]] = []
        self.anchor_id = ""

    async def create_interaction(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        payload = kwargs["input_payload"]
        if isinstance(payload, list):
            text = next(
                str(item.get("text", ""))
                for item in payload
                if isinstance(item, dict) and item.get("type") == "text"
            )
            context = json.loads(text)
            anchor_id = context["visual_fact_anchors"][0]["fact_id"]
            self.anchor_id = anchor_id
        else:
            anchor_id = self.anchor_id or "unknown-anchor"
        if self.unknown_anchor:
            anchor_id = "unknown-anchor"
        response = {
            "account_summary": (
                "The image presents a person visibly holding a product packet."
            ),
            "image_claims": [
                {
                    "claim_key": "person_product_relation",
                    "statement": (
                        "The visible person is holding the shown product packet."
                    ),
                    "kind": "relation",
                    "predicate": "holds",
                    "anchor_fact_ids": [anchor_id],
                    "salience": "high",
                }
            ],
            "search_hypotheses": [
                {
                    "hypothesis_key": "source_capture",
                    "statement": (
                        "A traceable source capture may clarify what the person held."
                    ),
                    "queries": ["person source capture held object"],
                    "expected_information": (
                        "A source page or reference image for direct comparison."
                    ),
                    "suggested_tools": [
                        "text_search",
                        "compare_with_reference",
                    ],
                    "priority": 1,
                }
            ],
        }
        request_number = len(self.requests)
        return {
            "id": f"image-account-planning-{request_number}",
            "status": "completed",
            "usage": {
                "total_input_tokens": 1,
                "total_output_tokens": 1,
                "total_thought_tokens": 0,
            },
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {"type": "text", "text": json.dumps(response)}
                    ],
                }
            ],
        }


class PolicyRevisionPlanningBackend(ImageAccountPlanningBackend):
    async def create_interaction(self, **kwargs: Any) -> dict[str, Any]:
        response = await super().create_interaction(**kwargs)
        if len(self.requests) == 1:
            content = response["steps"][0]["content"][0]
            payload = json.loads(content["text"])
            payload["search_hypotheses"][0]["queries"] = [
                "viral image hoax visible person product"
            ]
            content["text"] = json.dumps(payload)
        return response


class PlanningThenReactBackend(ImageAccountPlanningBackend):
    def __init__(self) -> None:
        super().__init__()
        self.react_task_id = ""
        self.react_count = 0

    async def create_interaction(self, **kwargs: Any) -> dict[str, Any]:
        system = str(kwargs.get("system_instruction", ""))
        if "Image Account Planning root" in system:
            return await super().create_interaction(**kwargs)
        self.requests.append(kwargs)
        if "sparse multimodal Discrepancy Decision checkpoint" in system:
            payload = kwargs["input_payload"]
            if isinstance(payload, list):
                user_input = payload[-1]
                context = json.loads(user_input["content"][0]["text"])
            else:
                context = json.loads(payload)
            claim = context["image_claims"][0]
            evidence = context["reviewed_evidence"][0]
            return {
                "id": "discrepancy-decision-1",
                "status": "completed",
                "usage": {
                    "total_input_tokens": 1,
                    "total_output_tokens": 1,
                    "total_thought_tokens": 0,
                },
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "claim_assessments": [
                                            {
                                                "claim_id": claim["claim_id"],
                                                "assessment": "refuted",
                                                "selected_evidence_ids": [
                                                    evidence["evidence_id"]
                                                ],
                                                "remaining_gap": "",
                                                "rationale": (
                                                    "The direct source contradicts "
                                                    "the depicted relation."
                                                ),
                                            }
                                        ],
                                        "material_discrepancy": {
                                            "statement": (
                                                "The source says the person held a "
                                                "microphone rather than the shown packet."
                                            ),
                                            "affected_claim_ids": [
                                                claim["claim_id"]
                                            ],
                                            "evidence_ids": [
                                                evidence["evidence_id"]
                                            ],
                                            "materiality": "decisive",
                                            "status": "established",
                                            "rationale": (
                                                "The changed held object alters the "
                                                "high-salience relation."
                                            ),
                                        },
                                        "retire_hypothesis_ids": [],
                                        "new_hypotheses": [],
                                        "visual_reinspection": None,
                                        "visual_evidence_disposition": None,
                                        "verdict_proposal": "fake",
                                        "rationale": (
                                            "Qualified Evidence establishes a "
                                            "decisive discrepancy."
                                        ),
                                    }
                                ),
                            }
                        ],
                    }
                ],
            }
        if "constrained final synthesizer for discrepancy-first-v4" in system:
            payload = kwargs["input_payload"]
            assert isinstance(payload, str)
            context = json.loads(payload)
            basis = context["compiled_basis"]
            return {
                "id": "discrepancy-judgment-1",
                "status": "completed",
                "usage": {
                    "total_input_tokens": 1,
                    "total_output_tokens": 1,
                    "total_thought_tokens": 0,
                },
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "verdict": context["compiled_verdict"],
                                        "confidence": 0.99,
                                        "overall_assessment": (
                                            "The compiled Evidence establishes the "
                                            "selected material discrepancy."
                                        ),
                                    }
                                ),
                            }
                        ],
                    }
                ],
            }
        self.react_count += 1
        payload = kwargs["input_payload"]
        if isinstance(payload, list):
            user_input = payload[-1]
            context = json.loads(user_input["content"][0]["text"])
        else:
            context = json.loads(payload)
        active_task = context["active_tasks"][0]
        self.react_task_id = active_task["task_id"]
        if self.react_count == 1:
            tool_name = "text_search"
            arguments = {
                "question_id": self.react_task_id,
                "query": "person source capture held object",
            }
        else:
            tool_name = "visit"
            arguments = {
                "question_id": self.react_task_id,
                "url": "https://example.org/source",
                "claim_id": active_task["owned_claims"][0]["claim_id"],
            }
        return {
            "id": f"discrepancy-react-{self.react_count}",
            "status": "requires_action",
            "usage": {
                "total_input_tokens": 1,
                "total_output_tokens": 1,
                "total_thought_tokens": 0,
            },
            "steps": [
                {
                    "type": "function_call",
                    "id": f"call-v4-{tool_name}",
                    "name": tool_name,
                    "arguments": arguments,
                }
            ],
        }


def _state(image_path: Path) -> tuple[VerificationState, ImageOnlyInvestigationState]:
    entity = VisualEntity(
        entity_id="entity-person",
        name="visible person",
        entity_type="person",
        region=[0.1, 0.1, 0.8, 0.9],
        confidence=0.98,
    )
    fact = VisualFact(
        fact_id="fact-visible-person-product",
        kind="relation",
        statement="A visible person is holding a product packet.",
        subject_entity_id=entity.entity_id,
        predicate="visible_in",
        status="active",
        basis_ids=[entity.entity_id],
        decision_relevance="supporting",
        origin=FactOrigin(type="input_image", origin_ids=[entity.entity_id]),
    )
    investigation = ImageOnlyInvestigationState(
        brief=InvestigationBrief(
            brief_id="brief-v4-planning",
            case_id="case-v4-planning",
        ),
        entities=[entity],
        facts=[fact],
    )
    verification = VerificationState(
        image_path=str(image_path),
        image_id="case-v4-planning",
        input_mode="image_only",
        decision_policy_version="reinspect-v2",
        perception=PerceptionReport(
            scene_description="A person holds a product packet.",
            image_type="photo",
            entities=[
                Entity(
                    name="visible person",
                    entity_type="person",
                    bbox=[0.1, 0.1, 0.8, 0.9],
                    confidence=0.98,
                )
            ],
        ),
        investigation_state=investigation,
    )
    return verification, investigation


def test_image_account_planning_is_image_root_and_installs_claim_graph(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "planning.jpg"
    image_path.write_bytes(b"v4-planning-image")
    state, investigation = _state(image_path)
    backend = ImageAccountPlanningBackend()
    orchestrator = Orchestrator(validate_startup=False)
    orchestrator.llm = backend
    session = InteractionSession()

    asyncio.run(
        orchestrator._run_image_account_planning(
            state,
            investigation,
            interaction_session=session,
        )
    )

    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request["previous_interaction_id"] is None
    assert any(
        isinstance(item, dict) and item.get("type") == "image"
        for item in request["input_payload"]
    )
    assert session.previous_interaction_id == "image-account-planning-1"
    assert investigation.core_verdict_fact_id is None
    assert len(investigation.image_claims) == 1
    assert len(investigation.search_hypotheses) == 1
    claim = investigation.image_claims[0]
    hypothesis = investigation.search_hypotheses[0]
    task = next(item for item in investigation.tasks if item.task_id == hypothesis.task_id)
    assert hypothesis.claim_ids == [claim.claim_id]
    assert task.claim_ids == [claim.claim_id]
    assert task.hypothesis_id == hypothesis.hypothesis_id
    planning_step = state.all_steps[0]
    assert planning_step.stage_name == "image_account_planning"
    snapshot = planning_step.metadata["policy_input"]["input_payload"]
    assert any(item.get("runtime_image") is True for item in snapshot)
    assert all("data" not in item for item in snapshot if isinstance(item, dict))


def test_image_account_planning_rejects_policy_query_before_atomic_commit(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "planning-query-policy.jpg"
    image_path.write_bytes(b"v4-planning-query-policy")
    state, investigation = _state(image_path)
    backend = PolicyRevisionPlanningBackend()
    orchestrator = Orchestrator(
        validate_startup=False,
        source_access_policy=SourceAccessPolicy(
            policy_id="evaluation",
            excluded_domains=frozenset({"factcrescendo.com"}),
        ),
    )
    orchestrator.llm = backend

    asyncio.run(
        orchestrator._run_image_account_planning(
            state,
            investigation,
            interaction_session=InteractionSession(),
        )
    )

    assert len(backend.requests) == 2
    assert state.all_steps[0].action_type == "planning_revision"
    assert "ready-made fact-check verdict" in state.all_steps[0].metadata[
        "planning_revision_reason"
    ]
    assert state.all_steps[1].action_type != "planning_revision"
    assert investigation.search_hypotheses[0].queries == [
        "person source capture held object"
    ]
    assert investigation.tasks[-1].suggested_queries == [
        "person source capture held object"
    ]


def test_image_account_planning_context_excludes_judgment_controls(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "planning-projection.jpg"
    image_path.write_bytes(b"v4-planning-projection")
    state, investigation = _state(image_path)

    context = json.loads(
        render_image_account_planning_context(
            investigation,
            perception=state.perception,
        )
    )

    assert context["case"] == {
        "case_id": investigation.brief.case_id,
        "input_mode": investigation.brief.input_mode,
        "media_type": investigation.brief.media_type,
    }
    assert "brief" not in context
    assert "required_output" not in json.dumps(context)
    assert "stop_policy" not in json.dumps(context)
    assert context["context_role"] == "observations_only"
    assert "bootstrap_tasks" not in context
    assert "visual_entities" not in context
    assert "pixel_ocr_visual_facts" not in context
    assert "visual_fact_anchors" in context


def test_image_account_planning_context_deduplicates_internal_bootstrap_rows(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "planning-compact.jpg"
    image_path.write_bytes(b"v4-planning-compact")
    state, investigation = _state(image_path)
    base_fact = investigation.facts[0]
    for index in range(40):
        investigation.facts.append(
            VisualFact(
                fact_id=f"fact-derived-{index}",
                kind="relation",
                statement=(
                    f'The visible text "TOKEN-{index}" is a candidate anchor '
                    "for identifying the depicted entity, place, or event."
                ),
                subject_entity_id=base_fact.subject_entity_id,
                predicate="context_suggested_by_text",
                object_entity_id=base_fact.object_entity_id,
                status="candidate",
                basis_ids=[base_fact.fact_id],
                origin=FactOrigin(type="ocr", origin_ids=[base_fact.fact_id]),
            )
        )
    for index in range(35):
        investigation.facts.append(
            VisualFact(
                fact_id=f"fact-observed-{index}",
                kind="attribute",
                statement=f"Distinct useful visual observation {index}.",
                subject_entity_id=base_fact.subject_entity_id,
                predicate="visible_detail",
                status="candidate",
                basis_ids=[base_fact.fact_id],
                origin=FactOrigin(type="input_image", origin_ids=[base_fact.fact_id]),
            )
        )
        investigation.retrieval_anchors.append(
            RetrievalAnchor(
                anchor_id=f"anchor-observed-{index}",
                kind="entity",
                value=f"distinct useful retrieval clue {index}",
                confidence=0.8,
            )
        )

    rendered = render_image_account_planning_context(
        investigation,
        perception=state.perception,
    )
    context = json.loads(rendered)

    assert all(
        item["predicate"] != "context_suggested_by_text"
        for item in context["visual_fact_anchors"]
    )
    assert sum(
        item["predicate"] == "visible_detail"
        for item in context["visual_fact_anchors"]
    ) == 35
    assert len(context["retrieval_clues"]) == 35


def test_image_account_planning_revisions_are_atomic_and_inherit_image_root(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "invalid-planning.jpg"
    image_path.write_bytes(b"invalid-v4-planning-image")
    state, investigation = _state(image_path)
    backend = ImageAccountPlanningBackend(unknown_anchor=True)
    orchestrator = Orchestrator(validate_startup=False)
    orchestrator.llm = backend
    session = InteractionSession()
    before = investigation.model_dump(mode="json")

    with pytest.raises(RuntimeError, match="valid claim graph"):
        asyncio.run(
            orchestrator._run_image_account_planning(
                state,
                investigation,
                interaction_session=session,
            )
        )

    assert investigation.model_dump(mode="json") == before
    assert len(backend.requests) == 3
    assert backend.requests[0]["previous_interaction_id"] is None
    assert backend.requests[1]["previous_interaction_id"] == (
        "image-account-planning-1"
    )
    assert backend.requests[2]["previous_interaction_id"] == (
        "image-account-planning-2"
    )
    assert any(
        item.get("type") == "image"
        for item in backend.requests[0]["input_payload"]
        if isinstance(item, dict)
    )
    assert all(
        not isinstance(request["input_payload"], list)
        for request in backend.requests[1:]
    )
    assert all(step.action_type == "planning_revision" for step in state.all_steps)


def test_planning_to_react_keeps_one_image_chain_and_claim_ownership(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "planning-react.jpg"
    image_path.write_bytes(b"v4-planning-react-image")
    state, investigation = _state(image_path)
    runtime_case = ImageOnlyRuntimeCase(
        case_id="case-v4-planning",
        image_path=str(image_path),
        image_sha256=image_sha256(str(image_path)),
    )
    state.runtime_case = runtime_case
    backend = PlanningThenReactBackend()
    search = TextSearchToolFixture()
    orchestrator = Orchestrator(validate_startup=False)
    orchestrator.llm = backend
    orchestrator.all_tools = {"text_search": search}
    orchestrator.verification_tool_limits = {"text_search": 2}
    session = InteractionSession()

    asyncio.run(
        orchestrator._run_image_account_planning(
            state,
            investigation,
            interaction_session=session,
        )
    )
    update = asyncio.run(
        orchestrator._run_discrepancy_react_action(
            state,
            investigation,
            str(image_path),
            runtime_case,
            interaction_session=session,
        )
    )

    assert len(backend.requests) == 2
    planning_request, react_request = backend.requests
    assert planning_request["previous_interaction_id"] is None
    assert react_request["previous_interaction_id"] == "image-account-planning-1"
    assert not isinstance(react_request["input_payload"], list)
    assert update["task_id"] == backend.react_task_id
    assert update["created_discovery_ids"]
    task = next(
        item for item in investigation.tasks if item.task_id == update["task_id"]
    )
    hypothesis = next(
        item
        for item in investigation.search_hypotheses
        if item.hypothesis_id == task.hypothesis_id
    )
    assert task.claim_ids == hypothesis.claim_ids
    assert hypothesis.status == "active"
    assert hypothesis.attempt_count == 1
    discovery = investigation.discoveries[0]
    assert discovery.task_id == task.task_id
    assert discovery.fact_ids == task.fact_ids
    assert investigation.evidence == []
    assert search.calls == [
        {
            "query": "person source capture held object",
            "goal": "A source page or reference image for direct comparison.",
        }
    ]
    assert session.previous_interaction_id == "discrepancy-react-1"
    assert len(session.pending_input) == 1


def test_discrepancy_action_selects_one_task_scoped_claim_set(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "one-task-routes.jpg"
    image_path.write_bytes(b"one-task-routes")
    state, investigation = _state(image_path)
    backend = ImageAccountPlanningBackend()
    orchestrator = Orchestrator(validate_startup=False)
    orchestrator.llm = backend
    asyncio.run(
        orchestrator._run_image_account_planning(
            state,
            investigation,
            interaction_session=None,
        )
    )
    first = investigation.tasks[0]
    second = first.model_copy(deep=True)
    second.task_id = "task-second-route"
    second.hypothesis_id = "hypothesis-second-route"
    second.priority = 2
    investigation.tasks.append(second)
    second_hypothesis = investigation.search_hypotheses[0].model_copy(deep=True)
    second_hypothesis.hypothesis_id = second.hypothesis_id
    second_hypothesis.task_id = second.task_id
    second_hypothesis.statement = "A second independent route."
    investigation.search_hypotheses.append(second_hypothesis)

    selected = select_discrepancy_react_tasks(investigation)[:1]
    assert [task.task_id for task in selected] == [first.task_id]
    claim_options = orchestrator._discrepancy_task_claim_options(
        investigation,
        task_ids={first.task_id},
    )
    assert list(claim_options) == [first.task_id]
    assert set(claim_options[first.task_id]) == set(first.claim_ids)
    react_context = json.loads(
        render_discrepancy_react_context(
            investigation,
            task_ids={first.task_id},
        )
    )
    assert [item["task_id"] for item in react_context["active_tasks"]] == [
        first.task_id
    ]


def test_discrepancy_action_skips_exhausted_active_task(
    tmp_path: Path,
) -> None:
    """An active sibling with no route must not mask executable work."""

    image_path = tmp_path / "skip-exhausted-route.jpg"
    image_path.write_bytes(b"skip-exhausted-route")
    state, investigation = _state(image_path)
    backend = ImageAccountPlanningBackend()
    orchestrator = Orchestrator(validate_startup=False)
    orchestrator.llm = backend
    asyncio.run(
        orchestrator._run_image_account_planning(
            state,
            investigation,
            interaction_session=None,
        )
    )
    exhausted = investigation.tasks[0]
    executable = exhausted.model_copy(deep=True)
    executable.task_id = "task-executable-sibling"
    executable.hypothesis_id = "hypothesis-executable-sibling"
    executable.priority = 2
    investigation.tasks.append(executable)
    executable_hypothesis = investigation.search_hypotheses[0].model_copy(
        deep=True
    )
    executable_hypothesis.hypothesis_id = executable.hypothesis_id
    executable_hypothesis.task_id = executable.task_id
    executable_hypothesis.statement = "An executable sibling route."
    investigation.search_hypotheses.append(executable_hypothesis)

    # Consume the first task's only text-search route while leaving its
    # deterministic state active, which is the state observed in the real
    # failed baseline trace.
    investigation.attempted_routes.append(
        json.dumps(
            {
                "tool": "text_search",
                "task_id": exhausted.task_id,
                "queries": ["person source capture held object"],
                "outcome": "empty",
            }
        )
    )
    investigation.attempted_routes.append(
        json.dumps(
            {
                "tool": "text_search",
                "task_id": exhausted.task_id,
                "queries": ["person source capture held object alternative"],
                "outcome": "empty",
            }
        )
    )

    assert remaining_claim_hypothesis_routes(
        investigation,
        task_ids={exhausted.task_id},
    ) == []
    assert remaining_claim_hypothesis_routes(
        investigation,
        task_ids={executable.task_id},
    ) == [f"text_search:{executable.task_id}"]
    assert [
        task.task_id for task in select_discrepancy_react_tasks(investigation)
    ] == [executable.task_id]


def test_discrepancy_decision_consumes_pending_result_on_same_chain(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "planning-react-decision.jpg"
    image_path.write_bytes(b"v4-planning-react-decision-image")
    state, investigation = _state(image_path)
    runtime_case = ImageOnlyRuntimeCase(
        case_id="case-v4-planning",
        image_path=str(image_path),
        image_sha256=image_sha256(str(image_path)),
    )
    state.runtime_case = runtime_case
    backend = PlanningThenReactBackend()
    search = TextSearchToolFixture()
    orchestrator = Orchestrator(validate_startup=False)
    orchestrator.llm = backend
    orchestrator.all_tools = {"text_search": search}
    orchestrator.verification_tool_limits = {"text_search": 2}
    session = InteractionSession()

    asyncio.run(
        orchestrator._run_image_account_planning(
            state,
            investigation,
            interaction_session=session,
        )
    )
    react_update = asyncio.run(
        orchestrator._run_discrepancy_react_action(
            state,
            investigation,
            str(image_path),
            runtime_case,
            interaction_session=session,
        )
    )
    claim = investigation.image_claims[0]
    task = next(
        item for item in investigation.tasks if item.task_id == react_update["task_id"]
    )
    evidence = InvestigationEvidence(
        evidence_id="evidence-direct-source",
        task_id=task.task_id,
        fact_ids=[claim.fact_id],
        function_call_id="call-direct-source",
        tool_name="visit",
        evidence_kind="web_span",
        source_url="https://example.org/source",
        source_family="domain:example.org",
        exact_text="The source photograph shows the person holding a microphone.",
        span_start=0,
        span_end=61,
        artifact_sha256="a" * 64,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        stance="refute",
        quality="strong",
        directness="direct",
        claim_binding="source_assertion",
        relation_scope="same_relation",
        relation_stance="contradicts",
    )
    investigation.evidence.append(evidence)
    finding = Finding(
        finding_id="finding-direct-source",
        task_id=task.task_id,
        fact_ids=[claim.fact_id],
        statement="The direct source refutes the shown held-object relation.",
        stance="refute",
        evidence_ids=[evidence.evidence_id],
        source_family_ids=[evidence.source_family],
        quality="decisive",
    )
    investigation.findings.append(finding)
    task.finding_ids.append(finding.finding_id)

    decision_context = json.loads(
        render_discrepancy_decision_context(
            investigation,
            reviewed_evidence_ids=[evidence.evidence_id],
            trigger="qualified_evidence",
        )
    )
    assert decision_context["reviewable_claim_ids"] == [claim.claim_id]
    assert "assessable_claim_ids" not in decision_context
    assert decision_context["reviewed_evidence_ownership"] == [
        {
            "evidence_id": evidence.evidence_id,
            "task_id": task.task_id,
            "claim_ids": [claim.claim_id],
            "hypothesis_id": task.hypothesis_id,
        }
    ]
    assert decision_context["reviewed_evidence"][0]["admissible_stances"] == [
        "refute"
    ]
    assert decision_context["claim_update_space"] == [
        {
            "claim_id": claim.claim_id,
            "allowed_visual_anchor_fact_ids": claim.anchor_fact_ids,
        }
    ]
    assert decision_context["reviewed_directional_chains"] == [
        {
            "stance": "refute",
            "finding_id": finding.finding_id,
            "evidence_ids": [evidence.evidence_id],
            "task_id": task.task_id,
            "task_owned_claim_ids": [claim.claim_id],
        }
    ]
    assert "does not prove" in decision_context["ownership_note"]

    update = asyncio.run(
        orchestrator._run_discrepancy_decision(
            state,
            investigation,
            reviewed_evidence_ids=[evidence.evidence_id],
            trigger="qualified_evidence",
            interaction_session=session,
        )
    )

    assert len(backend.requests) == 3
    decision_request = backend.requests[2]
    assert decision_request["previous_interaction_id"] == "discrepancy-react-1"
    decision_input = decision_request["input_payload"]
    assert [item["type"] for item in decision_input] == [
        "function_result",
        "user_input",
    ]
    assert decision_input[0]["call_id"] == "call-v4-text_search"
    assert "reviewed_evidence" in decision_input[1]["content"][0]["text"]
    assert update["verdict_proposal"] == "fake"
    assert investigation.proposed_verdict == "fake"
    assert investigation.image_claims[0].status == "refuted"
    assert len(investigation.material_discrepancies) == 1
    assert investigation.material_discrepancies[0].evidence_ids == [
        evidence.evidence_id
    ]
    assert session.previous_interaction_id == "discrepancy-decision-1"
    assert session.pending_input == []


def test_discrepancy_context_marks_neutral_reference_as_non_directional(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "neutral-reference.jpg"
    image_path.write_bytes(b"neutral-reference-image")
    state, investigation = _state(image_path)
    backend = ImageAccountPlanningBackend()
    orchestrator = Orchestrator(validate_startup=False)
    orchestrator.llm = backend
    asyncio.run(
        orchestrator._run_image_account_planning(
            state,
            investigation,
            interaction_session=None,
        )
    )
    claim = investigation.image_claims[0]
    task = next(item for item in investigation.tasks if item.claim_ids)
    evidence = InvestigationEvidence(
        evidence_id="evidence-neutral-reference",
        task_id=task.task_id,
        fact_ids=[claim.fact_id],
        function_call_id="call-neutral-reference",
        tool_name="compare_with_reference",
        evidence_kind="reference_comparison",
        source_url="https://example.org/reference.jpg",
        source_family="domain:example.org",
        exact_text="The images are crops of the same original capture.",
        artifact_sha256="b" * 64,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        stance="neutral",
        quality="moderate",
        directness="direct",
        claim_binding="same_capture",
        same_subject_or_scene=True,
        same_capture_or_near_duplicate=True,
        likely_different_original_capture=False,
        edit_evidence_present=False,
    )
    investigation.evidence.append(evidence)

    context = json.loads(
        render_discrepancy_decision_context(
            investigation,
            reviewed_evidence_ids=[evidence.evidence_id],
            trigger="qualified_evidence",
        )
    )

    assert context["reviewed_evidence"][0]["stance"] == "neutral"
    assert context["reviewed_evidence"][0]["admissible_stances"] == []
    assert context["reviewed_directional_chains"] == []


def test_discrepancy_main_loop_stops_on_first_decisive_discrepancy(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "v4-main-loop.jpg"
    image_path.write_bytes(b"v4-main-loop-image")
    state, investigation = _state(image_path)
    runtime_case = ImageOnlyRuntimeCase(
        case_id="case-v4-planning",
        image_path=str(image_path),
        image_sha256=image_sha256(str(image_path)),
    )
    state.runtime_case = runtime_case
    backend = PlanningThenReactBackend()
    search = TextSearchToolFixture()
    visit = VisitToolFixture()
    orchestrator = Orchestrator(validate_startup=False)
    orchestrator.llm = backend
    orchestrator.all_tools = {"text_search": search, "visit": visit}
    orchestrator.verification_tool_limits = {"text_search": 2, "visit": 4}
    session = InteractionSession()

    asyncio.run(
        orchestrator._run_image_account_planning(
            state,
            investigation,
            interaction_session=session,
        )
    )
    asyncio.run(
        orchestrator._run_discrepancy_investigation(
            state,
            investigation,
            str(image_path),
            runtime_case,
        )
    )

    assert investigation.action_count == 2
    assert len(investigation.discoveries) == 1
    assert len(investigation.evidence) == 1
    assert len(investigation.discrepancy_decisions) == 1
    assert len(investigation.material_discrepancies) == 1
    assert investigation.proposed_verdict == "fake"
    assert investigation.stop_reason == "verdict_determined"
    assert investigation.discrepancy_coverage_audits[-1].complete is True
    assert backend.react_count == 2
    assert [request["previous_interaction_id"] for request in backend.requests] == [
        None,
        None,
        None,
        None,
    ]
    assert search.calls
    assert visit.calls
    assert all(
        step.stage_name != "image_only_discrepancy_investigation"
        for step in state.all_steps
        if step.round > investigation.action_count
    )


def test_default_workflow_runs_complete_discrepancy_first_v4_path(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "default-v4-workflow.jpg"
    image_path.write_bytes(b"default-v4-workflow-image")
    runtime_case = ImageOnlyRuntimeCase(
        case_id="case-default-v4-workflow",
        image_path=str(image_path),
        image_sha256=image_sha256(str(image_path)),
    )
    backend = PlanningThenReactBackend()
    search = TextSearchToolFixture()
    visit = VisitToolFixture()
    image_schema = {
        "type": "object",
        "properties": {"image_input": {"type": "string"}},
        "required": ["image_input"],
    }
    orchestrator = Orchestrator(validate_startup=False)
    orchestrator.vlm_provider = "controlled"
    orchestrator.llm = backend
    orchestrator.all_tools = {
        "perceive_scene": StaticToolFixture(
            "perceive_scene",
            {
                "status": "success",
                "entities": [
                    {
                        "name": "visible person",
                        "entity_type": "person",
                        "bbox": [0.1, 0.1, 0.8, 0.9],
                        "confidence": 0.98,
                    }
                ],
                "scene_description": "A person holds a product packet.",
                "image_type": "photo",
            },
            image_schema,
        ),
        "ocr_with_position": StaticToolFixture(
            "ocr_with_position",
            {"status": "success", "text_regions": []},
            image_schema,
        ),
        "text_search": search,
        "visit": visit,
    }
    orchestrator.tool_health_summary = {
        name: {"available": True, "error": ""}
        for name in orchestrator.all_tools
    }
    orchestrator.verification_tool_limits = {"text_search": 2, "visit": 4}
    workflow = VerificationWorkflow(
        WorkflowConfig(
            output_dir=str(tmp_path / "traces"),
            save_traces=True,
        )
    )
    workflow._orchestrator = orchestrator

    result = asyncio.run(
        workflow.run_single(
            str(image_path),
            runtime_case.case_id,
            runtime_case=runtime_case,
        )
    )

    assert result["decision_policy_version"] == "discrepancy-first-v4"
    assert result["verdict"] == "fake"
    assert result["termination"] == "success"
    assert result["verdict_basis"]["policy_rule_id"] == "discrepancy-first-v4"
    assert result["verdict_basis"]["discrepancy_ids"]
    investigation = result["state"]["investigation_state"]
    assert investigation["core_verdict_fact_id"] is None
    assert investigation["action_count"] == 2
    assert investigation["stop_reason"] == "verdict_determined"
    assert investigation["discrepancy_judgment"]["verdict"] == "fake"
    trace_path = tmp_path / "traces" / f"{runtime_case.case_id}.json"
    assert trace_path.is_file()
    report = audit_trace(trace_path)
    assert not report.failures(strict_scheduler=True)


def test_public_orchestrator_rejects_frozen_v3_policy(tmp_path: Path) -> None:
    image_path = tmp_path / "reject-v3.jpg"
    image_path.write_bytes(b"reject-public-v3-policy")
    runtime_case = ImageOnlyRuntimeCase(
        case_id="case-reject-v3",
        image_path=str(image_path),
        image_sha256=image_sha256(str(image_path)),
    )
    orchestrator = Orchestrator(validate_startup=False)

    with pytest.raises(RuntimeError, match="discrepancy-first-v4"):
        asyncio.run(
            orchestrator.run(
                str(image_path),
                runtime_case,
                decision_policy_version="reinspect-v2",
            )
        )

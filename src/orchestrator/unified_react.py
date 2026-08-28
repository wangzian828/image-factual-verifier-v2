"""First-class unified ReAct runtime contract for ``unified-react-v1``.

This module owns only orchestrator concerns: dynamic tool exposure, adapter-only
arguments, and deterministic state reduction.  It deliberately delegates every
real tool call to the existing mature tool implementation unchanged.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence

from src.orchestrator.bootstrap import build_visual_bootstrap
from src.orchestrator.evidence_policy import query_policy_violation
from src.orchestrator.investigation_models import (
    ImageOnlyInvestigationState,
    InvestigationIntent,
    UnifiedReactBootstrapFailure,
)
from src.orchestrator.progress_control import record_action_progress
from src.orchestrator.route_policy import (
    route_signature,
    routes_semantically_equivalent,
)
from src.orchestrator.state import ImageOnlyRuntimeCase, PerceptionReport
from src.orchestrator.source_provenance import canonicalize_url
from src.orchestrator.task_store import (
    MAX_TOOL_ACTIONS,
    apply_initial_action_intent,
    apply_unified_stop_route,
    build_empty_investigation,
    record_tool_observation,
    remaining_claim_hypothesis_routes,
    remaining_root_image_reverse_branches,
    validate_initial_action_intent,
)
from src.orchestrator.tool_result import parse_tool_result
from src.tools.base import BaseTool


UNIFIED_REACT_POLICY_VERSION = "unified-react-v1"
UNIFIED_REACT_STAGE = "unified_react"
_BOOTSTRAP_TOOLS = ("perceive_scene", "ocr_with_position")
_FIRST_INVESTIGATION_TOOLS = (
    "text_search",
    "reverse_image_search",
    "focused_visual_inspection",
    "crop_and_inspect",
    "check_consistency",
    "analyze_visual_anomalies",
)
_ADAPTER_ONLY_FIELDS = frozenset({"task_id", "investigation_intent"})


def _parse_initial_investigation_intent(
    raw_intent: Any,
    *,
    tool_name: str,
) -> InvestigationIntent:
    """Parse compact provider input and bind the actual first tool.

    The canonical model keeps two or three complete route objects.  The
    provider-facing contract deliberately emits only a primary route, so the
    model validator expands it before the reducer sees it.  Binding the actual
    function here also prevents a harmless provider omission from becoming a
    protocol rejection: the first route is, by definition, the route started
    by this function call.
    """

    intent = InvestigationIntent.model_validate(raw_intent)
    payload = intent.model_dump(mode="python")
    routes = list(payload.get("routes", []) or [])
    if not routes:
        raise ValueError("initial investigation intent has no routes")
    first = dict(routes[0])
    first["suggested_tools"] = list(
        dict.fromkeys([tool_name, *(first.get("suggested_tools", []) or [])])
    )[:4]
    routes[0] = first
    payload["routes"] = routes
    return InvestigationIntent.model_validate(payload)


def _investigation_intent_tool_schema() -> dict[str, Any]:
    """Return a compact Gemini-compatible schema without JSON-schema refs.

    The runtime stores two or three routes, but the provider only needs to emit
    one primary route plus optional route-focus hints.  Keeping the wire shape
    compact avoids invalid-argument failures on Gemini 3.6 while preserving the
    canonical multi-route state after normalization.
    """

    route_schema = {
        "type": "object",
        "properties": {
            "route_focus": {
                "type": "string",
                "enum": [
                    "same_capture_reference",
                    "entity_event_identity",
                    "relation_value",
                    "scene_world_constraints",
                    "visual_consistency",
                ],
            },
            "expected_information": {
                "type": "string",
                "description": (
                    "Concrete information this route should recover about the "
                    "same target relation."
                ),
            },
            "priority": {"type": "integer", "enum": [1, 2, 3]},
        },
        "required": ["route_focus", "expected_information", "priority"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "target_fact": {
                "type": "object",
                "properties": {
                    "statement": {
                        "type": "string",
                        "description": (
                            "Positive atomic real-world proposition conveyed by "
                            "the image. State the depicted subject, event, "
                            "relation, or value; do not phrase an AI-generation, "
                            "manipulation, real/fake, or provenance question."
                        ),
                    },
                    "kind": {
                        "type": "string",
                        "enum": [
                            "attribute",
                            "relation",
                            "internal_consistency",
                            "text_claim",
                        ],
                    },
                    "predicate": {
                        "type": "string",
                        "description": "Short relation/property name.",
                    },
                    "anchor_fact_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "VisualFact IDs from completed scene/OCR observations."
                        ),
                    },
                },
                "required": ["statement", "kind", "predicate", "anchor_fact_ids"],
                "additionalProperties": False,
            },
            "route": {
                **route_schema,
                "description": "PRIMARY route for the first investigation action.",
            },
            "alternate_route_focuses": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "string",
                    "enum": [
                        "same_capture_reference",
                        "entity_event_identity",
                        "relation_value",
                        "scene_world_constraints",
                        "visual_consistency",
                    ],
                },
                "description": (
                    "Optional distinct route focuses. The runtime fills any "
                    "missing complementary routes."
                ),
            },
        },
        "required": ["target_fact", "route"],
        "additionalProperties": False,
    }


def new_unified_react_state(
    runtime_case: ImageOnlyRuntimeCase,
) -> ImageOnlyInvestigationState:
    """Create the empty canonical workspace for one unified-ReAct episode."""

    return build_empty_investigation(
        case_id=runtime_case.case_id,
        image_sha256=runtime_case.image_sha256,
    )


def bootstrap_complete(state: ImageOnlyInvestigationState) -> bool:
    return set(state.unified_react_bootstrap_tools_completed) == set(
        _BOOTSTRAP_TOOLS
    )


def available_unified_react_tool_names(
    state: ImageOnlyInvestigationState,
) -> list[str]:
    """Return the exact tools the policy may call in the current workspace."""

    completed = set(state.unified_react_bootstrap_tools_completed)
    missing_bootstrap = [
        tool_name for tool_name in _BOOTSTRAP_TOOLS if tool_name not in completed
    ]
    if missing_bootstrap:
        return missing_bootstrap
    if not state.target_facts:
        return list(_FIRST_INVESTIGATION_TOOLS)

    routes = remaining_claim_hypothesis_routes(state)
    names = list(dict.fromkeys(route.split(":", 1)[0] for route in routes))
    if _stoppable_task_ids(state):
        names.append("stop_route")
    return list(dict.fromkeys(names))


def _stoppable_task_ids(
    state: ImageOnlyInvestigationState,
) -> tuple[str, ...]:
    """Return only route IDs that the control reducer can close now.

    ``stop_route`` is exposed through a dynamic native schema.  Giving Gemini a
    task that still has a pending page or reference candidate guarantees a
    rejected turn, so use the reducer itself as the single source of truth when
    constructing that schema.
    """

    if state.pending_archive_read_ids:
        return ()
    task_ids: list[str] = []
    for task in state.tasks:
        if task.status not in {"active", "pending"}:
            continue
        preflight = apply_unified_stop_route(
            state.model_copy(deep=True),
            task_id=task.task_id,
            rationale="Schema preflight only; no state is persisted.",
            function_call_id="stop-route-schema-preflight",
        )
        if preflight.get("accepted", False):
            task_ids.append(task.task_id)
    return tuple(task_ids)


def _task_ids_for_tool(
    state: ImageOnlyInvestigationState,
    *,
    tool_name: str,
) -> list[str]:
    task_ids: list[str] = []
    for route in remaining_claim_hypothesis_routes(state):
        parts = route.split(":", 2)
        if not parts or parts[0] != tool_name or len(parts) < 2:
            continue
        task_id = (
            parts[2]
            if tool_name == "reverse_image_search" and len(parts) == 3
            else parts[1]
        )
        if task_id:
            task_ids.append(task_id)
    return list(dict.fromkeys(task_ids))


def _candidate_urls_for_tool(
    state: ImageOnlyInvestigationState,
    *,
    tool_name: str,
    task_id: str,
) -> list[str]:
    """Return the current runtime-owned URL candidates for one task/tool."""

    if tool_name not in {"visit", "compare_with_reference"}:
        return []
    prefix = f"{tool_name}:{task_id}:"
    candidates: list[str] = []
    for route in remaining_claim_hypothesis_routes(
        state,
        task_ids={task_id},
    ):
        if not route.startswith(prefix):
            continue
        value = route[len(prefix):].strip()
        if value and canonicalize_url(value):
            candidates.append(value)
    return list(dict.fromkeys(candidates))


def _annotate_runtime_url_candidates(
    tool: BaseTool,
    *,
    candidates: Sequence[str],
) -> None:
    """Explain dynamic URL ownership without sending a fragile enum."""

    if tool.name not in {"visit", "compare_with_reference"} or not candidates:
        return
    property_name = "url" if tool.name == "visit" else "reference_url"
    properties = tool.parameters.setdefault("properties", {})
    property_schema = properties.get(property_name)
    if not isinstance(property_schema, dict):
        return
    description = str(property_schema.get("description", "")).strip()
    candidate_hint = (
        " Runtime-owned candidates for the active task are: "
        + "; ".join(candidates[:8])
        + ". Use only one of these exact candidates; the runtime rejects "
        "stale or invented URLs."
    )
    if candidate_hint not in description:
        property_schema["description"] = description + candidate_hint


def _base_parameters(
    tool: BaseTool,
    *,
    task_ids: Sequence[str] = (),
    require_initial_intent: bool = False,
) -> dict[str, Any]:
    parameters = deepcopy(tool.parameters or {})
    parameters.setdefault("type", "object")
    properties = parameters.setdefault("properties", {})
    required = list(parameters.get("required", []) or [])
    if task_ids:
        properties["task_id"] = {
            "type": "string",
            "enum": list(task_ids),
            "description": (
                "Runtime-owned active route/task ID advanced by this tool call. "
                "Use one value from the supplied set exactly."
            ),
        }
        required.append("task_id")
    if require_initial_intent:
        properties["investigation_intent"] = _investigation_intent_tool_schema()
        properties["investigation_intent"]["description"] = (
            "Required only for this first non-bootstrap investigation action. "
            "It states one image-grounded target and two or three materially "
            "different candidate routes. The first route must include this "
            "actual tool call. The runtime validates and persists all routes, "
            "while the provider tool never receives this field."
        )
        required.append("investigation_intent")
    parameters["required"] = list(dict.fromkeys(required))
    return parameters


@dataclass
class UnifiedReactToolAdapter(BaseTool):
    """Expose adapter fields to the policy while shielding mature tools."""

    delegate: BaseTool = field(repr=False)
    task_ids: tuple[str, ...] = ()
    require_initial_intent: bool = False
    image_path: str = ""
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = self.delegate.name
        self.description = self.delegate.description
        self.parameters = _base_parameters(
            self.delegate,
            task_ids=self.task_ids,
            require_initial_intent=self.require_initial_intent,
        )

    @staticmethod
    def _provider_args(params: Mapping[str, Any]) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in dict(params).items()
            if str(key) not in _ADAPTER_ONLY_FIELDS
            and not str(key).startswith("__unified_")
        }

    def _bound_delegate(self) -> BaseTool:
        """Bind image-path tools to this request without mutating shared tools.

        ``StageRunner`` owns one adapter per case and writes ``image_path`` on
        it.  Mature comparison/anomaly tools keep their input image as an
        instance attribute, so forwarding through the adapter must preserve that
        contract.  A shallow copy makes the mutable path request-local while
        intentionally sharing the delegate's existing clients and caches.
        """

        if not self.image_path or not hasattr(self.delegate, "image_path"):
            return self.delegate
        bound = copy(self.delegate)
        bound.image_path = self.image_path
        return bound

    def call(self, params: Dict[str, Any]) -> Any:
        return self._bound_delegate().call(self._provider_args(params))

    async def call_async(self, params: Dict[str, Any]) -> Any:
        delegate = self._bound_delegate()
        call_async = getattr(delegate, "call_async", None)
        provider_args = self._provider_args(params)
        if callable(call_async):
            result = call_async(provider_args)
            if inspect.isawaitable(result):
                return await result
            return result
        return await asyncio.to_thread(delegate.call, provider_args)

    def set_source_access_policy(self, policy: Any) -> None:
        setter = getattr(self.delegate, "set_source_access_policy", None)
        if callable(setter):
            setter(policy)


@dataclass
class StopRouteTool(BaseTool):
    """Runtime-only control action; it never contacts an external provider."""

    name: str = "stop_route"
    description: str = (
        "Close exactly one active route that has no pending page/reference "
        "inspection and no worthwhile remaining step. This cannot decide a "
        "verdict or affect other routes."
    )
    parameters: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The one active route/task to close.",
                },
                "rationale": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Why this route alone has no useful next step.",
                },
            },
            "required": ["task_id", "rationale"],
        }
    )

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "success",
            "control": "stop_route",
            "task_id": str(params.get("task_id", "")).strip(),
        }


def build_unified_react_tools(
    state: ImageOnlyInvestigationState,
    all_tools: Mapping[str, BaseTool],
    *,
    excluded_tool_names: Iterable[str] = (),
) -> list[BaseTool]:
    """Build a fresh narrow model schema for the current unified-ReAct turn."""

    excluded = {
        str(name).strip()
        for name in excluded_tool_names
        if str(name).strip()
    }
    names = [
        name
        for name in available_unified_react_tool_names(state)
        if name not in excluded
    ]
    tools: list[BaseTool] = []
    if names == list(_BOOTSTRAP_TOOLS) or (
        not bootstrap_complete(state) and names
    ):
        return [
            UnifiedReactToolAdapter(delegate=all_tools[name])
            for name in names
            if name in all_tools
        ]
    if not state.target_facts:
        return [
            UnifiedReactToolAdapter(
                delegate=all_tools[name],
                require_initial_intent=True,
            )
            for name in names
            if name in all_tools
        ]

    for name in names:
        if name == "stop_route":
            task_ids = _stoppable_task_ids(state)
            if task_ids:
                control = StopRouteTool()
                control.parameters = _base_parameters(
                    control,
                    task_ids=task_ids,
                )
                tools.append(control)
            continue
        delegate = all_tools.get(name)
        if delegate is None:
            continue
        task_ids = tuple(_task_ids_for_tool(state, tool_name=name))
        if task_ids:
            adapter = UnifiedReactToolAdapter(
                delegate=delegate,
                task_ids=task_ids,
            )
            candidate_urls = [
                candidate
                for task_id in task_ids
                for candidate in _candidate_urls_for_tool(
                    state,
                    tool_name=name,
                    task_id=task_id,
                )
            ]
            _annotate_runtime_url_candidates(
                adapter,
                candidates=list(dict.fromkeys(candidate_urls)),
            )
            tools.append(adapter)
    return tools


def validate_unified_react_action(
    state: ImageOnlyInvestigationState,
    *,
    tool_name: str,
    tool_args: Mapping[str, Any],
    source_access_policy: Any = None,
) -> str:
    """Validate action ownership and route eligibility before a real call."""

    allowed = set(available_unified_react_tool_names(state))
    if tool_name not in allowed:
        return f"{tool_name} is not available in the current unified-ReAct state"
    if not bootstrap_complete(state):
        return ""
    if not state.target_facts:
        raw_intent = tool_args.get("investigation_intent")
        try:
            intent = _parse_initial_investigation_intent(
                raw_intent,
                tool_name=tool_name,
            )
        except Exception as exc:
            return f"invalid investigation_intent: {exc}"
        return validate_initial_action_intent(
            state,
            intent,
            tool_name=tool_name,
            tool_args=tool_args,
            source_access_policy=source_access_policy,
        )
    if tool_name == "stop_route":
        task_id = str(tool_args.get("task_id", "")).strip()
        if not task_id:
            return "stop_route requires task_id"
        dry_run = apply_unified_stop_route(
            state.model_copy(deep=True),
            task_id=task_id,
            rationale=str(tool_args.get("rationale", "")).strip(),
            function_call_id="stop-route-preflight",
        )
        if not dry_run.get("accepted", False):
            return str(
                dry_run.get(
                    "rejected_reason",
                    "stop_route is not currently permitted",
                )
            )
        return ""

    task_id = str(tool_args.get("task_id", "")).strip()
    task = next((item for item in state.tasks if item.task_id == task_id), None)
    if task is None or task.status not in {"active", "pending"}:
        return "tool action must reference one active runtime task_id"
    if tool_name not in _task_tools(state, task_id):
        return f"{tool_name} is not currently executable for task_id={task_id}"
    if tool_name in {"visit", "compare_with_reference"}:
        candidate_urls = _candidate_urls_for_tool(
            state,
            tool_name=tool_name,
            task_id=task_id,
        )
        argument_name = "url" if tool_name == "visit" else "reference_url"
        raw_urls = tool_args.get(argument_name, [])
        if isinstance(raw_urls, str):
            raw_urls = [raw_urls]
        if not isinstance(raw_urls, list):
            raw_urls = []
        requested_urls = [
            canonicalize_url(str(value))
            for value in raw_urls
            if canonicalize_url(str(value))
        ]
        allowed_urls = {
            canonicalize_url(value)
            for value in candidate_urls
            if canonicalize_url(value)
        }
        if not requested_urls:
            return (
                f"{tool_name} requires one current runtime-owned "
                f"{argument_name} candidate"
            )
        stale = [
            value
            for value in requested_urls
            if value not in allowed_urls
        ]
        if stale:
            return (
                f"{tool_name} received a stale or unavailable "
                f"{argument_name}; choose only current candidates: "
                + ", ".join(candidate_urls[:8])
            )
    if tool_name == "text_search":
        query = tool_args.get("queries", "")
        if isinstance(query, list):
            query = query[0] if query else ""
        query = str(query).strip()
        violation = query_policy_violation(
            query,
            source_access_policy=source_access_policy,
        )
        if violation:
            return f"text_search query violates source policy: {violation}"
        for raw in state.attempted_routes:
            try:
                prior = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(prior, Mapping):
                continue
            if str(prior.get("task_id", "")) != task_id:
                continue
            if routes_semantically_equivalent(
                "text_search",
                {"task_id": task_id, "queries": query},
                str(prior.get("tool", "")),
                prior,
            ):
                return "text_search query duplicates an earlier route for this task"
    if tool_name == "reverse_image_search":
        branch = str(tool_args.get("branch", "")).strip().lower()
        if branch not in set(remaining_root_image_reverse_branches(state)):
            return "reverse_image_search branch is no longer available"
    return ""


def _task_tools(state: ImageOnlyInvestigationState, task_id: str) -> set[str]:
    allowed: set[str] = set()
    for route in remaining_claim_hypothesis_routes(state, task_ids={task_id}):
        parts = route.split(":", 2)
        if parts:
            allowed.add(parts[0])
    return allowed


def reduce_visual_bootstrap_action(
    state: ImageOnlyInvestigationState,
    *,
    step: Any,
    runtime_case: ImageOnlyRuntimeCase,
    perception: PerceptionReport,
) -> dict[str, Any]:
    """Reduce one model-selected scene/OCR action into the empty workspace."""

    tool_name = str(getattr(step, "tool_name", "")).strip()
    call_id = str(
        getattr(step, "metadata", {}).get("function_call_id", "")
    ).strip()
    if tool_name not in _BOOTSTRAP_TOOLS:
        raise ValueError(f"{tool_name!r} is not a visual bootstrap tool")
    try:
        _data, succeeded = parse_tool_result(
            str(getattr(step, "tool_result", "") or "")
        )
    except Exception:
        succeeded = False
    if not succeeded:
        message = str(getattr(step, "tool_result", "") or "tool failed")[:4000]
        state.unified_react_bootstrap_failures.append(
            UnifiedReactBootstrapFailure(
                tool_name=tool_name,
                function_call_id=call_id or f"bootstrap-{tool_name}",
                message=message,
            )
        )
        state.stop_reason = "engineering_error"
        return {
            "accepted": False,
            "phase": "visual_bootstrap",
            "failure": {
                "tool_name": tool_name,
                "function_call_id": call_id,
                "message": message,
            },
            "next_available_tools": [],
        }

    if tool_name not in state.unified_react_bootstrap_tools_completed:
        state.unified_react_bootstrap_tools_completed.append(tool_name)
    materialized = False
    if bootstrap_complete(state):
        bootstrap = build_visual_bootstrap(runtime_case, perception)
        state.brief = bootstrap.brief
        state.entities = list(bootstrap.entities)
        state.facts = list(bootstrap.facts)
        state.retrieval_anchors = list(bootstrap.retrieval_anchors)
        state.tasks = []
        materialized = True
    return {
        "accepted": True,
        "phase": "visual_bootstrap",
        "completed_tools": list(state.unified_react_bootstrap_tools_completed),
        "created_entity_ids": [
            item.entity_id for item in state.entities
        ] if materialized else [],
        "created_visual_fact_ids": [
            item.fact_id for item in state.facts
        ] if materialized else [],
        "created_retrieval_anchor_ids": [
            item.anchor_id for item in state.retrieval_anchors
        ] if materialized else [],
        "next_available_tools": available_unified_react_tool_names(state),
    }


def reduce_unified_react_action(
    state: ImageOnlyInvestigationState,
    *,
    step: Any,
    runtime_case: ImageOnlyRuntimeCase,
    source_access_policy: Any = None,
) -> dict[str, Any]:
    """Apply canonical state changes after one accepted unified-ReAct action."""

    tool_name = str(getattr(step, "tool_name", "")).strip()
    metadata = getattr(step, "metadata", {}) or {}
    tool_args = dict(getattr(step, "tool_args", {}) or {})
    call_id = str(metadata.get("function_call_id", "")).strip()
    if tool_name in _BOOTSTRAP_TOOLS:
        raise ValueError("visual bootstrap must use reduce_visual_bootstrap_action")
    if tool_name == "stop_route":
        update = apply_unified_stop_route(
            state,
            task_id=str(tool_args.get("task_id", "")).strip(),
            rationale=str(tool_args.get("rationale", "")).strip(),
            function_call_id=call_id or "stop-route",
        )
        if update.get("accepted"):
            progress = record_action_progress(state, update)
            update["progress"] = progress.model_dump(mode="json")
            update["next_available_tools"] = available_unified_react_tool_names(state)
        return update

    initial_update: dict[str, Any] = {}
    if not state.target_facts:
        intent = _parse_initial_investigation_intent(
            tool_args.get("investigation_intent"),
            tool_name=tool_name,
        )
        initial_update = apply_initial_action_intent(
            state,
            intent,
            tool_name=tool_name,
            tool_args=tool_args,
            source_access_policy=source_access_policy,
        )
        if not initial_update.get("accepted"):
            return initial_update
        tool_args["task_id"] = str(initial_update["task_id"])
        step.tool_args = tool_args
        metadata["accepted_investigation_intent"] = initial_update[
            "accepted_intent"
        ]
        metadata["runtime_bound_task_id"] = tool_args["task_id"]
        step.metadata = metadata

    update = record_tool_observation(
        state,
        step,
        image_sha256=runtime_case.image_sha256,
    )
    update["accepted"] = True
    progress = record_action_progress(state, update)
    update["progress"] = progress.model_dump(mode="json")
    update["next_available_tools"] = available_unified_react_tool_names(state)
    if initial_update:
        update["initial_intent"] = initial_update["accepted_intent"]
        update["created_target_fact_ids"] = initial_update.get(
            "accepted_fact_ids", []
        )
        update["created_task_ids"] = initial_update.get("accepted_task_ids", [])
    return update


def unified_react_delta(
    *,
    step: Any,
    update: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable, compact state-delta trace record for one action."""

    metadata = getattr(step, "metadata", {}) or {}
    return {
        "schema_version": "ifv-unified-react-delta-v1",
        "action_id": str(metadata.get("function_call_id", "")).strip(),
        "interaction_id": str(metadata.get("interaction_id", "")).strip(),
        "function_call_id": str(metadata.get("function_call_id", "")).strip(),
        "tool_name": str(getattr(step, "tool_name", "")).strip(),
        "validated_arguments": dict(getattr(step, "tool_args", {}) or {}),
        "accepted_investigation_intent": metadata.get(
            "accepted_investigation_intent"
        ),
        "state_update": dict(update),
    }

"""Unified image-grounded ReAct runtime.

The active agent keeps one small investigation memory instead of asking the
policy model to create a target/fact/route graph before it can investigate.
Mature tools keep their existing implementation contracts; the adapter below
only supplies their internal image and extraction context at execution time.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.orchestrator.evidence_policy import query_policy_violation
from src.orchestrator.source_access import SourceAccessPolicy
from src.orchestrator.source_provenance import canonicalize_url
from src.orchestrator.state import (
    ImageOnlyRuntimeCase,
    PerceptionReport,
)
from src.orchestrator.tool_result import parse_tool_result
from src.tools.base import BaseTool


REACT_RUNTIME_SCHEMA_VERSION = "ifv-unified-react-v1"
UNIFIED_REACT_RUNTIME_POLICY_VERSION = "unified-react-v1"
UNIFIED_REACT_RUNTIME_STAGE = "unified_react"
MAX_REACT_ACTIONS = 24

REACT_RUNTIME_TOOLS = (
    "perceive_scene",
    "ocr_with_position",
    "current_time",
    "reverse_image_search",
    "text_search",
    "visit",
    "compare_with_reference",
    "check_consistency",
    "analyze_visual_anomalies",
    "crop_and_inspect",
    "focused_visual_inspection",
    "count_objects",
)
_INTERNAL_FIELDS = {
    "image_input",
    "image_claim",
    "retrieval_goal",
    "visual_question_id",
    "active_fact",
    "evidence_context",
    "trace_stage",
    "trace_purpose",
    "source_evidence_id",
    "source_discovery_id",
    "before_understanding_version",
}
_URL_FIELDS = {
    "url",
    "reference_url",
    "source_page_url",
    "candidate_url",
    "reference_image_url",
}


class UnifiedReactState(BaseModel):
    """Compact, append-only investigation memory exposed to the policy."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = REACT_RUNTIME_SCHEMA_VERSION
    case_id: str = Field(min_length=1, max_length=200)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    objective: str = Field(
        default=(
            "Verify the factual content expressed by the image and decide "
            "whether the image should be labeled real or fake."
        ),
        min_length=1,
        max_length=800,
    )
    visual_memory: Dict[str, Any] = Field(default_factory=dict)
    discoveries: list[Dict[str, Any]] = Field(default_factory=list, max_length=160)
    evidence: list[Dict[str, Any]] = Field(default_factory=list, max_length=160)
    failures: list[Dict[str, Any]] = Field(default_factory=list, max_length=120)
    attempted_queries: list[str] = Field(default_factory=list, max_length=120)
    visited_urls: list[str] = Field(default_factory=list, max_length=160)
    attempted_actions: list[Dict[str, Any]] = Field(
        default_factory=list,
        max_length=160,
    )
    recent_actions: list[Dict[str, Any]] = Field(
        default_factory=list,
        max_length=24,
    )
    open_questions: list[str] = Field(default_factory=list, max_length=12)
    current_focus: str = Field(default="", max_length=800)
    action_count: int = Field(default=0, ge=0, le=MAX_REACT_ACTIONS)
    no_gain_streak: int = Field(default=0, ge=0, le=MAX_REACT_ACTIONS)
    stop_reason: str = Field(default="", max_length=200)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


def _compact(value: Any, *, depth: int = 0, max_string: int = 1600) -> Any:
    """Bound a provider payload before it enters the model-visible memory."""

    if depth > 4:
        return "[nested content omitted]"
    if isinstance(value, str):
        text = value.strip()
        if len(text) <= max_string:
            return text
        return text[: max_string - 1].rstrip() + "…"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for raw_key, child in list(value.items())[:80]:
            key = str(raw_key)
            if key.casefold() in {
                "data_url",
                "image_input",
                "base64",
                "content_bytes",
                "raw_html",
                "html",
            }:
                result[key] = "[omitted media/content]"
            else:
                result[key] = _compact(
                    child,
                    depth=depth + 1,
                    max_string=max_string,
                )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _compact(item, depth=depth + 1, max_string=max_string)
            for item in list(value)[:32]
        ]
    return _compact(str(value), depth=depth + 1, max_string=max_string)


def _one_line(value: Any, limit: int = 800) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return f"{prefix}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def new_unified_react_runtime_state(
    runtime_case: ImageOnlyRuntimeCase,
) -> UnifiedReactState:
    return UnifiedReactState(
        case_id=runtime_case.case_id,
        image_sha256=runtime_case.image_sha256,
    )


def available_unified_react_runtime_tools(
    state: UnifiedReactState,
) -> list[str]:
    if state.action_count >= MAX_REACT_ACTIONS:
        return ["finish_investigation"]
    return [
        name
        for name in REACT_RUNTIME_TOOLS
        if name not in _exhausted_tools(state)
    ] + ["finish_investigation"]


def _exhausted_tools(state: UnifiedReactState) -> set[str]:
    limits = {
        "current_time": 1,
        "reverse_image_search": 2,
        "text_search": 16,
        "visit": 16,
        "compare_with_reference": 6,
        "check_consistency": 3,
        "analyze_visual_anomalies": 3,
        "crop_and_inspect": 4,
        "focused_visual_inspection": 4,
        "count_objects": 2,
    }
    counts: Dict[str, int] = {}
    for item in state.attempted_actions:
        name = str(item.get("tool_name", "")).strip()
        if name:
            counts[name] = counts.get(name, 0) + 1
    return {
        name for name, limit in limits.items() if counts.get(name, 0) >= limit
    }


def is_unified_react_runtime_budget_action(step: Mapping[str, Any]) -> bool:
    return (
        str(step.get("stage", "")).strip() == UNIFIED_REACT_RUNTIME_STAGE
        and str(step.get("action_type", "")).strip() == "tool_call"
        and str(step.get("tool_name", "")).strip() != "finish_investigation"
    )


def _public_parameters(delegate: BaseTool) -> Dict[str, Any]:
    parameters = copy.deepcopy(delegate.parameters or {})
    properties = parameters.setdefault("properties", {})
    for name in _INTERNAL_FIELDS:
        properties.pop(name, None)
    required = [
        name for name in parameters.get("required", []) or []
        if name not in _INTERNAL_FIELDS
    ]
    name = str(delegate.name).strip()
    if name == "visit":
        properties["question"] = {
            "type": "string",
            "description": (
                "What concrete image fact or relationship this action is "
                "checking on the selected page."
            ),
        }
        properties["context"] = {
            "type": "string",
            "description": "Short relevant context from the current investigation.",
        }
    if name == "focused_visual_inspection":
        properties["question"] = {
            "type": "string",
            "description": "One focused visual question about the original image.",
        }
        properties["expected_property"] = {
            "type": "string",
            "description": "The concrete visible property to inspect.",
        }
        properties["scope"] = {
            "type": "string",
            "enum": ["subject", "relation", "scene", "text", "integrity"],
        }
        properties["anchor_regions"] = {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
            "maxItems": 4,
        }
    if name == "finish_investigation":
        return parameters
    parameters["required"] = list(dict.fromkeys(required))
    parameters["additionalProperties"] = False
    return parameters


@dataclass
class RuntimeToolAdapter(BaseTool):
    """Hide legacy/internal tool fields from the active policy schema."""

    delegate: BaseTool = field(repr=False)
    state: UnifiedReactState = field(repr=False)
    image_path: str = ""
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = self.delegate.name
        self.description = self.delegate.description
        if self.name == "visit":
            self.description = (
                "Inspect a selected web candidate for the concrete image "
                "fact or relationship in the current question."
            )
        elif self.name == "focused_visual_inspection":
            self.description = (
                "Reinspect the original image to answer one concrete visual "
                "question raised by the latest observation."
            )
        self.parameters = _public_parameters(self.delegate)

    def _bound_delegate(self) -> BaseTool:
        if not self.image_path or not hasattr(self.delegate, "image_path"):
            return self.delegate
        bound = copy.copy(self.delegate)
        bound.image_path = self.image_path
        return bound

    def _provider_args(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        args = {
            str(key): value
            for key, value in dict(params).items()
            if str(key) not in _INTERNAL_FIELDS
        }
        name = self.name
        focus = (
            _one_line(
                args.get("question")
                or args.get("context")
                or args.get("focus")
                or args.get("aspect")
                or self.state.current_focus
            )
            or "Check the most relevant factual detail in the image."
        )
        if name == "visit":
            args.pop("question", None)
            context = args.pop("context", "")
            args["image_claim"] = focus
            args["retrieval_goal"] = _one_line(context or focus)
        elif name == "focused_visual_inspection":
            args["image_input"] = self.image_path
            args.setdefault("question", focus)
            args.setdefault("expected_property", focus)
            args.setdefault("scope", "scene")
            args.setdefault("anchor_regions", [])
            args["visual_question_id"] = _stable_id(
                "visual-question",
                self.state.case_id,
                self.state.action_count,
                focus,
            )
            args["active_fact"] = focus
            args["evidence_context"] = _one_line(
                json.dumps(
                    {
                        "recent_evidence": self.state.evidence[-4:],
                        "recent_discoveries": self.state.discoveries[-4:],
                    },
                    ensure_ascii=False,
                ),
                5000,
            )
            args["trace_stage"] = UNIFIED_REACT_RUNTIME_STAGE
            args["trace_purpose"] = "image_grounded_react_observation"
        elif name in {
            "perceive_scene",
            "reverse_image_search",
            "check_consistency",
            "count_objects",
            "ocr_with_position",
            "crop_and_inspect",
        }:
            args["image_input"] = self.image_path
        if name == "compare_with_reference":
            reference = canonicalize_url(str(args.get("reference_url", "")))
            for item in reversed(self.state.discoveries):
                if canonicalize_url(str(item.get("reference_image_url", ""))) == reference:
                    page = str(item.get("candidate_url", "")).strip()
                    if page and not str(args.get("source_page_url", "")).strip():
                        args["source_page_url"] = page
                    break
        return args

    def call(self, params: Dict[str, Any]) -> Any:
        if self.name == "finish_investigation":
            return {
                "status": "success",
                "control": "finish_investigation",
                "rationale": _one_line(params.get("rationale"), 1200),
            }
        return self._bound_delegate().call(self._provider_args(params))

    async def call_async(self, params: Dict[str, Any]) -> Any:
        if self.name == "finish_investigation":
            return self.call(params)
        delegate = self._bound_delegate()
        method = getattr(delegate, "call_async", None)
        args = self._provider_args(params)
        if callable(method):
            result = method(args)
            if hasattr(result, "__await__"):
                return await result
            return result
        import asyncio

        return await asyncio.to_thread(delegate.call, args)


@dataclass
class FinishInvestigationTool(BaseTool):
    name: str = "finish_investigation"
    description: str = (
        "End the investigation when the next available actions are not "
        "worthwhile. This does not choose the final real/fake verdict."
    )
    parameters: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "rationale": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Why further investigation is not worthwhile.",
                }
            },
            "required": ["rationale"],
            "additionalProperties": False,
        }
    )

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "success",
            "control": self.name,
            "rationale": _one_line(params.get("rationale"), 1200),
        }


def build_react_runtime_tools(
    state: UnifiedReactState,
    all_tools: Mapping[str, BaseTool],
    *,
    image_path: str = "",
    excluded_tool_names: Iterable[str] = (),
) -> list[BaseTool]:
    excluded = {str(name).strip() for name in excluded_tool_names}
    result: list[BaseTool] = []
    for name in available_unified_react_runtime_tools(state):
        if name in excluded:
            continue
        if name == "finish_investigation":
            result.append(FinishInvestigationTool())
            continue
        delegate = all_tools.get(name)
        if delegate is None:
            continue
        result.append(
            RuntimeToolAdapter(
                delegate=delegate,
                state=state,
                image_path=image_path,
            )
        )
    return result


def _requested_urls(tool_name: str, args: Mapping[str, Any]) -> list[str]:
    field_name = "reference_url" if tool_name == "compare_with_reference" else "url"
    raw = args.get(field_name, [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return list(
        dict.fromkeys(
            canonicalize_url(str(value))
            for value in raw
            if canonicalize_url(str(value))
        )
    )


def validate_react_action(
    state: UnifiedReactState,
    *,
    tool_name: str,
    tool_args: Mapping[str, Any],
    source_access_policy: SourceAccessPolicy | None = None,
) -> str:
    allowed = set(available_unified_react_runtime_tools(state))
    if tool_name not in allowed:
        return f"{tool_name} is not available in the current unified ReAct state"
    if tool_name == "finish_investigation":
        if not _one_line(tool_args.get("rationale")):
            return "finish_investigation requires a rationale"
        return ""
    if any(
        str(key) in {"claim_id", "task_id", "question_id", "__claim_id", "__question_id"}
        for key in tool_args
    ):
        return "claim/task/question IDs are not part of the unified ReAct contract"
    policy = source_access_policy or SourceAccessPolicy()
    if tool_name == "text_search":
        query = tool_args.get("queries", "")
        if isinstance(query, list):
            query = query[0] if query else ""
        query = _one_line(query, 1200)
        if not query:
            return "text_search requires one non-empty query"
        violation = query_policy_violation(query, source_access_policy=policy)
        if violation:
            return f"text_search query violates source policy: {violation}"
        normalized = " ".join(query.casefold().split())
        if any(normalized == " ".join(item.casefold().split()) for item in state.attempted_queries):
            return "text_search query duplicates an earlier query"
    if tool_name in {"visit", "compare_with_reference"}:
        urls = _requested_urls(tool_name, tool_args)
        if not urls:
            return f"{tool_name} requires a non-empty URL"
        if any(not policy.allows(url) for url in urls):
            return f"{tool_name} URL is blocked by the active source policy"
        if tool_name == "visit":
            visited = set(state.visited_urls)
            if any(url in visited for url in urls):
                return "visit received a URL already inspected in this episode"
        else:
            candidates = {
                canonicalize_url(str(item.get("reference_image_url", "")))
                for item in state.discoveries
            }
            if any(url not in candidates for url in urls):
                return "compare_with_reference requires an unverified image candidate from prior search"
    if tool_name == "reverse_image_search":
        branch = str(tool_args.get("branch", "lens")).strip().lower()
        if branch not in {"lens", "semantic"}:
            return "reverse_image_search branch must be lens or semantic"
        key = f"reverse_image_search:{branch}"
        if any(str(item.get("dedup_key", "")) == key for item in state.attempted_actions):
            return "reverse_image_search branch duplicates an earlier branch"
    return ""


def _failure_code(error: str) -> tuple[str, bool]:
    lowered = str(error).casefold()
    if any(token in lowered for token in ("ssl", "captcha", "timeout", "timed out", "download", "unavailable", "403", "404", "429")):
        return "external_unavailable", True
    if any(token in lowered for token in ("schema", "contract", "invalid_argument", "malformed", "required field")):
        return "engineering_error", False
    return "provider_error", True


def _append_failure(
    state: UnifiedReactState,
    *,
    tool_name: str,
    call_id: str,
    error: str,
) -> Dict[str, Any]:
    code, recoverable = _failure_code(error)
    item = {
        "failure_id": _stable_id("failure", state.case_id, call_id, error),
        "tool_name": tool_name,
        "function_call_id": call_id,
        "code": code,
        "recoverable": recoverable,
        "error": _one_line(error, 2400),
    }
    state.failures.append(item)
    return item


def _iter_candidate_rows(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for key in (
        "queries",
        "results",
        "lens_results",
        "semantic_results",
        "items",
        "visits",
        "pages",
    ):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping):
                yield row
            if key == "queries" and isinstance(row, Mapping):
                nested = row.get("results")
                if isinstance(nested, list):
                    for child in nested:
                        if isinstance(child, Mapping):
                            yield child


def _append_discoveries(
    state: UnifiedReactState,
    *,
    tool_name: str,
    call_id: str,
    payload: Mapping[str, Any],
) -> list[str]:
    ids: list[str] = []
    rows = list(_iter_candidate_rows(payload))
    if tool_name == "text_search":
        # Search results are deliberately Discovery only; snippets never become
        # Evidence merely because they contain matching words.
        for row in rows:
            url = str(row.get("url") or row.get("link") or "").strip()
            if not url:
                continue
            item = {
                "discovery_id": _stable_id("discovery", call_id, url),
                "tool_name": tool_name,
                "candidate_url": url,
                "reference_image_url": "",
                "title": _one_line(row.get("title"), 600),
                "snippet": _one_line(row.get("snippet"), 1000),
                "candidate_status": "unverified",
                "source_query": _one_line(
                    payload.get("query")
                    or payload.get("queries")
                    or "",
                    800,
                ),
            }
            if not any(
                str(old.get("candidate_url", "")).strip() == url
                and str(old.get("tool_name", "")).strip() == tool_name
                for old in state.discoveries
            ):
                state.discoveries.append(item)
            ids.append(item["discovery_id"])
    elif tool_name == "reverse_image_search":
        for row in rows:
            candidate_url = str(row.get("url") or row.get("link") or "").strip()
            image_url = str(
                row.get("image_url")
                or row.get("thumbnail")
                or row.get("imageUrl")
                or ""
            ).strip()
            if not candidate_url and not image_url:
                continue
            item = {
                "discovery_id": _stable_id(
                    "discovery",
                    call_id,
                    candidate_url,
                    image_url,
                ),
                "tool_name": tool_name,
                "candidate_url": candidate_url,
                "reference_image_url": image_url,
                "title": _one_line(row.get("title"), 600),
                "snippet": _one_line(row.get("snippet"), 1000),
                "candidate_status": "unverified",
                "match_status": "unverified",
            }
            if not any(
                str(old.get("candidate_url", "")) == candidate_url
                and str(old.get("reference_image_url", "")) == image_url
                for old in state.discoveries
            ):
                state.discoveries.append(item)
            ids.append(item["discovery_id"])
    return ids


def _evidence_class(payload: Mapping[str, Any]) -> str:
    explicit = str(
        payload.get("evidence_class")
        or payload.get("relation_stance")
        or ""
    ).strip()
    if explicit in {
        "decision_capable_support",
        "decision_capable_refute",
        "context_only",
        "irrelevant",
        "invalid",
    }:
        return explicit
    stance = str(payload.get("stance", "")).strip().lower()
    directness = str(payload.get("directness", "")).strip().lower()
    relevance = str(payload.get("relevance", "")).strip().lower()
    if stance in {"support", "supports"} and directness == "direct":
        return "decision_capable_support"
    if stance in {"refute", "contradict", "contradicts"} and directness == "direct":
        return "decision_capable_refute"
    if relevance in {"low", "none", "irrelevant"}:
        return "irrelevant"
    return "context_only"


def _append_evidence(
    state: UnifiedReactState,
    *,
    tool_name: str,
    call_id: str,
    payload: Mapping[str, Any],
) -> list[str]:
    if tool_name in {"text_search", "reverse_image_search"}:
        return []
    rows: list[Mapping[str, Any]] = []
    if tool_name == "visit":
        rows.extend(
            row for row in _iter_candidate_rows(payload)
            if any(
                str(row.get(key, "")).strip()
                for key in ("evidence", "summary", "text", "content", "excerpt")
            )
        )
    elif tool_name in {
        "focused_visual_inspection",
        "crop_and_inspect",
        "check_consistency",
        "analyze_visual_anomalies",
        "compare_with_reference",
        "ocr_with_position",
        "perceive_scene",
        "crop_and_search",
    }:
        rows.append(payload)
    evidence_ids: list[str] = []
    for index, row in enumerate(rows[:12]):
        excerpt = _one_line(
            row.get("excerpt")
            or row.get("evidence")
            or row.get("summary")
            or row.get("details")
            or row.get("description")
            or row.get("overall_observation")
            or row.get("full_text")
            or row.get("text")
            or json.dumps(_compact(row), ensure_ascii=False),
            2400,
        )
        if not excerpt:
            continue
        evidence_kind = (
            "web_span"
            if tool_name == "visit"
            else "reference_comparison"
            if tool_name == "compare_with_reference"
            else "image_observation"
        )
        item = {
            "evidence_id": _stable_id("evidence", call_id, index, excerpt),
            "tool_name": tool_name,
            "function_call_id": call_id,
            "status": "success",
            "successful_call": True,
            "evidence_kind": evidence_kind,
            "source_url": _one_line(
                row.get("url")
                or row.get("selected_url")
                or row.get("source_url")
                or "",
                2400,
            ),
            "excerpt": excerpt,
            "evidence_class": _evidence_class(row),
            "match_status": (
                str(row.get("candidate_match_status", "")).strip()
                or "not_applicable"
            ),
        }
        state.evidence.append(item)
        evidence_ids.append(item["evidence_id"])
    return evidence_ids


def _focus_from_args(tool_name: str, args: Mapping[str, Any]) -> str:
    values = [
        args.get("question"),
        args.get("focus_question"),
        args.get("expected_property"),
        args.get("focus"),
        args.get("goal"),
        args.get("queries"),
        args.get("aspect"),
        args.get("target_object"),
    ]
    for value in values:
        text = _one_line(value, 800)
        if text:
            return text
    return tool_name


def update_react_visual_memory(
    state: UnifiedReactState,
    perception: PerceptionReport,
) -> None:
    """Replace the bounded visual index with the latest merged observation."""

    state.visual_memory = {
        "scene_description": _one_line(perception.scene_description, 1800),
        "image_type": perception.image_type,
        "entities": [
            item.model_dump(mode="json") for item in perception.entities[:24]
        ],
        "relations": [
            item.model_dump(mode="json") for item in perception.relations[:32]
        ],
        "text_regions": [
            item.model_dump(mode="json")
            for item in perception.text_regions[:32]
        ],
        "notable_details": list(perception.notable_details[:16]),
        "uncertainties": list(perception.uncertainties[:12]),
    }


def reduce_react_action(
    state: UnifiedReactState,
    *,
    tool_name: str,
    tool_args: Mapping[str, Any],
    call_id: str,
    serialized_result: str,
    perception: PerceptionReport | None = None,
) -> Dict[str, Any]:
    try:
        payload, succeeded = parse_tool_result(serialized_result)
    except Exception as exc:
        payload = {"status": "error", "error": str(exc)}
        succeeded = False
    if perception is not None:
        update_react_visual_memory(state, perception)
    state.attempted_actions.append(
        {
            "tool_name": tool_name,
            "function_call_id": call_id,
            "dedup_key": (
                f"reverse_image_search:{str(tool_args.get('branch', '')).strip().lower()}"
                if tool_name == "reverse_image_search"
                else _stable_id("action", tool_name, tool_args)
            ),
        }
    )
    focus = _focus_from_args(tool_name, tool_args)
    state.current_focus = focus
    queries = tool_args.get("queries", "")
    if isinstance(queries, list):
        queries = queries[0] if queries else ""
    query = _one_line(queries, 1200)
    if query and query not in state.attempted_queries:
        state.attempted_queries.append(query)
    for url in _requested_urls(tool_name, tool_args):
        if tool_name == "visit" and url not in state.visited_urls:
            state.visited_urls.append(url)

    if tool_name == "finish_investigation":
        state.stop_reason = "model_finished"
        state.recent_actions.append(
            {
                "tool_name": tool_name,
                "function_call_id": call_id,
                "status": "success" if succeeded else "error",
                "observation": _compact(payload),
            }
        )
        return {
            "accepted": succeeded,
            "finished": succeeded,
            "next_available_tools": [],
        }

    state.action_count = min(MAX_REACT_ACTIONS, state.action_count + 1)

    if not succeeded:
        failure = _append_failure(
            state,
            tool_name=tool_name,
            call_id=call_id,
            error=str(payload.get("error", "tool failed")),
        )
        state.no_gain_streak += 1
        state.open_questions = list(
            dict.fromkeys([*state.open_questions, f"{tool_name}: retry or choose another route"])
        )[-12:]
        state.recent_actions.append(
            {
                "tool_name": tool_name,
                "function_call_id": call_id,
                "status": "error",
                "observation": _compact(payload),
                "failure_id": failure["failure_id"],
            }
        )
        return {
            "accepted": True,
            "tool_success": False,
            "failure": failure,
            "next_available_tools": available_unified_react_runtime_tools(state),
        }

    discovery_ids = _append_discoveries(
        state,
        tool_name=tool_name,
        call_id=call_id,
        payload=payload,
    )
    evidence_ids = _append_evidence(
        state,
        tool_name=tool_name,
        call_id=call_id,
        payload=payload,
    )
    gain = bool(discovery_ids or evidence_ids)
    state.no_gain_streak = 0 if gain else state.no_gain_streak + 1
    limitations = payload.get("limitations")
    if isinstance(limitations, list):
        state.open_questions = list(
            dict.fromkeys(
                [
                    *state.open_questions,
                    *(_one_line(item, 500) for item in limitations if _one_line(item)),
                ]
            )
        )[-12:]
    state.recent_actions.append(
        {
            "tool_name": tool_name,
            "function_call_id": call_id,
            "status": "success",
            "focus": focus,
            "discovery_ids": discovery_ids,
            "evidence_ids": evidence_ids,
            "observation": _compact(payload),
        }
    )
    return {
        "accepted": True,
        "tool_success": True,
        "created_discovery_ids": discovery_ids,
        "created_evidence_ids": evidence_ids,
        "substantive_gain": gain,
        "next_available_tools": available_unified_react_runtime_tools(state),
    }


def render_react_runtime_context(
    state: UnifiedReactState,
) -> str:
    payload = {
        "phase": "unified_react_investigation",
        "objective": state.objective,
        "original_image": {
            "attached_to_this_request": True,
            "note": "Use the pixels together with the structured memory; it is not a substitute for the image.",
        },
        "visual_memory": _compact(state.visual_memory, max_string=1800),
        "current_focus": state.current_focus,
        "discoveries": [
            _compact(item, max_string=1000)
            for item in state.discoveries[-16:]
        ],
        "evidence": [
            _compact(item, max_string=1800)
            for item in state.evidence[-16:]
        ],
        "failures": [
            _compact(item, max_string=1000)
            for item in state.failures[-10:]
        ],
        "attempted_queries": state.attempted_queries[-32:],
        "visited_urls": state.visited_urls[-32:],
        "recent_actions": [
            _compact(item, max_string=1400)
            for item in state.recent_actions[-8:]
        ],
        "open_questions": state.open_questions[-12:],
        "budget": {
            "actions_used": state.action_count,
            "actions_remaining": max(0, MAX_REACT_ACTIONS - state.action_count),
            "no_gain_streak": state.no_gain_streak,
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def compile_react_judgment_basis(
    state: UnifiedReactState,
) -> Dict[str, Any]:
    return {
        "schema_version": "ifv-unified-judgment-basis-v1",
        "decision_mode": "bounded_binary_judgment",
        "objective": state.objective,
        "visual_memory": _compact(state.visual_memory, max_string=1800),
        "evidence_ids": [
            str(item.get("evidence_id"))
            for item in state.evidence[-40:]
            if str(item.get("evidence_id", "")).strip()
        ],
        "evidence": [
            _compact(item, max_string=2200) for item in state.evidence[-40:]
        ],
        "discoveries": [
            _compact(item, max_string=1200) for item in state.discoveries[-24:]
        ],
        "failures": [
            _compact(item, max_string=1000) for item in state.failures[-16:]
        ],
        "open_questions": state.open_questions[-12:],
        "attempted_queries": state.attempted_queries[-32:],
        "visited_urls": state.visited_urls[-32:],
        "action_count": state.action_count,
        "stop_reason": state.stop_reason,
    }


def render_react_judgment_context(
    state: UnifiedReactState,
    basis: Mapping[str, Any],
) -> str:
    payload = {
        "stage": "final_judgment",
        "objective": state.objective,
        "original_image": {
            "attached_to_this_request": True,
            "instruction": "Inspect the image again when a detail matters.",
        },
        "investigation": dict(basis),
        "output_requirements": {
            "verdict": ["real", "fake"],
            "report": [
                "headline",
                "claim_under_review",
                "verdict_summary",
                "key_findings",
                "evidence_summary",
                "remaining_uncertainties",
            ],
            "rules": [
                "Keep the claim under review faithful to what the image expresses.",
                "Use recorded observations and sources without inventing facts.",
                "A lack of evidence is uncertainty, not proof of fake.",
                "Visible artifacts or image quality alone are not a factual verdict.",
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

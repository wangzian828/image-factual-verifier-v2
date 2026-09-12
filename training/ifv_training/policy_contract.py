from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from . import _repo_import  # noqa: F401  (adds the repository root to sys.path)

from src.orchestrator.investigation_models import RawHistoryJudgmentOutput


ANSWER_RE = re.compile(r"(<answer>\s*)(.*?)(\s*</answer>)", re.DOTALL | re.IGNORECASE)
STAGE_CONTROL_RE = re.compile(
    r"(?:\n\n(?P<legacy>\{\s*\"stage\"\s*:.*\})\s*$|"
    r"<stage_control>(?P<tagged>.*?)</stage_control>)",
    re.DOTALL,
)


class PolicyCausalContractError(ValueError):
    """A model-visible row cannot be projected to the live runtime contract."""

    def __init__(self, message: str, *, codes: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.codes = tuple(dict.fromkeys(str(code) for code in codes if str(code)))


@dataclass(frozen=True)
class ToolProjection:
    name: str
    arguments: dict[str, Any]
    status: str
    removed_arguments: tuple[str, ...] = ()
    issue_code: str = ""
    issue_fields: tuple[str, ...] = ()

    @property
    def executable(self) -> bool:
        return self.status in {"exact", "unknown_arguments_removed"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _json_object(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a JSON object or JSON string")
    parsed = json.loads(value)
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} must decode to a JSON object")
    return dict(parsed)


def _normalize_schema(value: Any) -> Any:
    """Mirror StageRunner._normalize_native_schema without importing providers."""

    if isinstance(value, list):
        return [_normalize_schema(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    normalized = {str(key): _normalize_schema(child) for key, child in value.items()}
    schema_type = normalized.get("type")
    if isinstance(schema_type, list):
        if "array" in schema_type:
            normalized["type"] = "array"
            normalized.setdefault("items", {"type": "string"})
        elif "string" in schema_type:
            normalized["type"] = "string"
        elif schema_type:
            normalized["type"] = schema_type[0]
    if normalized.get("type") == "array":
        normalized.setdefault("items", {"type": "string"})
    return normalized


def _value_issue(value: Any, spec: Mapping[str, Any], *, path: str) -> tuple[str, str]:
    """Return the first runtime-equivalent schema issue and its field path."""

    expected = spec.get("type")
    if expected == "string":
        if not isinstance(value, str):
            return "invalid_type", path
        minimum = spec.get("minLength")
        maximum = spec.get("maxLength")
        if isinstance(minimum, int) and len(value.strip()) < minimum:
            return "invalid_value", path
        if isinstance(maximum, int) and len(value) > maximum:
            return "invalid_value", path
    if expected == "array":
        if not isinstance(value, list):
            return "invalid_type", path
        minimum = spec.get("minItems")
        maximum = spec.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            return "invalid_value", path
        if isinstance(maximum, int) and len(value) > maximum:
            return "invalid_value", path
        item_spec = spec.get("items")
        if isinstance(item_spec, Mapping):
            for index, item in enumerate(value):
                issue = _value_issue(item, item_spec, path=f"{path}[]")
                if issue[0]:
                    return issue
    if expected == "object":
        if not isinstance(value, Mapping):
            return "invalid_type", path
        properties = _mapping(spec.get("properties"))
        if spec.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                return "nested_unknown_argument", f"{path}.{unknown[0]}"
        for name in spec.get("required", []) or []:
            if name not in value:
                return "nested_missing_required", f"{path}.{name}"
        for name, child in value.items():
            child_spec = properties.get(name)
            if isinstance(child_spec, Mapping):
                issue = _value_issue(child, child_spec, path=f"{path}.{name}")
                if issue[0]:
                    return issue
    if expected == "number" and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        return "invalid_type", path
    if expected == "integer" and (
        isinstance(value, bool) or not isinstance(value, int)
    ):
        return "invalid_type", path
    if expected == "boolean" and not isinstance(value, bool):
        return "invalid_type", path
    allowed = spec.get("enum")
    if isinstance(allowed, list) and value not in allowed:
        return "invalid_enum", path
    return "", ""


def _schema_issues(
    tool_name: str,
    arguments: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> list[tuple[str, str]]:
    schema = _normalize_schema(schema)
    properties = _mapping(schema.get("properties"))
    issues: list[tuple[str, str]] = []
    for name in sorted(set(arguments) - set(properties)):
        issues.append(("unknown_argument", name))
    for name in schema.get("required", []) or []:
        if name in arguments:
            continue
        if (
            tool_name == "crop_and_inspect"
            and str(arguments.get("visual_question_id", "")).strip()
            and name in {"bbox", "focus_question"}
        ):
            continue
        issues.append(("missing_required", str(name)))
    for name, value in arguments.items():
        spec = properties.get(name)
        if isinstance(spec, Mapping):
            issue = _value_issue(value, spec, path=str(name))
            if issue[0]:
                issues.append(issue)
    return issues


def _tool_functions(tools: Any) -> list[dict[str, Any]]:
    if isinstance(tools, str):
        if not tools.strip():
            return []
        tools = json.loads(tools)
    if not isinstance(tools, list):
        raise ValueError("tools must be a JSON list")
    result: list[dict[str, Any]] = []
    for item in tools:
        if not isinstance(item, Mapping):
            continue
        function = item.get("function") if isinstance(item.get("function"), Mapping) else item
        name = str(function.get("name", "")).strip()
        if not name:
            continue
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(function.get("description", "")),
                    "parameters": _normalize_schema(function.get("parameters", {})),
                },
            }
        )
    return result


def _schema_variants(tools: Any) -> dict[str, list[dict[str, Any]]]:
    variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for item in _tool_functions(tools):
        function = item["function"]
        name = function["name"]
        schema = dict(_mapping(function.get("parameters")))
        fingerprint = json.dumps(schema, ensure_ascii=False, sort_keys=True)
        if fingerprint not in seen[name]:
            variants[name].append(schema)
            seen[name].add(fingerprint)
    return dict(variants)


def project_tool_call(call: Mapping[str, Any], tools: Any) -> ToolProjection:
    name = str(call.get("name", "")).strip()
    if not name:
        return ToolProjection("", {}, "unrepairable", issue_code="missing_tool_name")
    try:
        arguments = _json_object(call.get("arguments", {}), label="tool arguments")
    except (TypeError, ValueError, json.JSONDecodeError):
        return ToolProjection(name, {}, "unrepairable", issue_code="malformed_arguments")
    variants = _schema_variants(tools).get(name, [])
    if not variants:
        return ToolProjection(name, arguments, "unrepairable", issue_code="unknown_tool")

    for schema in variants:
        if not _schema_issues(name, arguments, schema):
            return ToolProjection(name, arguments, "exact")

    repair_candidates: list[tuple[int, dict[str, Any], tuple[str, ...]]] = []
    all_issues: list[list[tuple[str, str]]] = []
    for schema in variants:
        properties = set(_mapping(schema.get("properties")))
        removed = tuple(sorted(set(arguments) - properties))
        sanitized = {key: value for key, value in arguments.items() if key in properties}
        issues = _schema_issues(name, sanitized, schema)
        all_issues.append(_schema_issues(name, arguments, schema))
        if removed and not issues:
            repair_candidates.append((len(removed), sanitized, removed))
    if repair_candidates:
        _, sanitized, removed = min(
            repair_candidates,
            key=lambda item: (item[0], item[2]),
        )
        return ToolProjection(
            name,
            sanitized,
            "unknown_arguments_removed",
            removed_arguments=removed,
        )

    best = min(all_issues, key=lambda issues: (len(issues), issues)) if all_issues else []
    issue_code = best[0][0] if best else "schema_mismatch"
    issue_fields = tuple(field for _, field in best)
    return ToolProjection(
        name,
        arguments,
        "unrepairable",
        issue_code=issue_code,
        issue_fields=issue_fields,
    )


def _strip_dynamic_enums(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_dynamic_enums(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    return {
        str(key): _strip_dynamic_enums(child)
        for key, child in value.items()
        if key != "enum"
    }


def canonicalize_tool_schemas(tools: Any) -> tuple[str, dict[str, Any]]:
    """Collapse duplicate dynamic schemas to one static schema per tool name."""

    functions = _tool_functions(tools)
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in functions:
        by_name[item["function"]["name"]].append(item)
    output: list[dict[str, Any]] = []
    duplicate_names = 0
    conflicting_names: list[str] = []
    for name, items in by_name.items():
        duplicate_names += max(0, len(items) - 1)
        schemas = [dict(_mapping(item["function"].get("parameters"))) for item in items]
        stripped = [_strip_dynamic_enums(schema) for schema in schemas]
        properties: dict[str, Any] = {}
        property_conflict = False
        for schema in stripped:
            for field, spec in _mapping(schema.get("properties")).items():
                if field not in properties:
                    properties[str(field)] = deepcopy(spec)
                elif properties[field] != spec:
                    property_conflict = True
        if property_conflict:
            conflicting_names.append(name)
            continue
        required_sets = [set(schema.get("required", []) or []) for schema in stripped]
        required = sorted(set.intersection(*required_sets)) if required_sets else []
        output.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": items[0]["function"].get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        )
    if conflicting_names:
        raise PolicyCausalContractError(
            "conflicting static schemas cannot be merged",
            codes=["conflicting_tool_schemas"],
        )
    stats = {
        "input_schema_count": len(functions),
        "output_schema_count": len(output),
        "duplicate_schema_entries_removed": duplicate_names,
    }
    return json.dumps(output, ensure_ascii=False, separators=(",", ":")), stats


def extract_final_answer(content: str) -> dict[str, Any]:
    match = ANSWER_RE.search(content)
    if match is None:
        raise ValueError("final assistant message lacks an <answer> JSON block")
    payload = json.loads(match.group(2).strip())
    if not isinstance(payload, Mapping):
        raise ValueError("final <answer> payload must be a JSON object")
    return dict(payload)


def _replace_final_answer(content: str, payload: Mapping[str, Any]) -> str:
    match = ANSWER_RE.search(content)
    if match is None:
        raise ValueError("final assistant message lacks an <answer> JSON block")
    rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return content[: match.start(2)] + rendered + content[match.end(2) :]


def _raw_json_prefix(content: str) -> Any:
    text = content.strip()
    if text.startswith("<tool_response>"):
        text = text[len("<tool_response>") :].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(text)
        return value
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def successful_observation_ids(messages: Sequence[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    for message in messages:
        if str(message.get("role", "")) != "tool_response":
            continue
        payload = _raw_json_prefix(str(message.get("content", "")))
        locator = _mapping(_mapping(payload).get("observation_locator"))
        observation_id = str(locator.get("observation_id", "")).strip()
        succeeded = locator.get("tool_success") is True or locator.get("successful") is True
        if observation_id and succeeded:
            result.add(observation_id)
    return result


def _response_outcome(content: str) -> str:
    payload = _raw_json_prefix(content)
    value = _mapping(payload)
    result = value.get("result") if "result" in value else payload
    result_value = _mapping(result)
    status = str(result_value.get("status", value.get("status", ""))).strip().casefold()
    if status in {"error", "failed", "failure"}:
        return "error"
    if status in {"ok", "success", "completed"}:
        return "success"
    text = str(content).strip().casefold()
    if text.startswith("error") or '"status":"error"' in text.replace(" ", ""):
        return "error"
    return "unknown"


def project_final_observation_ids(
    messages: list[dict[str, str]],
) -> dict[str, int | bool]:
    if not messages or messages[-1].get("role") != "assistant":
        return {
            "valid_final_output": False,
            "original": 0,
            "retained": 0,
            "dropped_not_explicitly_visible": 0,
            "duplicate_ids_removed": 0,
        }
    try:
        payload = extract_final_answer(messages[-1]["content"])
        RawHistoryJudgmentOutput.model_validate(payload)
    except Exception:
        # Keep the complete historical trajectory available as context while
        # ensuring an invalid terminal contract is never a positive target.
        messages[-1]["loss"] = False
        return {
            "valid_final_output": False,
            "original": 0,
            "retained": 0,
            "dropped_not_explicitly_visible": 0,
            "duplicate_ids_removed": 0,
        }
    original = [
        str(item).strip()
        for item in payload.get("verdict_observation_ids", []) or []
        if str(item).strip()
    ]
    visible = successful_observation_ids(messages[:-1])
    retained: list[str] = []
    for item in original:
        if item in visible and item not in retained:
            retained.append(item)
    payload["verdict_observation_ids"] = retained
    messages[-1]["content"] = _replace_final_answer(messages[-1]["content"], payload)
    RawHistoryJudgmentOutput.model_validate(payload)
    return {
        "valid_final_output": True,
        "original": len(original),
        "retained": len(retained),
        "dropped_not_explicitly_visible": len(original) - len(retained),
        "duplicate_ids_removed": len(original) - len(set(original)),
    }


def project_converted_policy_row(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    projected = deepcopy(dict(row))
    messages = projected.get("messages")
    if not isinstance(messages, list):
        raise PolicyCausalContractError("messages must be a list", codes=["invalid_messages"])
    tools = projected.get("tools", "")
    call_counts: Counter[str] = Counter()
    removed_fields: Counter[str] = Counter()
    masked_call_counts: Counter[str] = Counter()
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "tool_call":
            continue
        try:
            call = _json_object(message.get("content", ""), label="tool call")
        except (TypeError, ValueError, json.JSONDecodeError):
            projection = ToolProjection(
                "", {}, "unrepairable", issue_code="malformed_tool_call"
            )
            call_counts[projection.status] += 1
            masked_call_counts[projection.issue_code] += 1
            message["loss"] = False
            if message_index > 0 and messages[message_index - 1].get("role") == "assistant":
                messages[message_index - 1]["loss"] = False
            continue
        projection = project_tool_call(call, tools)
        call_counts[projection.status] += 1
        if not projection.executable:
            # Preserve the entire historical turn and its observation as
            # context, but do not teach a call rejected by the live runtime.
            message["loss"] = False
            if message_index > 0 and messages[message_index - 1].get("role") == "assistant":
                messages[message_index - 1]["loss"] = False
            masked_call_counts[projection.issue_code or "schema_mismatch"] += 1
            continue
        for field in projection.removed_arguments:
            removed_fields[f"{projection.name}.{field}"] += 1
        message["content"] = json.dumps(
            {
                "name": projection.name,
                "arguments": json.dumps(
                    projection.arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    try:
        canonical_tools, schema_stats = canonicalize_tool_schemas(tools)
    except PolicyCausalContractError:
        canonical_tools = str(tools)
        schema_stats = {
            "input_schema_count": len(_tool_functions(tools)),
            "output_schema_count": len(_tool_functions(tools)),
            "duplicate_schema_entries_removed": 0,
            "conflicting_schema_projection_skipped": 1,
        }
    if canonical_tools != "[]" or str(tools).strip():
        projected["tools"] = canonical_tools
    final_stats = project_final_observation_ids(messages)
    response_counts = Counter(
        _response_outcome(str(message.get("content", "")))
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "tool_response"
    )
    return projected, {
        "tool_calls": dict(call_counts),
        "masked_unrepairable_tool_calls": dict(masked_call_counts),
        "removed_argument_fields": dict(removed_fields),
        "tool_schemas": schema_stats,
        "final_observation_ids": final_stats,
        "tool_responses": dict(response_counts),
    }


def audit_policy_contract_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    row_count = 0
    rows_with_calls = 0
    rows_with_unmasked_unrepairable_calls = 0
    rows_with_final_errors = 0
    rows_with_masked_final_errors = 0
    rows_with_invisible_final_ids = 0
    rows_with_conflicting_schemas = 0
    rows_with_legacy_stage_control = 0
    call_status: Counter[str] = Counter()
    call_issue_codes: Counter[str] = Counter()
    call_issue_fields: Counter[str] = Counter()
    repair_removed_fields: Counter[str] = Counter()
    response_status: Counter[str] = Counter()
    schema_variant_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    locator_counts: Counter[str] = Counter()

    for row in rows:
        row_count += 1
        messages = row.get("messages")
        messages = messages if isinstance(messages, list) else []
        tools = row.get("tools", "")
        try:
            variants = _schema_variants(tools)
            for name, schemas in variants.items():
                if len(schemas) > 1:
                    schema_variant_counts[name] += len(schemas) - 1
            try:
                canonicalize_tool_schemas(tools)
            except PolicyCausalContractError:
                rows_with_conflicting_schemas += 1
        except Exception:
            variants = {}
            rows_with_conflicting_schemas += 1

        row_call_count = 0
        row_has_unmasked_unrepairable = False
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role", ""))
            role_counts[role] += 1
            content = str(message.get("content", ""))
            if role == "tool_response":
                response_status[_response_outcome(content)] += 1
                locator = _mapping(
                    _mapping(_raw_json_prefix(content)).get("observation_locator")
                )
                if str(locator.get("observation_id", "")).strip():
                    locator_counts[
                        "successful"
                        if locator.get("tool_success") is True
                        or locator.get("successful") is True
                        else "unsuccessful_or_unknown"
                    ] += 1
                if STAGE_CONTROL_RE.search(content):
                    rows_with_legacy_stage_control += 1
            if role != "tool_call":
                continue
            row_call_count += 1
            try:
                call = _json_object(content, label="tool call")
                projection = project_tool_call(call, tools)
            except Exception:
                projection = ToolProjection(
                    "", {}, "unrepairable", issue_code="malformed_tool_call"
                )
            call_status[projection.status] += 1
            if projection.status == "unknown_arguments_removed":
                for field in projection.removed_arguments:
                    repair_removed_fields[f"{projection.name}.{field}"] += 1
            if not projection.executable:
                if message.get("loss") is not False:
                    row_has_unmasked_unrepairable = True
                call_issue_codes[projection.issue_code or "schema_mismatch"] += 1
                for field in projection.issue_fields:
                    call_issue_fields[f"{projection.name}.{field}"] += 1
        if row_call_count:
            rows_with_calls += 1
        if row_has_unmasked_unrepairable:
            rows_with_unmasked_unrepairable_calls += 1

        if not messages or not isinstance(messages[-1], Mapping):
            rows_with_final_errors += 1
            continue
        content = str(messages[-1].get("content", ""))
        try:
            payload = extract_final_answer(content)
            RawHistoryJudgmentOutput.model_validate(payload)
        except Exception:
            if messages[-1].get("loss") is False:
                rows_with_masked_final_errors += 1
            else:
                rows_with_final_errors += 1
            continue
        ids = [
            str(item).strip()
            for item in payload.get("verdict_observation_ids", []) or []
            if str(item).strip()
        ]
        explicit_ids = successful_observation_ids(messages[:-1])
        prefix_text = "\n".join(
            str(message.get("content", ""))
            for message in messages[:-1]
            if isinstance(message, Mapping)
        )
        explicit = sum(1 for item in ids if item in explicit_ids)
        literal = sum(1 for item in ids if item in prefix_text)
        final_counts["target_ids"] += len(ids)
        final_counts["explicitly_grounded_ids"] += explicit
        final_counts["literal_prefix_matches"] += literal
        final_counts["duplicate_target_ids"] += len(ids) - len(set(ids))
        if len(ids) != explicit and messages[-1].get("loss") is not False:
            rows_with_invisible_final_ids += 1

    blockers = {
        "invalid_final_output_rows": rows_with_final_errors,
        "rows_with_noncausal_final_ids": rows_with_invisible_final_ids,
        "rows_with_unmasked_unrepairable_tool_calls": (
            rows_with_unmasked_unrepairable_calls
        ),
    }
    return {
        "schema_version": "ifv-policy-causal-contract-audit-v1",
        "passed": not any(blockers.values()),
        "row_count": row_count,
        "production_blockers": blockers,
        "roles": dict(sorted(role_counts.items())),
        "tool_calls": {
            "rows_with_calls": rows_with_calls,
            "status_counts": dict(sorted(call_status.items())),
            "unrepairable_issue_counts": dict(sorted(call_issue_codes.items())),
            "unrepairable_field_counts": dict(call_issue_fields.most_common(100)),
            "repairable_removed_field_counts": dict(
                repair_removed_fields.most_common(100)
            ),
        },
        "tool_schemas": {
            "duplicate_variant_entries_by_name": dict(
                schema_variant_counts.most_common(100)
            ),
            "rows_with_conflicting_variants": rows_with_conflicting_schemas,
        },
        "tool_responses": {
            "status_counts": dict(sorted(response_status.items())),
            "observation_locator_counts": dict(sorted(locator_counts.items())),
            "legacy_stage_control_occurrences": rows_with_legacy_stage_control,
        },
        "final_targets": dict(sorted(final_counts.items())),
        "masked_invalid_final_output_rows": rows_with_masked_final_errors,
    }

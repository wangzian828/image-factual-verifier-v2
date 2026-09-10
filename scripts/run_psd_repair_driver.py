"""Run one auditable IFV PSD repair attempt on the server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TRAINING_ROOT = REPO_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.llm_backend import APIBackend
from src.orchestrator.runtime_events import CaseRuntimeStore
from src.trajectory.scoring import score_process_trace

from ifv_training.io import load_json, write_json, write_jsonl
from ifv_training.psd_repair import (
    PSDModelRoles,
    _sha,
    build_psd_attempt_record,
    locate_failure_site,
)
from ifv_training.psd_repair_runtime import QwenContinuationAdapter
from ifv_training.psd_repair_verifier import (
    verify_continuation_pair,
    verify_source_rollout_failure,
)


PSD_VERIFICATION_BUNDLE_SCHEMA_VERSION = "ifv-psd-repair-verification-bundle-v1"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _capture_from_steps(steps: list[Any]) -> dict[str, Any]:
    for step in steps:
        capture = _mapping(getattr(step, "metadata", {}).get("policy_token_capture"))
        if _text(capture.get("status")) == "complete":
            return dict(capture)
    return {}


def _step_ids(capture: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    prompt = capture.get("prompt_token_ids")
    completion = capture.get("completion_token_ids")
    if not isinstance(prompt, list) or not isinstance(completion, list):
        return [], []
    return [int(item) for item in prompt], [int(item) for item in completion]


def _load_bound_artifact(value: Any, *, root: Path) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    path_value = _text(value)
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    loaded = load_json(path)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"verification artifact must be an object: {path}")
    return loaded


def _load_verification_bundle(path: Path | None) -> dict[int, Mapping[str, Any]]:
    if path is None:
        return {}
    value = load_json(path)
    if value.get("schema_version") != PSD_VERIFICATION_BUNDLE_SCHEMA_VERSION:
        raise ValueError("PSD repair verification bundle schema is invalid")
    rows = value.get("attempts")
    if not isinstance(rows, list):
        raise ValueError("PSD repair verification bundle requires attempts")
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("PSD repair verification attempt must be an object")
        index = row.get("hint_index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("PSD repair verification hint_index is invalid")
        if index in result:
            raise ValueError(f"duplicate PSD repair verification hint_index: {index}")
        result[index] = row
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--semantic-verification", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--public-context", type=Path, required=True)
    parser.add_argument("--private-context", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-provider", default="qwen_local")
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--policy-base-url")
    parser.add_argument("--policy-wire-api", default="chat_completions")
    parser.add_argument("--hint-constructor-provider", required=True)
    parser.add_argument("--hint-constructor-model", required=True)
    parser.add_argument("--hint-constructor-base-url")
    parser.add_argument("--hint-constructor-wire-api")
    parser.add_argument("--round-start-checkpoint", required=True)
    parser.add_argument("--hint-count", type=int, default=4)
    parser.add_argument("--hint-level", type=int, default=1)
    parser.add_argument("--max-suffix-actions", type=int, default=8)
    parser.add_argument("--verification-bundle", type=Path)
    return parser


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    trace = load_json(args.trace)
    audit = load_json(args.audit)
    semantic_verification = (
        load_json(args.semantic_verification)
        if args.semantic_verification
        else None
    )
    gold = load_json(args.gold)
    public_context = load_json(args.public_context)
    private_context = load_json(args.private_context) if args.private_context else None
    site = locate_failure_site(trace, audit, semantic_verification)
    if site is None:
        raise RuntimeError("no observed PSD repair failure site in trace/audit")
    source_verification = verify_source_rollout_failure(trace, gold=gold)
    if source_verification["passed"] is not True:
        raise RuntimeError(
            "source no-hint rollout was not explicitly verified as failed: "
            + ",".join(source_verification["reasons"])
        )
    verification_rows = _load_verification_bundle(args.verification_bundle)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    runtime_root = output_dir / "runtime"
    runtime_store = CaseRuntimeStore(
        runtime_root,
        case_id=_text(_mapping(trace.get("state")).get("runtime_case", {}).get("case_id"))
        or _text(trace.get("image_id")),
        attempt_id="psd-repair",
    )
    policy_orchestrator = Orchestrator(
        provider=args.policy_provider,
        model_name=args.policy_model,
        llm_base_url=args.policy_base_url,
        llm_wire_api=args.policy_wire_api,
        image_access_mode="direct_multimodal",
        validate_startup=True,
    )
    hint_constructor_llm = APIBackend(
        provider=args.hint_constructor_provider,
        model_name=args.hint_constructor_model,
        base_url=args.hint_constructor_base_url,
        wire_api=args.hint_constructor_wire_api,
        temperature=0.0,
        max_tokens=2048,
    )
    model_roles = PSDModelRoles(
        hint_constructor_provider=args.hint_constructor_provider,
        hint_constructor_model=args.hint_constructor_model,
        frozen_self_teacher_provider=args.policy_provider,
        frozen_self_teacher_model=args.policy_model,
        round_start_checkpoint=args.round_start_checkpoint,
        trainable_student_provider=args.policy_provider,
        trainable_student_model=args.policy_model,
        trainable_student_initial_checkpoint=args.round_start_checkpoint,
    )
    adapter = QwenContinuationAdapter(
        policy_llm=policy_orchestrator.llm,
        hint_constructor_llm=hint_constructor_llm,
        model_roles=model_roles,
        tools=list(policy_orchestrator.all_tools.values()),
        image_path=str(args.image),
        runtime_store=runtime_store,
        tool_cache=policy_orchestrator.tool_cache,
        cacheable_tools=list(policy_orchestrator.cacheable_tools),
        source_access_policy=policy_orchestrator.source_access_policy,
        tool_call_limits=policy_orchestrator.verification_tool_limits,
        source_runtime_store_path=site.runtime_store_path,
    )
    try:
        proposals = await adapter.propose_hints(
            failure_site=site,
            public_trace_context=public_context,
            private_context=private_context,
            hint_count=args.hint_count,
            hint_level=args.hint_level,
        )
        records: list[dict[str, Any]] = []
        for index, hint in enumerate(proposals):
            continuation = await adapter.run_hinted_episode(
                failure_site=site,
                hint=hint,
                max_suffix_actions=args.max_suffix_actions,
            )
            bound = verification_rows.get(index, {})
            expected_hint_sha = _sha(hint.text)
            if bound and _text(bound.get("hint_sha256")) != expected_hint_sha:
                raise ValueError(
                    f"verification bundle hint hash mismatch at index {index}"
                )
            bundle_root = (
                args.verification_bundle.parent
                if args.verification_bundle is not None
                else Path.cwd()
            )
            local_verification = _load_bound_artifact(
                bound.get("local_verification"),
                root=bundle_root,
            )
            hinted_teacher_episode_trace = _load_bound_artifact(
                bound.get("hinted_teacher_episode_trace"),
                root=bundle_root,
            )
            unhinted_student_episode_trace = _load_bound_artifact(
                bound.get("unhinted_student_episode_trace"),
                root=bundle_root,
            )
            verification, artifacts = verify_continuation_pair(
                base_trace=trace,
                teacher_steps=continuation.teacher_steps,
                student_steps=continuation.student_steps,
                hinted_teacher_episode_trace=hinted_teacher_episode_trace,
                unhinted_student_episode_trace=unhinted_student_episode_trace,
                gold=gold,
                local_verification=local_verification,
                repair_step_id=site.step_id,
            )
            teacher_capture = _capture_from_steps(continuation.teacher_steps)
            teacher_prompt_ids, completion_ids = _step_ids(teacher_capture)
            student_capture = _capture_from_steps(continuation.student_steps)
            student_prompt_ids, _student_completion_ids = _step_ids(student_capture)
            record = build_psd_attempt_record(
                candidate_id=_text(trace.get("image_id")) + f":repair:{index}",
                case_id=_text(_mapping(trace.get("state")).get("runtime_case", {}).get("case_id"))
                or _text(trace.get("image_id")),
                episode_id=_text(trace.get("image_id")),
                failure_site=site,
                hint=hint,
                model_roles=model_roles,
                verification=verification,
                student_prompt_ids=student_prompt_ids,
                teacher_prompt_ids=teacher_prompt_ids,
                completion_ids=completion_ids,
                teacher_token_capture=teacher_capture,
                source_trace_sha256=_sha(trace),
            )
            record["continuation"] = {
                "teacher_complete": continuation.teacher_complete,
                "student_complete": continuation.student_complete,
                "stop_reason": continuation.stop_reason,
                "teacher_step_count": len(continuation.teacher_steps),
                "student_step_count": len(continuation.student_steps),
                "hinted_teacher_episode_verified": artifacts[
                    "hinted_teacher_episode_verified"
                ],
                "unhinted_student_diagnostic": artifacts[
                    "unhinted_student_diagnostic"
                ],
                "unhinted_student_affects_acceptance": False,
            }
            record["local_verification"] = dict(local_verification or {})
            records.append(record)
        write_jsonl(output_dir / "repair_attempts.jsonl", records)
        result = {
            "schema_version": "ifv-psd-repair-driver-result-v1",
            "trace": str(args.trace),
            "runtime_archive": str(runtime_root),
            "candidate_count": len(records),
            "accepted_count": sum(1 for row in records if row.get("accepted") is True),
            "pending_hinted_episode_count": sum(
                1
                for row in records
                if "full_hinted_teacher_episode_trace_required"
                in (row.get("verification", {}).get("reasons", []) or [])
            ),
            "artifacts": {"repair_attempts": "repair_attempts.jsonl"},
        }
        write_json(output_dir / "manifest.json", result)
        return result
    finally:
        await hint_constructor_llm.aclose()
        await policy_orchestrator.aclose()


def main() -> None:
    args = _parser().parse_args()
    print(json.dumps(asyncio.run(_run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

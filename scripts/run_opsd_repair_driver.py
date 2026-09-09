"""Run one auditable IFV OPSD repair attempt on the server."""

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
from src.orchestrator.runtime_events import CaseRuntimeStore
from src.trajectory.scoring import score_process_trace

from ifv_training.io import load_json, write_json, write_jsonl
from ifv_training.opsd import (
    build_opsd_attempt_record,
    locate_failure_site,
)
from ifv_training.opsd_runtime import QwenContinuationAdapter
from ifv_training.opsd_verifier import verify_continuation_pair


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--public-context", type=Path, required=True)
    parser.add_argument("--private-context", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", default="qwen_local")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--wire-api", default="chat_completions")
    parser.add_argument("--hint-count", type=int, default=4)
    parser.add_argument("--hint-level", type=int, default=1)
    parser.add_argument("--max-suffix-actions", type=int, default=8)
    parser.add_argument(
        "--student-episode-trace",
        type=Path,
        help="Optional complete no-hint student episode for strict causal verification.",
    )
    return parser


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    trace = load_json(args.trace)
    audit = load_json(args.audit)
    gold = load_json(args.gold)
    public_context = load_json(args.public_context)
    private_context = load_json(args.private_context) if args.private_context else None
    site = locate_failure_site(trace, audit)
    if site is None:
        raise RuntimeError("no observed OPSD failure site in trace/audit")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    runtime_root = output_dir / "runtime"
    runtime_store = CaseRuntimeStore(
        runtime_root,
        case_id=_text(_mapping(trace.get("state")).get("runtime_case", {}).get("case_id"))
        or _text(trace.get("image_id")),
        attempt_id="opsd-repair",
    )
    orchestrator = Orchestrator(
        provider=args.provider,
        model_name=args.model,
        llm_base_url=args.base_url,
        llm_wire_api=args.wire_api,
        image_access_mode="direct_multimodal",
        validate_startup=True,
    )
    adapter = QwenContinuationAdapter(
        llm=orchestrator.llm,
        tools=list(orchestrator.all_tools.values()),
        image_path=str(args.image),
        runtime_store=runtime_store,
        tool_cache=orchestrator.tool_cache,
        cacheable_tools=list(orchestrator.cacheable_tools),
        source_access_policy=orchestrator.source_access_policy,
        tool_call_limits=orchestrator.verification_tool_limits,
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
        student_episode_trace = (
            load_json(args.student_episode_trace)
            if args.student_episode_trace
            else None
        )
        for index, hint in enumerate(proposals):
            continuation = await adapter.run_hinted_episode(
                failure_site=site,
                hint=hint,
                max_suffix_actions=args.max_suffix_actions,
            )
            verification, artifacts = verify_continuation_pair(
                base_trace=trace,
                teacher_steps=continuation.teacher_steps,
                student_steps=continuation.student_steps,
                student_episode_trace=student_episode_trace,
                gold=gold,
                local_pass=bool(
                    any(item.action_type == "tool_call" for item in continuation.teacher_steps)
                    and any(item.action_type == "tool_call" for item in continuation.student_steps)
                ),
            )
            teacher_capture = _capture_from_steps(continuation.teacher_steps)
            teacher_prompt_ids, completion_ids = _step_ids(teacher_capture)
            record = build_opsd_attempt_record(
                candidate_id=_text(trace.get("image_id")) + f":repair:{index}",
                case_id=_text(_mapping(trace.get("state")).get("runtime_case", {}).get("case_id"))
                or _text(trace.get("image_id")),
                episode_id=_text(trace.get("image_id")),
                failure_site=site,
                hint=hint,
                verification=verification,
                teacher_prompt_ids=teacher_prompt_ids,
                completion_ids=completion_ids,
                teacher_token_capture=teacher_capture,
            )
            record["continuation"] = {
                "teacher_complete": continuation.teacher_complete,
                "student_complete": continuation.student_complete,
                "stop_reason": continuation.stop_reason,
                "teacher_step_count": len(continuation.teacher_steps),
                "student_step_count": len(continuation.student_steps),
                "student_episode_verified": artifacts["student_episode_verified"],
            }
            records.append(record)
        write_jsonl(output_dir / "repair_attempts.jsonl", records)
        result = {
            "schema_version": "ifv-opsd-driver-result-v1",
            "trace": str(args.trace),
            "runtime_archive": str(runtime_root),
            "candidate_count": len(records),
            "accepted_count": sum(1 for row in records if row.get("accepted") is True),
            "pending_full_episode_count": sum(
                1
                for row in records
                if "full_student_episode_trace_required"
                in (row.get("verification", {}).get("reasons", []) or [])
            ),
            "artifacts": {"repair_attempts": "repair_attempts.jsonl"},
        }
        write_json(output_dir / "manifest.json", result)
        return result
    finally:
        await orchestrator.aclose()


def main() -> None:
    args = _parser().parse_args()
    print(json.dumps(asyncio.run(_run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

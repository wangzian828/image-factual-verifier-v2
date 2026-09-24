"""Submit a separately audited Gemini Batch judge round for frozen Agent traces.

Batch currently uses generateContent, unlike the live Interactions judge.  The
model, prompt, image bytes, JSON schema, and thinking settings stay fixed, but
results are never silently merged with the Interactions round.  A persisted
create intent makes an uncertain Batch submission a stop, not a replay.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Any

from scripts.audit_agent_private_gold import _overlay_report_sidecar, _read_json
from scripts.audit_direct_qa_baseline import _private_gold, _read_jsonl
from scripts.run_direct_qa_baseline import _parse_json_object, _resolve_image_path
from src.eval.agent_private_gold import agent_candidate_answer, build_agent_private_gold_candidate
from src.eval.evaluator_private_gold import private_gold_index
from src.eval.private_gold_judge_contract import (
    PRIVATE_GOLD_JUDGE_PROMPT,
    PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA,
)

MODEL = "gemini-3.7-flash"
MAX_SHARD_BYTES = 15_000_000
MAX_SHARD_CASES = 20


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stat_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def selected_successes(rollout: Path, frozen_ids: list[str]) -> list[tuple[str, Path, dict[str, Any]]]:
    expected = set(frozen_ids)
    successes: dict[str, tuple[Path, dict[str, Any]]] = {}
    for ledger in sorted(rollout.glob("*/run_results.jsonl")):
        for row in _read_jsonl(ledger):
            case_id = str(row.get("case_id") or "")
            if case_id not in expected:
                raise ValueError(f"unknown frozen case in Agent ledger: {case_id}")
            if not (row.get("status") == "success" and row.get("termination") == "success"
                    and row.get("verdict") in {"real", "fake"} and row.get("trace_path")):
                continue
            if case_id in successes:
                raise ValueError(f"duplicate successful Agent case: {case_id}")
            trace = (ledger.parent / str(row["trace_path"])).resolve()
            trace.relative_to(ledger.parent.resolve())
            if not trace.is_file():
                raise FileNotFoundError(trace)
            successes[case_id] = trace, row
    return [(case_id, *successes[case_id]) for case_id in frozen_ids if case_id in successes]


def prepare_one(case_id: str, trace_path: Path, source: dict[str, Any],
                gold: dict[str, Any], manifest_root: Path,
                allowed_image_root: Path) -> dict[str, Any]:
    trace, report_status = _overlay_report_sidecar(_read_json(trace_path), None)
    private_gold = _private_gold(gold)
    if not private_gold["auditable"]:
        raise ValueError(f"private gold is not auditable: {case_id}")
    candidate = build_agent_private_gold_candidate(trace)
    answer = agent_candidate_answer(candidate)
    if answer.get("verdict") != source.get("verdict"):
        raise ValueError(f"Agent verdict changed in candidate projection: {case_id}")
    image = _resolve_image_path({"image_path": trace.get("image_path")}, manifest_root).resolve()
    if not any(image == root.resolve() or root.resolve() in image.parents
               for root in (manifest_root, allowed_image_root)):
        raise ValueError(f"judge image is outside frozen roots: {case_id}")
    mime = mimetypes.guess_type(image.name)[0] or "image/jpeg"
    if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif", "image/avif"}:
        raise ValueError(f"unsupported judge image MIME type: {mime}")
    prompt = PRIVATE_GOLD_JUDGE_PROMPT + "\n\nAUDIT INPUT:\n" + json.dumps({
        "private_gold": private_gold,
        "candidate_material": {
            "mode": "agent_trace",
            "candidate_answer": answer,
            "supporting_trace_material": candidate,
        },
    }, ensure_ascii=False, indent=2)
    return {"case_id": case_id, "trace": stat_identity(trace_path),
            "source_verdict": source["verdict"], "gold_verdict": private_gold["expected_verdict"],
            "report_sidecar_status": report_status, "image": stat_identity(image),
            "image_mime": mime, "prompt": prompt}


def request_for(prepared: dict[str, Any]) -> dict[str, Any]:
    image = Path(prepared["image"]["path"])
    if stat_identity(image) != prepared["image"]:
        raise ValueError("frozen judge image identity changed")
    raw = image.read_bytes()
    return {"contents": [{"role": "user", "parts": [
        {"inlineData": {"mimeType": prepared["image_mime"],
                        "data": base64.b64encode(raw).decode("ascii")}},
        {"text": prepared["prompt"]},
    ]}], "generationConfig": {
        "maxOutputTokens": 32768,
        "thinkingConfig": {"thinkingLevel": "LOW", "includeThoughts": True},
        "responseMimeType": "application/json",
        "responseJsonSchema": PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA,
    }}


def plan(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    if binding["judge_model"] != MODEL or binding["thinking_level"] != "low" or binding["max_output_tokens"] != 32768:
        raise ValueError("live judge protocol changed")
    rollout = Path(binding["rollout_root"]).resolve()
    if rollout != args.rollout_root.resolve():
        raise ValueError("rollout root does not match judge binding")
    benchmark = Path(binding["benchmark"]["path"])
    manifest = Path(binding["manifest"]["path"])
    gold_path = Path(binding["gold"]["path"])
    for field, path in (("benchmark", benchmark), ("manifest", manifest), ("gold", gold_path)):
        if stat_identity(path) != binding[field]:
            raise ValueError(f"frozen {field} identity changed")
    frozen_ids = [str(row["case_id"]) for row in _read_jsonl(benchmark)]
    if len(frozen_ids) != 1526 or len(set(frozen_ids)) != 1526:
        raise ValueError("frozen case list invalid")
    gold = private_gold_index(_read_jsonl(gold_path))
    selected = selected_successes(rollout, frozen_ids)
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("no successful Agent traces ready for Batch")
    os.umask(0o077)
    output.mkdir(parents=True)
    prepared = [prepare_one(case_id, trace, row, gold[case_id], manifest.parent,
                            benchmark.parent)
                for case_id, trace, row in selected]
    # Each inline Batch creation has a 20 MB limit.  Leave generous overhead.
    shards: list[list[int]] = []
    current: list[int] = []
    size = 0
    for index, record in enumerate(prepared):
        estimate = len(record["prompt"].encode()) + 4 * ((record["image"]["size"] + 2) // 3) + 8192
        if estimate >= MAX_SHARD_BYTES:
            raise ValueError(f"case needs File API image path, not inline Batch: {record['case_id']}")
        if current and (size + estimate > MAX_SHARD_BYTES or len(current) >= MAX_SHARD_CASES):
            shards.append(current)
            current, size = [], 0
        current.append(index)
        size += estimate
    if current:
        shards.append(current)
    payload = {"schema_version": "ifv-psd-combined-batch-judge-plan-v1",
               "transport": "generateContent Batch (separate from Interactions)",
               "model": MODEL, "thinking_level": "low", "max_output_tokens": 32768,
               "prompt_sha256": hashlib.sha256(PRIVATE_GOLD_JUDGE_PROMPT.encode()).hexdigest(),
               "response_schema_sha256": hashlib.sha256(json.dumps(PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA, sort_keys=True).encode()).hexdigest(),
               "binding": binding, "case_count": len(prepared), "shards": shards,
               "prepared": prepared}
    save(output / "plan.json", payload)
    return {"case_count": len(prepared), "shard_count": len(shards), "output": str(output)}


def submit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    plan_data = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if plan_data["transport"] != "generateContent Batch (separate from Interactions)":
        raise ValueError("unexpected judge transport")
    shard_index = args.shard
    indices = plan_data["shards"][shard_index - 1]
    prepared = [plan_data["prepared"][index] for index in indices]
    receipt_dir = output / "shards" / f"shard-{shard_index:03d}"
    if receipt_dir.exists():
        raise FileExistsError("Batch create may not be retried without remote reconciliation")
    os.umask(0o077)
    receipt_dir.mkdir(parents=True)
    entries = [{"request": request_for(record),
                "metadata": {"key": record["case_id"], "case_id": record["case_id"]}}
               for record in prepared]
    payload = {"batch": {"display_name": f"psd-combined-judge-{output.name}-s{shard_index:03d}",
                         "input_config": {"requests": {"requests": entries}}}}
    payload_bytes = json.dumps(payload, ensure_ascii=False).encode()
    if len(payload_bytes) >= 20_000_000:
        raise ValueError("inline Batch request exceeds 20 MB")
    (receipt_dir / "request.json").write_bytes(payload_bytes)
    intent = {"phase": "create_intent", "model": MODEL,
              "case_ids": [record["case_id"] for record in prepared],
              "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
              "non_idempotent": True, "created_unix": time.time()}
    save(receipt_dir / "intent.json", intent)
    from dotenv import dotenv_values
    import httpx
    key = dotenv_values(args.credentials).get("GEMINI_API_KEY")
    if not key:
        raise ValueError("missing Gemini credential")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:batchGenerateContent"
    try:
        with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
            response = client.post(endpoint, headers={"x-goog-api-key": key,
                                                      "content-type": "application/json"},
                                   content=payload_bytes)
        body = response.json()
        receipt = {"http_status": response.status_code, "job_name": body.get("name"),
                   "metadata": body.get("metadata"), "error": body.get("error"),
                   "created_unix": time.time()}
        save(receipt_dir / "create-response.json", receipt)
        if response.status_code != 200 or not receipt["job_name"]:
            raise RuntimeError("Batch create rejected; inspect durable response")
        return {"case_count": len(prepared), "job_name": receipt["job_name"]}
    except Exception as exc:
        if not (receipt_dir / "create-response.json").exists():
            save(receipt_dir / "ambiguous.json", {"error_type": type(exc).__name__,
                                                  "error": str(exc)[:300],
                                                  "do_not_replay": True})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "submit"])
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", type=int, default=1)
    parser.add_argument("--credentials", type=Path,
                        default=Path("/volume/ybo/wza/private/runtime.env"))
    args = parser.parse_args()
    if args.limit < 0 or args.shard < 1:
        parser.error("limit and shard must be non-negative/positive")
    print(json.dumps(plan(args) if args.stage == "plan" else submit(args)))


if __name__ == "__main__":
    main()

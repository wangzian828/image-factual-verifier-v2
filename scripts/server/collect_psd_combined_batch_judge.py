"""Collect independently submitted Gemini Batch judge jobs without replaying them.

The generateContent Batch round is kept distinct from the earlier Interactions
stream.  A Batch operation may finish out of order; only exact, schema-valid,
image-conditioned results are accepted.  Raw responses remain available when
an item or the collector fails validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.eval.private_gold_judge_contract import (
    PRIVATE_GOLD_JUDGE_PROMPT,
    PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA,
)
from scripts.server.submit_psd_combined_batch_judge import MODEL, save, stat_identity


def validate_output(value: Any) -> list[str]:
    schema = PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA
    if not isinstance(value, dict):
        return ["judge output is not an object"]
    errors = []
    if set(value) != set(schema["required"]):
        errors.append("judge output keys differ from frozen schema")
    for key in ("quality_bucket", "fact_alignment", "reason_quality"):
        if value.get(key) not in schema["properties"][key]["enum"]:
            errors.append(f"invalid {key}")
    modes = value.get("failure_modes")
    if not isinstance(modes, list) or len(modes) > 12 or any(not isinstance(item, str) for item in modes):
        errors.append("invalid failure_modes")
    if not isinstance(value.get("explanation"), str):
        errors.append("invalid explanation")
    return errors


def validate_response(payload: dict[str, Any], expected: list[dict[str, Any]], job_name: str) -> list[dict[str, Any]]:
    if payload.get("metadata", {}).get("state") != "BATCH_STATE_SUCCEEDED":
        raise ValueError("Batch job is not succeeded")
    items = (payload.get("response", {}).get("inlinedResponses", {})
             .get("inlinedResponses"))
    if not isinstance(items, list) or len(items) != len(expected):
        raise ValueError("Batch result count differs from submitted cases")
    by_id = {str(row["case_id"]): row for row in expected}
    if len(by_id) != len(expected):
        raise ValueError("duplicate case in frozen Batch plan")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        case_id = str((item.get("metadata") or {}).get("case_id") or "")
        if case_id not in by_id or case_id in seen or (item.get("metadata") or {}).get("key") != case_id:
            raise ValueError("Batch result has missing, duplicate, or mismatched case key")
        seen.add(case_id)
        source = by_id[case_id]
        record: dict[str, Any] = {
            "case_id": case_id,
            "status": "error",
            "transport": "generateContent Batch",
            "job_name": job_name,
            "judge_model": MODEL,
            "source_trace": source["trace"],
            "candidate_verdict": source["source_verdict"],
            "gold_verdict": source["gold_verdict"],
            "verdict_matches_gold": source["source_verdict"] == source["gold_verdict"],
            "image": source["image"],
            "error": None,
        }
        if item.get("error") or not isinstance(item.get("response"), dict):
            record["error"] = "Batch item error or missing response"
            records.append(record)
            continue
        response = item["response"]
        candidates = response.get("candidates") or []
        image_tokens = sum(int(part.get("tokenCount") or 0) for part in
                           (response.get("usageMetadata") or {}).get("promptTokensDetails", [])
                           if part.get("modality") == "IMAGE")
        problems: list[str] = []
        if response.get("modelVersion") != MODEL:
            problems.append("judge model identity mismatch")
        if image_tokens <= 0:
            problems.append("no image tokens in judge request")
        if len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
            problems.append("judge did not finish normally")
        parsed = None
        if not problems:
            try:
                parts = candidates[0]["content"]["parts"]
                # includeThoughts=True returns a separate thought text part.
                # The structured verdict is the non-thought part only.
                text = "".join(part.get("text", "") for part in parts
                               if not part.get("thought"))
                parsed = json.loads(text)
                problems.extend(validate_output(parsed))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                problems.append("judge output is not valid JSON")
        if problems:
            record["error"] = "; ".join(problems)
        else:
            record.update({
                "status": "completed", "judge_output": parsed,
                "quality_bucket": parsed["quality_bucket"],
                "fact_alignment": parsed["fact_alignment"],
                "reason_quality": parsed["reason_quality"],
                "failure_modes": parsed["failure_modes"],
                "explanation": parsed["explanation"],
                "image_prompt_tokens": image_tokens,
                "judge_response_id": response.get("responseId"),
            })
        records.append(record)
    if seen != set(by_id):
        raise ValueError("Batch response omitted a frozen case")
    return records


def get_batch(job_name: str, api_key: str) -> dict[str, Any]:
    if not job_name.startswith("batches/"):
        raise ValueError("unexpected Batch job name")
    import httpx
    with httpx.Client(timeout=30) as client:
        response = client.get("https://generativelanguage.googleapis.com/v1beta/" + job_name,
                              headers={"x-goog-api-key": api_key})
    response.raise_for_status()
    return response.json()


def collect(output: Path, credentials: Path) -> dict[str, Any]:
    output = output.resolve()
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if (plan.get("transport") != "generateContent Batch (separate from Interactions)"
            or plan.get("model") != MODEL
            or plan.get("prompt_sha256") != hashlib.sha256(PRIVATE_GOLD_JUDGE_PROMPT.encode()).hexdigest()
            or plan.get("response_schema_sha256") != hashlib.sha256(json.dumps(
                PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA, sort_keys=True).encode()).hexdigest()):
        raise ValueError("frozen Batch judge protocol changed")
    from dotenv import dotenv_values
    api_key = dotenv_values(credentials).get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("missing Gemini credential")
    os.umask(0o077)
    summary: dict[str, Any] = {"collector_revision": 2,
                               "transport": "generateContent Batch", "model": MODEL,
                               "output": str(output), "shards": [], "counts": {}}
    all_records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for number, indices in enumerate(plan["shards"], 1):
        shard = output / "shards" / f"shard-{number:03d}"
        create_path = shard / "create-response.json"
        if not create_path.is_file():
            summary["shards"].append({"shard": number, "state": "not_submitted", "cases": len(indices)})
            counts["not_submitted"] += len(indices)
            continue
        receipt = json.loads(create_path.read_text(encoding="utf-8"))
        job_name = receipt.get("job_name")
        if receipt.get("http_status") != 200 or not job_name:
            raise ValueError(f"shard {number} has no confirmed Batch job")
        expected = [plan["prepared"][i] for i in indices]
        for record in expected:
            if (stat_identity(Path(record["trace"]["path"])) != record["trace"]
                    or stat_identity(Path(record["image"]["path"])) != record["image"]):
                raise ValueError("frozen Agent trace or judge image identity changed")
        raw_path = shard / "batch-response.json"
        payload = (json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists()
                   else get_batch(job_name, api_key))
        metadata = payload.get("metadata", {})
        state = metadata.get("state")
        info = {"shard": number, "job_name": job_name, "state": state,
                "cases": len(expected), "batch_stats": metadata.get("batchStats", {}),
                "create_time": metadata.get("createTime"), "end_time": metadata.get("endTime"),
                "batch_error": payload.get("error")}
        if state == "BATCH_STATE_SUCCEEDED":
            if not raw_path.exists():
                save(raw_path, payload)
            records = validate_response(payload, expected, job_name)
            # v1 receipts remain as evidence of its thought-part parser bug.
            result_path = shard / "records-v2.json"
            if result_path.exists() and json.loads(result_path.read_text(encoding="utf-8")) != records:
                raise ValueError("previously collected Batch records changed")
            if not result_path.exists():
                save(result_path, records)
            all_records.extend(records)
            info["completed"] = sum(row["status"] == "completed" for row in records)
            info["errors"] = len(records) - info["completed"]
            counts["completed"] += info["completed"]
            counts["error"] += info["errors"]
        else:
            counts["pending_or_failed"] += len(expected)
        summary["shards"].append(info)
    summary["counts"] = dict(counts)
    save(output / "collection-status-v2.json", summary)
    # This file contains only current, fully validated Batch records.  It is a
    # convenience index, not a replacement for immutable per-shard receipts.
    save(output / "collected-records-v2.json", all_records)
    return {"counts": summary["counts"], "shard_states": Counter(
        row["state"] for row in summary["shards"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credentials", type=Path,
                        default=Path("/volume/ybo/wza/private/runtime.env"))
    args = parser.parse_args()
    print(json.dumps(collect(args.output, args.credentials), default=dict))


if __name__ == "__main__":
    main()

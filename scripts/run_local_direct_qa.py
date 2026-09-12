"""Image-only Direct QA against an attested local OpenAI-compatible service.

Reuses the comparison's shared prompt, not the Agent. Public input only;
private-gold judging remains in audit_direct_qa_baseline.py after generation.
Completed answers (including wrong answers) are never regenerated on resume.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import hashlib
import json
import mimetypes
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_direct_qa_baseline import DEFAULT_PROMPT
import httpx

SCHEMA = {"type": "object", "properties": {"core_fact": {"type": "string"},
    "verdict": {"type": "string", "enum": ["real", "fake"]}, "reason": {"type": "string"}},
    "required": ["core_fact", "verdict", "reason"], "additionalProperties": False}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(path):
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not cases or len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("public cases must be nonempty and unique")
    for row in cases:
        if set(row) - {"case_id", "image_path", "image_sha256"}:
            raise ValueError("Direct QA accepts only the public image projection")
        image = (path.parent / row["image_path"]).resolve()
        if digest(image) != row["image_sha256"]:
            raise ValueError("public image hash mismatch")
        row["image_path"] = str(image)
    return cases


def payload(case, model, max_tokens):
    image = Path(case["image_path"])
    mime = mimetypes.guess_type(image.name)[0] or "image/jpeg"
    url = f"data:{mime};base64," + base64.b64encode(image.read_bytes()).decode()
    return {"model": model, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": DEFAULT_PROMPT}]}],
        "max_tokens": max_tokens, "temperature": 0, "seed": 1729,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"max_pixels": 262144},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "direct_qa", "strict": True, "schema": SCHEMA}}}


async def run(args):
    cases = load_cases(args.benchmark.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {"schema_version": "ifv-local-direct-qa-v1", "model": args.model,
        "model_path": str(args.model_path.resolve()), "base_url": args.base_url,
        "benchmark": str(args.benchmark.resolve()), "benchmark_sha256": digest(args.benchmark),
        "prompt_sha256": hashlib.sha256(DEFAULT_PROMPT.encode()).hexdigest(),
        "expected_count": len(cases), "concurrency": args.concurrency, "max_output_tokens": args.max_tokens,
        "enable_thinking": False, "temperature": 0, "seed": 1729, "max_pixels": 262144,
        "input_contract": "image_only", "request_fields": ["image", "shared_prompt"]}
    config_path = output / "run-config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Direct QA resume input/config changed")
    config_path.write_text(json.dumps(config, indent=2))
    (output / "prompt.txt").write_text(DEFAULT_PROMPT + "\n")
    results = output / "results.jsonl"
    latest = {}
    if results.exists():
        for line in results.read_text().splitlines():
            row = json.loads(line)
            latest[row["case_id"]] = row
    if set(latest) - {row["case_id"] for row in cases}:
        raise ValueError("unexpected cached case")
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False,
        limits=httpx.Limits(max_connections=args.concurrency + 4)) as client:
        cards = (await client.get(args.base_url + "/models")).json()["data"]
        if not any(card["id"] == args.model and Path(card["root"]).resolve() == args.model_path.resolve()
                   for card in cards):
            raise ValueError("local model root/alias mismatch")
        async def one(case):
            async with semaphore:
                started = time.monotonic()
                record = {"case_id": case["case_id"], "image_path": case["image_path"],
                    "image_sha256": case["image_sha256"], "model": args.model, "status": "error"}
                for attempt in range(args.retries + 1):
                    try:
                        response = await client.post(args.base_url + "/chat/completions",
                            json=payload(case, args.model, args.max_tokens))
                        response.raise_for_status()
                        body = response.json()
                        choice = body["choices"][0]
                        text = choice["message"]["content"]
                        parsed = json.loads(text)
                        if choice.get("finish_reason") != "stop" or parsed.get("verdict") not in ("real", "fake"):
                            raise ValueError("incomplete or invalid Direct QA response")
                        if any(not isinstance(parsed.get(key), str) or not parsed[key].strip() for key in SCHEMA["required"]):
                            raise ValueError("missing Direct QA response fields")
                        record.update(status="completed", model_output=parsed, model_output_text=text,
                            predicted_verdict=parsed["verdict"], usage=body.get("usage"),
                            native_thought=choice["message"].get("reasoning_content") or "",
                            json_parse_error=None, attempts=attempt + 1)
                        break
                    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                        record.update(error_type=type(exc).__name__, error=str(exc)[:500], attempts=attempt + 1)
                        if attempt < args.retries:
                            await asyncio.sleep(min(2 ** attempt, 8))
                record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                return record
        pending = [row for row in cases if latest.get(row["case_id"], {}).get("status") != "completed"]
        tasks = [asyncio.create_task(one(row)) for row in pending]
        with results.open("a") as file:
            for task in asyncio.as_completed(tasks):
                row = await task
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                latest[row["case_id"]] = row
                progress = {"expected": len(cases), "returned": len(latest),
                    "completed": sum(x["status"] == "completed" for x in latest.values()),
                    "phase": "direct_qa", "judge_complete": False}
                (output / "progress.json").write_text(json.dumps(progress))
                print(json.dumps(progress), flush=True)
    completed = sum(x["status"] == "completed" for x in latest.values())
    summary = {"expected": len(cases), "completed": completed, "errors": len(cases) - completed,
        "phase": "direct_qa_complete", "judge_complete": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0 if completed == len(cases) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "output", "model-path"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--model", default="ifv-qwen3.5-9b-base-qa")
    parser.add_argument("--base-url", default="http://127.0.0.1:8915/v1")
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--retries", type=int, default=2)
    raise SystemExit(asyncio.run(run(parser.parse_args())))

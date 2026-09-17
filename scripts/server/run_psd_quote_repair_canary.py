"""Run two bound quote-only source-review repairs against preserved cache."""
from __future__ import annotations

import asyncio
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path("/volume/ybo/wza")
CODE = ROOT / "training-artifacts/psd-abstention-quote-repair-20260917-v45/code"
ARTIFACT_ROOT = ROOT / "training-artifacts/psd-abstention-quote-repair-20260917-v45"
RUN = ROOT / "runs/psd-production400x8-20260917-v6"
PREP = ROOT / "runs/psd-pilot400-preparation-20260915-v1"
PREFETCH = RUN / "source-review-prefetch-v2-auto-retry"
EPISODES = [
    "route-aware-hrc-stage2-10000-20260817-baseline3113:0010:generated_route-aware-hrc-stage2-10000-20260817-baseline3113-round-001-batch-0002__r--c88c68c7--r002",
    "main-06143--52becd82--r002",
]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


async def main(output: Path, episodes: list[str]) -> None:
    output = output.resolve()
    output.relative_to(ARTIFACT_ROOT.resolve())
    global OUT
    OUT = output
    if OUT.exists():
        raise RuntimeError("Never overwrite a real quote-repair canary")
    sys.path[:0] = [str(CODE), str(CODE / "training")]
    if not (os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()):
        owner = module("psd_epoch3_owner",
            ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py")
        receipt = load(RUN / "source-review-prefetch-process.json")
        os.environ.update(owner.checked(receipt))

    from scripts.prefetch_psd_source_reviews import completed_source, load_scope, validate_prefetch_cache
    from ifv_training.psd_gemini_judge import VERSION as REQUEST_VERSION, _request
    from ifv_training.psd_repair import _sha
    from ifv_training import psd_source_review as source
    from src.integrations.gemini import GeminiInteractionsClient, GeminiRequestGate

    benchmark = PREP / "runtime-release/runtime_input/cases.jsonl"
    train_cases = PREP / "evaluator_private/case_split.jsonl"
    private_gold = PREP / "evaluator_private/private_gold.jsonl"
    kwargs = {"run_dir": RUN / "episodes", "benchmark": benchmark,
              "train_cases": train_cases, "private_gold": private_gold,
              "model": "gemini-3.1-pro-preview"}
    scope = load_scope(**kwargs)
    old_cache = validate_prefetch_cache(PREFETCH, **kwargs)
    cache = OUT / "judge-cache"
    cache.mkdir(parents=True)

    def cache_path(packet, prompt, schema, images):
        identity = {"version": REQUEST_VERSION, "model": kwargs["model"],
                    "prompt_sha256": _sha(prompt), "schema_sha256": _sha(schema),
                    "packet_sha256": _sha(packet), "images_sha256": _sha(images)}
        return Path(old_cache) / (_sha(identity) + ".json"), cache / (_sha(identity) + ".json")

    rows = []
    async with GeminiInteractionsClient(timeout=240, max_retries=2,
            request_gate=GeminiRequestGate(1)) as client:
        for episode in episodes:
            trace, image, trace_path, binding = completed_source(scope, episode)
            images, media = source.review_images({"source": trace}, image_path=image)
            packet = source._packet(trace, scope["gold"][scope["expected"][episode]["case_id"]],
                                    media, include_transport_ids=True)
            first_old, first_new = cache_path(packet, source.PROMPT, source.SCHEMA, images)
            if not first_old.is_file():
                raise FileNotFoundError("First source-review cache is missing")
            shutil.copy2(first_old, first_new)
            first, _ = await _request(None, packet, prompt=source.PROMPT, schema=source.SCHEMA,
                                      model=kwargs["model"], images=images, cache_dir=cache)
            correction = source._correction_packet(packet, first)
            second_old, second_new = cache_path(correction, source.CORRECTION_PROMPT,
                                                source.SCHEMA, images)
            if not second_old.is_file():
                raise FileNotFoundError("Corrective source-review cache is missing")
            shutil.copy2(second_old, second_new)
            before = len(list(cache.glob("*.json")))
            artifact = await source.judge_source(client, trace,
                gold=scope["gold"][scope["expected"][episode]["case_id"]],
                image_path=image, model=kwargs["model"], cache_dir=cache)
            after = len(list(cache.glob("*.json")))
            status = source.validate_source_review(artifact, trace=trace,
                gold=scope["gold"][scope["expected"][episode]["case_id"]])
            if after != before + 1 or "evidence_quote_repair" not in artifact:
                raise ValueError("Canary did not perform exactly one bound quote-only request")
            artifact_path = OUT / "artifacts" / (str(len(rows)) + ".json")
            save(artifact_path, artifact)
            rows.append({"episode": episode, "status": status,
                         "trace_sha256": binding["trace_sha256"],
                         "artifact": str(artifact_path),
                         "artifact_sha256": source.sha256_file(artifact_path),
                         "provider_calls": 1,
                         "decision_unchanged_except_quotes": True})
    save(OUT / "result.json", {"passed": len(rows) == len(episodes), "records": rows,
        "active_reviewer_restarted": False, "policy_agent_changed": False})
    print(json.dumps({"passed": True, "records": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
        default=ARTIFACT_ROOT / "real-quote-repair-canary")
    parser.add_argument("--episode", action="append", choices=EPISODES,
        help="Run only a named frozen regression episode; may be repeated")
    args = parser.parse_args()
    asyncio.run(main(args.output, args.episode or EPISODES))

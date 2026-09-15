"""Read-only canonical-trace latency attribution; no provider calls or raw export."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def describe(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    def percentile(q):
        x = (len(values) - 1) * q
        lo, hi = math.floor(x), math.ceil(x)
        return round(values[lo] + (values[hi] - values[lo]) * (x - lo), 3)
    return {"n": len(values), "mean": round(statistics.mean(values), 3),
        "p50": percentile(.5), "p90": percentile(.9), "p95": percentile(.95),
        "max": round(values[-1], 3), "sum": round(sum(values), 3)}


def summarize(paths):
    cases, tools, stage_llm = [], defaultdict(list), defaultdict(list)
    outcomes, finish, unreadable = Counter(), Counter(), Counter()
    for path in paths:
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            unreadable[type(exc).__name__] += 1
            continue
        success = trace.get("termination") == "success" and not trace.get("error")
        outcomes["success" if success else "error_or_incomplete"] += 1
        if not success:
            continue
        state = trace.get("state", {})
        record = {k: 0.0 for k in ("wall_s", "llm_s", "tool_s", "prompt_tokens", "completion_tokens",
            "actions", "llm_requests", "reasoning_chars", "judgment_llm_s")}
        record["wall_s"] = float(trace.get("time_taken") or state.get("stage_timings", {}).get("total") or 0)
        for step in state.get("all_steps", []):
            meta = step.get("metadata", {})
            if meta.get("deterministic_segment_boundary"):
                continue
            latency = float(meta.get("llm_duration_ms") or 0) / 1000
            if latency:
                record["llm_s"] += latency
                record["llm_requests"] += 1
                stage = step.get("stage", "unknown")
                stage_llm[stage].append(latency)
                record["judgment_llm_s"] += latency if stage == "unified_judgment" else 0
                finish[str(meta.get("finish_reason") or "unknown")] += 1
                record["reasoning_chars"] += float(meta.get("response_reasoning_chars") or 0)
                tokens = step.get("tokens") or {}
                record["prompt_tokens"] += float(tokens.get("prompt") or 0)
                record["completion_tokens"] += float(tokens.get("completion") or 0)
            if step.get("action_type") == "tool_call":
                duration = float(meta.get("duration_ms") or 0) / 1000
                record["tool_s"] += duration
                record["actions"] += 1
                tools[step.get("tool_name") or "unknown"].append(duration)
        record["other_or_unattributed_s"] = record["wall_s"] - record["llm_s"] - record["tool_s"]
        cases.append(record)
    totals = {k: describe([c[k] for c in cases]) for k in (cases[0] if cases else [])}
    wall = totals.get("wall_s", {}).get("sum", 0)
    return {"schema_version": "ifv-agent-latency-v1", "read_only": True,
        "outcomes": dict(outcomes), "unreadable": dict(unreadable),
        "cohort": "completed successful attempts only; in-flight/failed long tails excluded; not formal quality metrics",
        "case_distributions": totals,
        "wall_share_percent": {k: round(totals[k]["sum"] / wall * 100, 2) if wall else None
            for k in ("llm_s", "tool_s", "other_or_unattributed_s") if k in totals},
        "per_tool_outer_wall_s": {k: describe(v) for k, v in sorted(tools.items())},
        "per_stage_llm_wall_s": {k: describe(v) for k, v in sorted(stage_llm.items())},
        "finish_reasons": dict(finish),
        "limits": ["LLM wall includes server queue, prefill, decode and network; no per-request TTFT in these traces",
            "tool wall includes any nested tool LLM requests; do not add subcall timings again",
            "reasoning_chars is characters, not tokens", "negative residuals signal overlapping/duplicate timings"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--last", type=int, help="Most recently completed files, not successful-outcome selection")
    args = parser.parse_args()
    root = args.run.resolve()
    root.relative_to(Path("/volume/ybo/wza"))
    paths = list(root.glob("traces/*.json")) + list(root.glob("*/traces/*.json"))
    paths.sort(key=lambda p: p.stat().st_mtime)
    if args.last:
        paths = paths[-args.last:]
    report = summarize(paths)
    report["files_examined"] = len(paths)
    report["run"] = str(root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

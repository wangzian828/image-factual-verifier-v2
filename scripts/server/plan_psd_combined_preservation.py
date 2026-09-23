"""Propose a balanced set of *whole* PSD preservation episodes from small indexes.

This is metadata-only. It does not read target payloads, image pixels, scores,
private verdicts, or gold labels, and cannot authorize scoring or training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def _read(path: Path, source: str, excluded: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if not rows:
        raise ValueError(f"empty {source} preservation index")
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for row in rows:
        case, episode = row.get("case_id"), row.get("episode_id")
        if not isinstance(case, str) or not case or case in seen:
            raise ValueError(f"duplicate/missing {source} preservation case")
        if not isinstance(episode, str) or not episode:
            raise ValueError(f"missing {source} preservation episode")
        seen.add(case)
        if type(row.get("step_count")) is not int or row["step_count"] <= 0:
            raise ValueError(f"invalid {source} step count")
        if type(row.get("max_prompt_tokens")) is not int or row["max_prompt_tokens"] <= 0:
            raise ValueError(f"invalid {source} context length")
        tools = row.get("tool_sequence")
        if not isinstance(tools, list) or not tools or any(not isinstance(t, str) or not t for t in tools):
            raise ValueError(f"invalid {source} native tool sequence")
        if source == "smallbank":
            if row.get("verified_full_task") is not True or row.get("strict_trace_audit_pass") is not True:
                raise ValueError("unverified smallbank preservation")
            if type(row.get("image_steps")) is not int or row["image_steps"] <= 0:
                raise ValueError("smallbank episode lacks image binding")
        else:
            ids = row.get("step_ids")
            if not isinstance(ids, list) or len(ids) != row["step_count"] or len(set(ids)) != len(ids):
                raise ValueError("old1000 fixed episode step IDs invalid")
            judgment_count = sum(":judgment:" in sid for sid in ids)
            if judgment_count != 1:
                if excluded is None:
                    raise ValueError("old1000 fixed episode has no unique judgment")
                excluded.append({"source": source, "case_id": case, "episode_id": episode,
                                 "reason": "ambiguous_judgment_steps",
                                 "judgment_steps": judgment_count,
                                 "preservation_steps": row["step_count"]})
                continue
        row["selection_source"] = source
        kept.append(row)
    if not kept:
        raise ValueError(f"no eligible {source} preservation episodes")
    return kept


def _quartile(values: list[int]) -> list[int]:
    ordered = sorted(values)
    bounds = [ordered[len(ordered) * q // 4] for q in (1, 2, 3)]
    return [sum(value > bound for bound in bounds) for value in values]


def _features(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    lengths = np.array([r["step_count"] for r in rows], dtype=float)
    result: dict[str, np.ndarray] = {"episodes": np.ones(len(rows))}
    for name, values in (("length", [r["step_count"] for r in rows]),
                         ("context", [r["max_prompt_tokens"] for r in rows])):
        bins = _quartile(values)
        for bucket in range(4):
            result[f"{name}_quartile_{bucket}"] = np.array(
                [lengths[i] if bins[i] == bucket else 0.0 for i in range(len(rows))])
    tools = sorted({tool for row in rows for tool in row["tool_sequence"]})
    for tool in tools:
        result[f"native_tool_{tool}"] = np.array(
            [row["tool_sequence"].count(tool) for row in rows], dtype=float)
    if rows[0]["selection_source"] == "smallbank":
        result["image_steps"] = np.array([r["image_steps"] for r in rows], dtype=float)
    return result


def propose(small: list[dict[str, Any]], old: list[dict[str, Any]],
            quotas: dict[str, int], *, time_limit: float = 40.0) -> dict[str, Any]:
    if set(quotas) != {"smallbank", "old1000"} or any(type(q) is not int or q <= 0 for q in quotas.values()):
        raise ValueError("two positive source-specific step quotas required")
    groups = {"smallbank": small, "old1000": old}
    if {r["case_id"] for r in small} & {r["case_id"] for r in old}:
        raise ValueError("cross-source preservation case overlap")
    rows = small + old
    n = len(rows)
    features = {source: _features(group) for source, group in groups.items()}
    # Variables: one binary decision per episode, two nonnegative deviations
    # per source quota and per metadata feature. All features are normalized by
    # their expected selected magnitude so rare tools cannot dominate.
    definitions: list[tuple[str, str, np.ndarray, float, float]] = []
    for source, group in groups.items():
        steps = float(sum(r["step_count"] for r in group))
        if quotas[source] >= steps:
            raise ValueError("quota must be below available complete episodes")
        offset = 0 if source == "smallbank" else len(small)
        mass = np.zeros(n)
        mass[offset:offset + len(group)] = [r["step_count"] for r in group]
        definitions.append((source, "selected_steps", mass, float(quotas[source]), 100.0))
        for name, local in features[source].items():
            vector = np.zeros(n)
            vector[offset:offset + len(group)] = local
            target = float(local.sum()) * quotas[source] / steps
            definitions.append((source, name, vector, target, 1.0 / max(target, 5.0)))
    m = len(definitions)
    c = np.zeros(n + 2 * m)
    for j, (_, _, _, _, weight) in enumerate(definitions):
        c[n + 2*j:n + 2*j + 2] = weight
    matrix = np.zeros((m, n + 2 * m))
    equal = np.zeros(m)
    for j, (_, _, vector, target, _) in enumerate(definitions):
        matrix[j, :n] = vector
        matrix[j, n + 2*j] = -1
        matrix[j, n + 2*j + 1] = 1
        equal[j] = target
    all_tools = sorted({tool for row in rows for tool in row["tool_sequence"]})
    cover = np.zeros((len(all_tools), n + 2*m))
    for j, tool in enumerate(all_tools):
        cover[j, :n] = [float(tool in row["tool_sequence"]) for row in rows]
    constraints = [LinearConstraint(matrix, equal, equal),
                   LinearConstraint(cover, np.ones(len(all_tools)), np.full(len(all_tools), np.inf))]
    integrality = np.zeros(n + 2*m)
    integrality[:n] = 1
    lower = np.zeros(n + 2*m)
    upper = np.full(n + 2*m, np.inf)
    upper[:n] = 1
    result = milp(c=c, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=constraints, options={"time_limit": time_limit, "mip_rel_gap": 0.02})
    if result.x is None or result.status not in (0, 1):
        raise RuntimeError(f"metadata selection solver failed: {result.message}")
    chosen = [row for i, row in enumerate(rows) if result.x[i] > 0.5]
    selected_cases = {r["case_id"] for r in chosen}
    if len(selected_cases) != len(chosen):
        raise AssertionError("duplicate selected cases")
    report: dict[str, Any] = {}
    gate_pass = True
    for source, group in groups.items():
        selected = [r for r in chosen if r["selection_source"] == source]
        selected_steps = sum(r["step_count"] for r in selected)
        step_error = abs(selected_steps - quotas[source]) / quotas[source]
        feature_report: dict[str, Any] = {}
        source_steps = sum(r["step_count"] for r in group)
        for name, local in features[source].items():
            source_rate = float(local.sum()) / source_steps
            selected_value = sum(local[i] for i, row in enumerate(group) if row["case_id"] in selected_cases)
            selected_rate = float(selected_value) / selected_steps
            feature_report[name] = {"source_rate_per_step": source_rate,
                                    "selected_rate_per_step": selected_rate,
                                    "absolute_rate_difference": abs(selected_rate - source_rate)}
        worst_quartile = max(feature_report[name]["absolute_rate_difference"] for name in feature_report
                             if name.startswith(("length_quartile_", "context_quartile_")))
        gate_pass &= step_error <= 0.03 and worst_quartile <= 0.10
        report[source] = {"available_episodes": len(group), "available_steps": source_steps,
                          "selected_episodes": len(selected), "selected_steps": selected_steps,
                          "requested_steps": quotas[source], "relative_step_error": step_error,
                          "worst_quartile_rate_difference": worst_quartile,
                          "features": feature_report}
    represented = {tool for row in chosen for tool in row["tool_sequence"]}
    gate_pass &= represented == set(all_tools)
    return {"selected": [{"source": r["selection_source"], "case_id": r["case_id"],
                          "episode_id": r["episode_id"], "step_count": r["step_count"]}
                         for r in sorted(chosen, key=lambda r: (r["selection_source"], r["case_id"]))],
            "report": report, "all_native_tools": all_tools,
            "represented_native_tools": sorted(represented),
            "metadata_coverage_gate_pass": bool(gate_pass),
            "solver_status": int(result.status), "solver_message": result.message,
            "solver_objective": float(result.fun)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-index", type=Path, required=True)
    parser.add_argument("--old-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--small-steps", type=int, required=True)
    parser.add_argument("--old-steps", type=int, required=True)
    args = parser.parse_args()
    small = _read(args.small_index, "smallbank")
    excluded: list[dict[str, Any]] = []
    old = _read(args.old_index, "old1000", excluded)
    result = propose(small, old, {"smallbank": args.small_steps, "old1000": args.old_steps})
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "selected-episodes.jsonl").open("x", encoding="utf-8") as out:
        for row in result.pop("selected"):
            out.write(json.dumps(row, sort_keys=True) + "\n")
    result.update({"schema_version": "ifv-psd-combined-whole-preservation-plan-v1",
                   "status": "metadata_only_not_authorized_to_score_or_train",
                   "small_index": str(args.small_index.resolve()),
                   "old_index": str(args.old_index.resolve()),
                   "selection_policy": "all_eligible_repairs_plus_milp_whole_episode_source_length_context_tool_image_balance",
                   "uses_gold_or_teacher_score": False, "reads_image_pixels": False,
                   "copies_target_payload": False, "formal_training_started": False})
    result["excluded_ambiguous_episodes"] = excluded
    (args.output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata_coverage_gate_pass": result["metadata_coverage_gate_pass"],
                      "counts": {k: (v["selected_episodes"], v["selected_steps"])
                                 for k, v in result["report"].items()}}))


if __name__ == "__main__":
    main()

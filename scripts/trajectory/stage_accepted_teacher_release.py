#!/usr/bin/env python3
"""Stage one strictly accepted teacher rollout per case for SFT export.

Teacher retries historically used one rollout per invocation, so their public
episode IDs equal case IDs and collide across run directories.  This tool selects
the best strict/structured/semantic-passing candidate per case and stages only that
candidate into one canonical run directory.  It never reads evaluator-private gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trajectory.exporter import export_policy_examples


SCHEMA_VERSION = "ifv-accepted-teacher-release-v1"
JSONL_ARTIFACTS = (
    "perception_trajectories.jsonl",
    "trajectory_scores.jsonl",
    "run_results.jsonl",
    "rollout_groups.jsonl",
    "process_metrics.jsonl",
    "post_rollout_rewards.jsonl",
)


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate_index(root: Path, suffix: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.glob(f"*{suffix}")):
        payload = _load_json(path)
        episode_id = str(
            payload.get("episode_id")
            or (payload.get("rollout") or {}).get("episode_id", "")
        ).strip()
        if not episode_id or episode_id in result:
            raise ValueError(f"invalid or duplicate gate artifact: {path}")
        result[episode_id] = {"payload": payload, "path": path}
    return result


def _eligible(
    trace: Mapping[str, Any],
    trace_sha256: str,
    eligibility: Mapping[str, Any],
    semantic: Mapping[str, Any],
) -> bool:
    eligibility_gates = eligibility.get("gates") or {}
    semantic_gates = semantic.get("gates") or {}
    semantic_rollout = semantic.get("rollout") or {}
    return bool(
        str(trace.get("termination", "")) == "success"
        and str(trace.get("verdict", "")) in {"real", "fake"}
        and eligibility_gates.get("strict_trace_audit_pass") is True
        and eligibility_gates.get("engineering_valid") is True
        and eligibility_gates.get("sft_eligibility_pass") is True
        and semantic_gates.get("strict_trace_audit_pass") is True
        and semantic_gates.get("engineering_valid") is True
        and semantic_gates.get("semantic_audit_pass") is True
        and str((eligibility.get("source_trace") or {}).get("sha256", ""))
        == trace_sha256
        and str((semantic.get("source_trace") or {}).get("sha256", ""))
        == trace_sha256
        and str(semantic_rollout.get("episode_id", ""))
        == str(trace.get("image_id", ""))
    )


def stage_release(
    sources: list[tuple[Path, Path, Path]],
    output_dir: Path,
    *,
    minimum_accepted_cases: int,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected: Dict[str, Dict[str, Any]] = {}
    source_manifests: list[Dict[str, Any]] = []

    for run_dir, eligibility_dir, semantic_dir in sources:
        manifest = _load_json(run_dir / "run_manifest.json")
        if manifest.get("status") not in {"completed", "completed_with_errors"}:
            raise ValueError(f"teacher run is not terminal: {run_dir}")
        source_manifests.append(
            {
                "run_id": manifest.get("run_id"),
                "run_dir": str(run_dir),
                "eligibility_dir": str(eligibility_dir),
                "semantic_dir": str(semantic_dir),
                "git_commit": manifest.get("git_commit"),
            }
        )
        eligibility_index = _gate_index(eligibility_dir, ".sft_eligibility.json")
        semantic_index = _gate_index(semantic_dir, ".semantic_reward.json")
        for trace_path in sorted((run_dir / "traces").glob("*.json")):
            trace = _load_json(trace_path)
            state = trace.get("state") if isinstance(trace.get("state"), Mapping) else {}
            runtime_case = state.get("runtime_case") if isinstance(state, Mapping) else {}
            case_id = str((runtime_case or {}).get("case_id", "")).strip()
            episode_id = str(trace.get("image_id", "")).strip()
            if not case_id or not episode_id:
                raise ValueError(f"trace has no case/episode ID: {trace_path}")
            eligibility_row = eligibility_index.get(episode_id)
            semantic_row = semantic_index.get(episode_id)
            if not eligibility_row or not semantic_row:
                continue
            trace_sha256 = _sha256(trace_path)
            eligibility = eligibility_row["payload"]
            semantic = semantic_row["payload"]
            if not _eligible(trace, trace_sha256, eligibility, semantic):
                continue
            score = float((semantic.get("metrics") or {}).get("overall_process_quality", 0.0))
            candidate = {
                "case_id": case_id,
                "episode_id": episode_id,
                "score": score,
                "trace_path": trace_path,
                "trace_sha256": trace_sha256,
                "run_dir": run_dir,
                "run_id": str(manifest.get("run_id", run_dir.name)),
                "eligibility_path": eligibility_row["path"],
                "eligibility": eligibility,
                "semantic_path": semantic_row["path"],
                "semantic": semantic,
                "source_metadata": {
                    "source_run_id": str(manifest.get("run_id", run_dir.name)),
                    "runtime_commit": str(manifest.get("git_commit", "")),
                    "release_id": str((manifest.get("benchmark") or {}).get("release_id", "")),
                    "runtime_contract_version": str(
                        (manifest.get("benchmark") or {}).get("runtime_contract_version", "")
                    ),
                    "process_reference_protocol_version": str(
                        manifest.get("process_reference_protocol_version", "")
                    ),
                },
            }
            current = selected.get(case_id)
            if current is None or (score, candidate["run_id"], episode_id) > (
                current["score"], current["run_id"], current["episode_id"]
            ):
                selected[case_id] = candidate

    if len(selected) < minimum_accepted_cases:
        raise ValueError(
            f"accepted too few unique cases: {len(selected)} < {minimum_accepted_cases}"
        )

    selected_rows = []
    selected_by_run: Dict[Path, set[str]] = {}
    selected_cases_by_run: Dict[Path, set[str]] = {}
    policy_rows: list[Dict[str, Any]] = []
    (output_dir / "eligibility").mkdir(parents=True, exist_ok=True)
    (output_dir / "semantic_rewards").mkdir(parents=True, exist_ok=True)
    for case_id, candidate in sorted(selected.items()):
        selected_by_run.setdefault(candidate["run_dir"], set()).add(candidate["episode_id"])
        selected_cases_by_run.setdefault(candidate["run_dir"], set()).add(case_id)
        destination = output_dir / "traces" / f"{candidate['episode_id']}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate["trace_path"], destination)
        shutil.copy2(
            candidate["eligibility_path"],
            output_dir / "eligibility" / candidate["eligibility_path"].name,
        )
        shutil.copy2(
            candidate["semantic_path"],
            output_dir / "semantic_rewards" / candidate["semantic_path"].name,
        )
        trace = _load_json(candidate["trace_path"])
        exported_policy = export_policy_examples(
            trace,
            source_metadata=candidate["source_metadata"],
        )
        if not exported_policy:
            raise ValueError(f"accepted trace exported no policy examples: {case_id}")
        policy_rows.extend(item.model_dump(mode="json") for item in exported_policy)
        selected_rows.append(
            {
                "case_id": case_id,
                "episode_id": candidate["episode_id"],
                "source_run_id": candidate["run_id"],
                "source_run_dir": str(candidate["run_dir"]),
                "source_trace_sha256": candidate["trace_sha256"],
                "teacher_score": candidate["score"],
                "sft_eligibility_artifact_id": candidate["eligibility"].get("artifact_id"),
                "semantic_reward_artifact_id": candidate["semantic"].get("artifact_id"),
            }
        )

    _write_jsonl(output_dir / "policy_trajectories.jsonl", policy_rows)
    for artifact_name in JSONL_ARTIFACTS:
        rows: list[Dict[str, Any]] = []
        for run_dir, episode_ids in selected_by_run.items():
            case_ids = selected_cases_by_run[run_dir]
            for row in _load_jsonl(run_dir / artifact_name):
                episode_id = str(row.get("episode_id") or row.get("image_id") or "")
                case_id = str(row.get("case_id") or "")
                if episode_id in episode_ids or (not episode_id and case_id in case_ids):
                    rows.append(row)
        _write_jsonl(output_dir / artifact_name, rows)

    first_manifest = _load_json(sources[0][0] / "run_manifest.json")
    first_manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": output_dir.name,
            "status": "completed",
            "accepted_case_count": len(selected_rows),
            "accepted_sources": source_manifests,
            "selected_episodes": "selected_episodes.jsonl",
            "source_run_kind": "strict-structured-semantic-selected",
        }
    )
    _write_json(output_dir / "run_manifest.json", first_manifest)
    _write_jsonl(output_dir / "selected_episodes.jsonl", selected_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "accepted_case_count": len(selected_rows),
        "sources": source_manifests,
        "artifacts": {
            "run_dir": str(output_dir),
            "selected_episodes": "selected_episodes.jsonl",
            "eligibility_dir": "eligibility",
            "semantic_rewards_dir": "semantic_rewards",
        },
    }
    _write_json(output_dir / "accepted_release_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("RUN_DIR", "ELIGIBILITY_DIR", "SEMANTIC_DIR"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-accepted-cases", type=int, default=1)
    args = parser.parse_args()
    sources = [tuple(Path(item).expanduser().resolve() for item in source) for source in args.source]
    result = stage_release(
        sources, args.output_dir.expanduser().resolve(), minimum_accepted_cases=args.minimum_accepted_cases
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

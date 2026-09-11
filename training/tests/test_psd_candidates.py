from __future__ import annotations

import json
from pathlib import Path

from ifv_training.psd_candidates import build_psd_candidate_package
from ifv_training.io import sha256_file
from ifv_training.psd_round import PSD_ROLLOUT_GATE_SCHEMA_VERSION


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _rollout_gate(run_dir: Path, train_cases: Path) -> Path:
    manifest = run_dir / "run_manifest.json"
    if not manifest.is_file():
        _write_json(manifest, {"run_id": run_dir.name, "git_commit": "abc"})
    gate = run_dir / "psd-rollout-gate.json"
    _write_json(
        gate,
        {
            "schema_version": PSD_ROLLOUT_GATE_SCHEMA_VERSION,
            "round_index": 1,
            "passed": True,
            "run": {
                "directory": str(run_dir.resolve()),
                "manifest_sha256": sha256_file(manifest),
                "post_rollout_rewards_sha256": sha256_file(
                    run_dir / "post_rollout_rewards.jsonl"
                ),
                "rollout_groups_sha256": sha256_file(
                    run_dir / "rollout_groups.jsonl"
                ),
            },
            "train_cases": {"sha256": sha256_file(train_cases)},
        },
    )
    return gate


def _trace(
    *,
    case_id: str,
    rejected: bool = False,
    private: bool = False,
) -> dict:
    capture = {
        "schema_version": "ifv-policy-token-capture-v1",
        "status": "complete",
        "prompt_token_ids": [10, 11],
        "completion_token_ids": [12],
        "completion_logprobs": [-0.1],
        "missing": [],
    }
    policy_input: dict[str, object] = {
        "system_instruction": "Decide the next action.",
        "input_payload": {"events": [{"type": "observation", "id": "obs-1"}]},
        "tools": [],
    }
    if private:
        policy_input["evaluation_gold"] = {"expected_verdict": "fake"}
    react = {
        "stage": "unified_react",
        "action_type": "tool_call",
        "metadata": {
            "interaction_id": "react-1",
            "policy_input": policy_input,
            "policy_action": {
                "type": "tool_call",
                "name": "perceive_scene",
                "arguments": {"image_input": "case.jpg"},
            },
            "policy_token_capture": capture,
        },
    }
    judgment = {
        "stage": "unified_judgment",
        "action_type": "output_rejected" if rejected else "llm_response",
        "metadata": {
            "interaction_id": "judgment-1",
            "error_class": "protocol_error" if rejected else "",
            "policy_input": {
                "system_instruction": "Choose the final judgment.",
                "input_payload": {"events": [{"type": "observation", "id": "obs-2"}]},
                "tools": [],
            },
            "policy_action": {"verdict": "real"},
            "policy_token_capture": capture,
        },
    }
    return {
        "image_id": f"episode-{case_id}",
        "state": {
            "runtime_case": {"case_id": case_id},
            "all_steps": [react, judgment],
        },
    }


def test_build_psd_candidate_package_separates_public_queues(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_json(run_dir / "run_manifest.json", {"run_id": "run-1", "git_commit": "abc"})
    _write_json(run_dir / "traces" / "episode-repair.json", _trace(case_id="repair"))
    _write_json(run_dir / "traces" / "episode-preserve.json", _trace(case_id="preserve"))
    _write_json(
        run_dir / "traces" / "episode-protocol.json",
        _trace(case_id="protocol", rejected=True),
    )
    _write_json(
        run_dir / "traces" / "episode-private.json",
        _trace(case_id="private", private=True),
    )
    _write_jsonl(
        run_dir / "rollout_groups.jsonl",
        [
            {
                "episode_id": f"episode-{case_id}",
                "trace_path": f"traces/episode-{case_id}.json",
            }
            for case_id in ("repair", "preserve", "protocol", "private")
        ],
    )
    _write_jsonl(
        run_dir / "post_rollout_rewards.jsonl",
        [
            {
                "case_id": "repair",
                "episode_id": "episode-repair",
                "classification_correct": False,
                "fatal_engineering_error": False,
                "strict_trace_audit_pass": True,
            },
            {
                "case_id": "preserve",
                "episode_id": "episode-preserve",
                "classification_correct": True,
                "fatal_engineering_error": False,
                "strict_trace_audit_pass": True,
            },
            {
                "case_id": "protocol",
                "episode_id": "episode-protocol",
                "classification_correct": True,
                "fatal_engineering_error": False,
                "strict_trace_audit_pass": False,
                "strict_trace_audit_failure_codes": ["protocol"],
            },
            {
                "case_id": "engineering",
                "episode_id": "episode-engineering",
                "classification_correct": False,
                "fatal_engineering_error": True,
                "strict_trace_audit_pass": False,
            },
            {
                "case_id": "private",
                "episode_id": "episode-private",
                "classification_correct": False,
                "fatal_engineering_error": False,
                "strict_trace_audit_pass": True,
            },
            {
                "case_id": "test-case",
                "episode_id": "episode-test-case",
                "classification_correct": False,
                "fatal_engineering_error": False,
                "strict_trace_audit_pass": True,
            },
        ],
    )
    train_cases = tmp_path / "train-cases.jsonl"
    _write_jsonl(
        train_cases,
        [
            {"case_id": "repair", "split": "train"},
            {"case_id": "preserve", "split": "train"},
            {"case_id": "protocol", "split": "train"},
            {"case_id": "engineering", "split": "train"},
            {"case_id": "private", "split": "train"},
            {"case_id": "test-case", "split": "test"},
        ],
    )

    output = tmp_path / "candidates"
    manifest = build_psd_candidate_package(
        run_dir=run_dir,
        train_cases_path=train_cases,
        rollout_gate_path=_rollout_gate(run_dir, train_cases),
        output_dir=output,
    )

    assert manifest["counts"] == {
        "repair_candidates": 2,
        "preservation_candidates": 1,
        "engineering_requeue": 1,
        "rejections": 2,
        "token_capture_requeue": 0,
    }
    repairs = [
        json.loads(line)
        for line in (output / "repair_candidates.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [item["repair_signal"] for item in repairs] == [
        "terminal_outcome_mismatch",
        "protocol_rejection",
    ]
    assert repairs[0]["repair_site"]["step_id"] == "episode-repair:judgment:judgment-1"
    assert repairs[0]["source"]["psd_round_index"] == 1
    assert len(repairs[0]["source"]["psd_rollout_gate_sha256"]) == 64
    assert repairs[1]["repair_site"]["step_id"] == "episode-protocol:judgment:judgment-1"
    assert "classification_correct" not in repairs[0]
    assert "evaluation_gold" not in (output / "repair_candidates.jsonl").read_text(
        encoding="utf-8"
    )

    preservation = json.loads(
        (output / "preservation_candidates.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()[0]
    )
    assert preservation["class"] == "base_pass_preserve"
    assert len(preservation["preservation_steps"]) == 2
    assert all(
        "policy_input" in item["model_visible"]
        for item in preservation["preservation_steps"]
    )

    queued = json.loads(
        (output / "engineering_requeue.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()[0]
    )
    assert queued["queue_reason"] == "fatal_engineering_error"

    rejections = [
        json.loads(line)
        for line in (output / "rejections.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {item["reason"] for item in rejections} == {
        "source_split_forbidden:test",
        "trace_rejected:private/evaluator field in policy snapshot: policy_input.evaluation_gold",
    }


def test_build_psd_candidate_package_requeues_old_trace_without_capture(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    trace = _trace(case_id="old")
    for step in trace["state"]["all_steps"]:
        step["metadata"].pop("policy_token_capture")
    _write_json(run_dir / "traces" / "episode-old.json", trace)
    _write_jsonl(
        run_dir / "rollout_groups.jsonl",
        [{"episode_id": "episode-old", "trace_path": "traces/episode-old.json"}],
    )
    _write_jsonl(
        run_dir / "post_rollout_rewards.jsonl",
        [
            {
                "case_id": "old",
                "episode_id": "episode-old",
                "classification_correct": True,
                "fatal_engineering_error": False,
                "strict_trace_audit_pass": True,
            }
        ],
    )
    train_cases = tmp_path / "train-cases.jsonl"
    _write_jsonl(train_cases, [{"case_id": "old", "split": "train"}])

    output = tmp_path / "candidates"
    manifest = build_psd_candidate_package(
        run_dir=run_dir,
        train_cases_path=train_cases,
        rollout_gate_path=_rollout_gate(run_dir, train_cases),
        output_dir=output,
    )

    assert manifest["counts"]["preservation_candidates"] == 0
    assert manifest["counts"]["token_capture_requeue"] == 1
    queue = json.loads(
        (output / "token_capture_requeue.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()[0]
    )
    assert queue["queue_reason"] == "missing_policy_token_capture"
    assert queue["incomplete_step_ids"] == [
        "episode-old:react:react-1",
        "episode-old:judgment:judgment-1",
    ]


def test_strict_audit_location_maps_raw_index_to_projected_policy_step(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    trace = _trace(case_id="located")
    trace["state"]["all_steps"].insert(
        0,
        {
            "stage": "runtime_bootstrap",
            "action_type": "snapshot",
            "metadata": {},
        },
    )
    _write_json(run_dir / "traces" / "episode-located.json", trace)
    _write_jsonl(
        run_dir / "rollout_groups.jsonl",
        [
            {
                "episode_id": "episode-located",
                "trace_path": "traces/episode-located.json",
            }
        ],
    )
    _write_jsonl(
        run_dir / "post_rollout_rewards.jsonl",
        [
            {
                "case_id": "located",
                "episode_id": "episode-located",
                "classification_correct": True,
                "fatal_engineering_error": False,
                "strict_trace_audit_pass": False,
                "strict_trace_audit_failures": [
                    {"location": "state.all_steps[1].metadata.policy_action"}
                ],
            }
        ],
    )
    train_cases = tmp_path / "train-cases.jsonl"
    _write_jsonl(train_cases, [{"case_id": "located", "split": "train"}])

    output = tmp_path / "candidates"
    build_psd_candidate_package(
        run_dir=run_dir,
        train_cases_path=train_cases,
        rollout_gate_path=_rollout_gate(run_dir, train_cases),
        output_dir=output,
    )

    candidate = json.loads(
        (output / "repair_candidates.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()[0]
    )
    localization = candidate["failure_localization"]
    assert localization["repair_anchor_step_index"] == 0
    assert localization["repair_anchor_source_step_index"] == 1
    assert localization["repair_anchor_step_id"] == (
        "episode-located:react:react-1"
    )


def test_terminal_mismatch_requires_privileged_attribution(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "traces" / "episode-terminal.json",
        _trace(case_id="terminal"),
    )
    _write_jsonl(
        run_dir / "rollout_groups.jsonl",
        [
            {
                "episode_id": "episode-terminal",
                "trace_path": "traces/episode-terminal.json",
            }
        ],
    )
    _write_jsonl(
        run_dir / "post_rollout_rewards.jsonl",
        [
            {
                "case_id": "terminal",
                "episode_id": "episode-terminal",
                "classification_correct": False,
                "fatal_engineering_error": False,
                "strict_trace_audit_pass": True,
            }
        ],
    )
    train_cases = tmp_path / "train-cases.jsonl"
    _write_jsonl(train_cases, [{"case_id": "terminal", "split": "train"}])

    output = tmp_path / "candidates"
    build_psd_candidate_package(
        run_dir=run_dir,
        train_cases_path=train_cases,
        rollout_gate_path=_rollout_gate(run_dir, train_cases),
        output_dir=output,
    )

    candidate = json.loads(
        (output / "repair_candidates.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()[0]
    )
    assert candidate["failure_localization"][
        "requires_privileged_attribution"
    ] is True
    assert candidate["candidate_status"] == "needs_privileged_localization"

from __future__ import annotations

import json
from pathlib import Path

from ifv_training.psd import build_psd_target_package
from ifv_training.psd_repairs import assemble_psd_repair_package


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _repair_candidate(candidate_id: str = "candidate-1") -> dict:
    return {
        "candidate_id": candidate_id,
        "class": "repair_seed",
        "case_id": "case-1",
        "episode_id": "episode-1",
        "repair_site": {
            "step_id": "episode-1:react:1",
            "stage": "image_only_investigation",
            "example_type": "react",
            "rollout_token_capture": {
                "status": "complete",
                "prompt_token_ids": [1, 2, 3],
                "completion_token_ids": [90],
                "completion_logprobs": [-0.5],
            },
        },
        "source": {
            "source_run_id": "run-1",
            "runtime_commit": "abc",
            "source_trace_sha256": "trace-sha",
        },
    }


def _attempt(
    attempt_id: str,
    *,
    hint: str,
    hint_level: int,
    teacher_prompt_ids: list[int] | None = None,
    **overrides: object,
) -> dict:
    row = {
        "candidate_id": "candidate-1",
        "attempt_id": attempt_id,
        "case_id": "case-1",
        "episode_id": "episode-1",
        "repair_step_id": "episode-1:react:1",
        "source_trace_sha256": "trace-sha",
        "hint": hint,
        "hint_level": hint_level,
        "accepted": True,
        "repair_tier": "causal_episode_pass",
        "teacher_prompt_ids": teacher_prompt_ids or [1, 2, 3, 4],
        "completion_ids": [5, 6],
        "verification": {
            "local_pass": True,
            "full_episode_pass": True,
            "strict_trace_audit_pass": True,
        },
    }
    row.update(overrides)
    return row


def _preservation_candidate() -> dict:
    return {
        "candidate_id": "preserve-1",
        "class": "base_pass_preserve",
        "case_id": "case-pass",
        "episode_id": "episode-pass",
        "verified_full_task": True,
        "strict_trace_audit_pass": True,
        "source": {
            "source_run_id": "run-1",
            "runtime_commit": "abc",
            "source_trace_sha256": "pass-sha",
        },
        "preservation_steps": [
            {
                "step_id": "episode-pass:planning:1",
                "stage": "image_only_planning",
                "example_type": "planning",
                "rollout_token_capture": {
                    "status": "complete",
                    "prompt_token_ids": [10, 11],
                    "completion_token_ids": [12],
                    "completion_logprobs": [-0.1],
                },
            },
            {
                "step_id": "episode-pass:react:2",
                "stage": "image_only_investigation",
                "example_type": "react",
                "rollout_token_capture": {
                    "status": "complete",
                    "prompt_token_ids": [10, 11, 12, 13],
                    "completion_token_ids": [14, 15],
                    "completion_logprobs": [-0.2, -0.3],
                },
            },
        ],
    }


def test_assembler_selects_weakest_then_shortest_verified_hint(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "repair_candidates.jsonl"
    attempts = tmp_path / "repair_attempts.jsonl"
    preservation = tmp_path / "preservation_candidates.jsonl"
    _write_jsonl(candidates, [_repair_candidate()])
    _write_jsonl(
        attempts,
        [
            _attempt("l2", hint="Inspect the unresolved relation first.", hint_level=2),
            _attempt(
                "l1-long",
                hint="Review the current evidence and choose one action that can add information.",
                hint_level=1,
            ),
            _attempt("l1-short", hint="Check the unresolved relation.", hint_level=1),
        ],
    )
    _write_jsonl(preservation, [_preservation_candidate()])

    output = tmp_path / "assembled"
    manifest = assemble_psd_repair_package(
        repair_candidates_path=candidates,
        repair_attempts_path=attempts,
        preservation_candidates_path=preservation,
        output_dir=output,
    )

    assert manifest["counts"]["selected_repairs"] == 1
    repair = json.loads(
        (output / "repairs.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert repair["attempt_id"] == "l1-short"
    assert repair["student_prompt_ids"] == [1, 2, 3]
    assert repair["teacher_prompt_ids"] == [1, 2, 3, 4]
    assert repair["source_trace_sha256"] == "trace-sha"
    reasons = [
        json.loads(line)["reason"]
        for line in (output / "attempt_rejections.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert reasons == [
        "superseded_by_weaker_or_shorter_verified_hint",
        "superseded_by_weaker_or_shorter_verified_hint",
    ]

    preserve = json.loads(
        (output / "preservation.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()[0]
    )
    assert preserve["preservation_steps"][0]["student_prompt_ids"] == [10, 11]
    assert preserve["preservation_steps"][1]["completion_ids"] == [14, 15]

    target_output = tmp_path / "targets"
    target_manifest = build_psd_target_package(
        repairs_path=output / "repairs.jsonl",
        preservation_path=output / "preservation.jsonl",
        output_dir=target_output,
    )
    assert target_manifest["counts"]["repair_targets"] == 1
    assert target_manifest["counts"]["preservation_targets"] == 2
    assert target_manifest["counts"]["rejections"] == 0


def test_assembler_rejects_identity_verifier_and_hint_leaks(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "repair_candidates.jsonl"
    attempts = tmp_path / "repair_attempts.jsonl"
    preservation = tmp_path / "preservation_candidates.jsonl"
    _write_jsonl(candidates, [_repair_candidate()])
    _write_jsonl(
        attempts,
        [
            _attempt(
                "wrong-trace",
                hint="Inspect another route.",
                hint_level=1,
                source_trace_sha256="other",
            ),
            _attempt(
                "not-full",
                hint="Inspect another route.",
                hint_level=1,
                verification={
                    "local_pass": True,
                    "full_episode_pass": False,
                    "strict_trace_audit_pass": True,
                },
            ),
            _attempt(
                "leak",
                hint="The answer is fake.",
                hint_level=1,
            ),
        ],
    )
    _write_jsonl(preservation, [])

    output = tmp_path / "assembled"
    manifest = assemble_psd_repair_package(
        repair_candidates_path=candidates,
        repair_attempts_path=attempts,
        preservation_candidates_path=preservation,
        output_dir=output,
    )

    assert manifest["counts"]["selected_repairs"] == 0
    assert manifest["counts"]["unrepaired_candidates"] == 1
    reasons = {
        json.loads(line)["reason"]
        for line in (output / "attempt_rejections.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    assert reasons == {
        "source_trace_sha256_mismatch",
        "verification_full_episode_pass_required",
        "hint_audit_failed:binary_verdict_leak",
    }


def test_assembler_uses_complete_repair_rollout_capture(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "repair_candidates.jsonl"
    attempts = tmp_path / "repair_attempts.jsonl"
    preservation = tmp_path / "preservation_candidates.jsonl"
    _write_jsonl(candidates, [_repair_candidate()])
    attempt = _attempt(
        "captured",
        hint="Inspect another unresolved route.",
        hint_level=1,
    )
    attempt.pop("teacher_prompt_ids")
    attempt.pop("completion_ids")
    attempt["repair_rollout_token_capture"] = {
        "status": "complete",
        "prompt_token_ids": [20, 21],
        "completion_token_ids": [22],
    }
    _write_jsonl(attempts, [attempt])
    _write_jsonl(preservation, [])

    output = tmp_path / "assembled"
    assemble_psd_repair_package(
        repair_candidates_path=candidates,
        repair_attempts_path=attempts,
        preservation_candidates_path=preservation,
        output_dir=output,
    )

    repair = json.loads(
        (output / "repairs.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert repair["teacher_prompt_ids"] == [20, 21]
    assert repair["completion_ids"] == [22]

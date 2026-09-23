import gzip
import json
from pathlib import Path

import pytest

from scripts.server.audit_psd_frozen_repair_index import audit_frozen_repairs, _stat


SHA = "d" * 64
MODEL = "ifv-qwen3.5-9b-sft-3084"
CHECKPOINT = "/frozen/sft3/model"


def _sample() -> dict:
    return {
        "accepted": True, "scaffold_only": False, "case_id": "main-1",
        "attempt_id": "accepted-1", "hinted_episode_pass": True,
        "hinted_local_pass": True, "hinted_strict_trace_audit_pass": True,
        "verification": {"expected_verdict": "real", "hinted_recorded_verdict": "real", "reasons": []},
        "local_verification": {"passed": True, "checks": [{"name": "strict", "passed": True}]},
        "teacher_prompt_ids": [1], "student_prompt_ids": [2], "completion_ids": [3, 4],
        "psd_media": {"image_paths": ["/frozen/image.png"]},
        "model_roles": {
            "frozen_self_teacher": {"provider": "qwen_local", "model": MODEL,
                                    "checkpoint_manifest_sha256": SHA,
                                    "round_start_checkpoint": CHECKPOINT},
            "trainable_student": {"provider": "qwen_local", "model": MODEL,
                                  "checkpoint_manifest_sha256": SHA,
                                  "initial_checkpoint": CHECKPOINT},
            "hint_constructor": {"provider": "gemini", "model": "gemini-3.7-flash"}},
        "hint_record": {"model": "gemini-3.7-flash"}}


def _input(tmp_path: Path, row: dict, *, accepted_count: int = 1) -> tuple[Path, Path]:
    run = tmp_path / "run"
    run.mkdir()
    attempts = run / "repair_attempts.jsonl.gz"
    with gzip.open(attempts, "wt", encoding="utf-8") as output:
        output.write(json.dumps(row) + "\n")
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"case_id": "main-1", "accepted_count": accepted_count,
        "attempts_path": str(attempts), "attempts_identity": _stat(attempts)}) + "\n")
    return index, run


def _audit(index: Path, run: Path, destination: Path):
    return audit_frozen_repairs(frozen_index=index, output_dir=destination,
        run_root=run, model=MODEL, checkpoint_sha=SHA, checkpoint=CHECKPOINT)


def test_audits_metadata_only_and_keeps_gemini_provenance(tmp_path):
    index, run = _input(tmp_path, _sample())
    destination = tmp_path / "audit"
    result = _audit(index, run, destination)
    assert result["accepted_cases"] == result["accepted_targets"] == 1
    assert result["gemini_hint_models"] == {"gemini-3.7-flash": 1}
    target = json.loads((destination / "accepted-target-index.jsonl").read_text())
    assert target["verification"] == "strict_complete_pass"
    assert "completion_ids" not in target and "image_paths" not in target


def test_text_only_step_needs_no_media_archive(tmp_path):
    row = _sample()
    row["psd_media"] = None
    index, run = _input(tmp_path, row)
    result = _audit(index, run, tmp_path / "audit")
    assert result["accepted_targets"] == 1


def test_image_step_must_bind_pixels(tmp_path):
    row = _sample()
    row["teacher_prompt_ids"] = [248056, 1]
    row["psd_media"] = None
    index, run = _input(tmp_path, row)
    with pytest.raises(ValueError, match="image-bearing"):
        _audit(index, run, tmp_path / "audit")


@pytest.mark.parametrize("mutation", [
    lambda r: r["verification"].update(hinted_recorded_verdict="fake"),
    lambda r: r["model_roles"]["trainable_student"].update(checkpoint_manifest_sha256="e" * 64),
    lambda r: r["hint_record"].update(model="gemini-3.6-flash"),
    lambda r: r["local_verification"]["checks"][0].update(passed=False),
])
def test_rejects_unverified_or_mixed_targets(tmp_path, mutation):
    row = _sample()
    mutation(row)
    index, run = _input(tmp_path, row)
    with pytest.raises(ValueError):
        _audit(index, run, tmp_path / "audit")
    assert not (tmp_path / "audit").exists()


def test_rejects_count_or_source_change(tmp_path):
    index, run = _input(tmp_path, _sample(), accepted_count=2)
    with pytest.raises(ValueError, match="disagree"):
        _audit(index, run, tmp_path / "audit")
    row = json.loads(index.read_text())
    row["attempts_identity"]["bytes"] += 1
    index.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="stat changed"):
        _audit(index, run, tmp_path / "audit")

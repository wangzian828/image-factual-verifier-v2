from pathlib import Path
import pytest
from ifv_training.io import write_json
from ifv_training.psd_resume import bind_resume


def test_same_round_resume_requires_matching_bank_and_complete_state(tmp_path):
    for name in ("datums", "initialization", "profile"):
        write_json(tmp_path / (name + ".json"), {"passed": True, "identity": name})
    args = dict(datum_manifest_path=tmp_path / "datums.json",
        initialization_gate_path=tmp_path / "initialization.json", profile_gate_path=tmp_path / "profile.json")
    root = tmp_path / "run-1"
    bind_resume(output_root=root, **args)
    checkpoint = root / "version-1/checkpoint-1"
    write_json(checkpoint / "trainer_state.json", {"global_step": 1})
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(b"unit test state")
    assert bind_resume(output_root=tmp_path / "run-resume", checkpoint=checkpoint, **args)["passed"]
    with pytest.raises(ValueError, match="another PSD"):
        bind_resume(output_root=tmp_path / "different-recipe", checkpoint=checkpoint, seed=42, **args)
    (checkpoint / "rng_state.pth").unlink()
    with pytest.raises(ValueError, match="complete"):
        bind_resume(output_root=tmp_path / "missing-state", checkpoint=checkpoint, **args)
    write_json(args["datum_manifest_path"], {"changed": True})
    with pytest.raises(ValueError, match="another PSD"):
        bind_resume(output_root=tmp_path / "different-bank", checkpoint=checkpoint, **args)

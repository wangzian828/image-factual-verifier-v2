from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import run_psd_round
from ifv_training.io import sha256_file, write_json, write_jsonl


def test_training_entrypoint_carries_adapter_and_same_round_resume(monkeypatch, tmp_path):
    ready = {"round_index": 2, "serving_profile": str(tmp_path / "serving.json"),
        "checkpoint_manifest": str(tmp_path / "checkpoint.json"),
        "datums": str(tmp_path / "datums.jsonl"), "adapter": str(tmp_path / "previous-adapter"),
        "model": str(tmp_path / "base-model")}
    monkeypatch.setattr(run_psd_round, "load_ready", lambda _: ready)
    calls = []
    monkeypatch.setattr(run_psd_round.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    args = SimpleNamespace(ready=tmp_path / "ready.json", model_profile=tmp_path / "model.env",
        psd_profile=tmp_path / "psd.env", experiment_id="test-round-2",
        resume_checkpoint=tmp_path / "same-round/checkpoint-1",
        training_data_root=tmp_path)
    assert run_psd_round.train(args)["round_index"] == 2
    command, kwargs = calls[0]
    assert command[-1] == str(args.resume_checkpoint.resolve())
    assert kwargs["env"]["IFV_PSD_INITIAL_ADAPTER"] == ready["adapter"]
    assert kwargs["env"]["IFV_PSD_ROUND_READY"] == str(args.ready.resolve())
    assert kwargs["env"]["IFV_MODEL_ID"] == ready["model"]
    assert kwargs["env"]["IFV_TRAINING_DATA_ROOT"] == str(tmp_path.resolve())
    assert kwargs["check"] is True


def test_finalization_rejects_an_unrelated_optimizer_initialization(monkeypatch, tmp_path):
    monkeypatch.setattr(run_psd_round, "load_ready", lambda _: {"initialization": {"checkpoint": "round-1"}})
    gate = tmp_path / "gate.json"
    write_json(gate, {"checkpoint": "other-round"})
    with pytest.raises(ValueError, match="initialization differs"):
        run_psd_round.finalize(SimpleNamespace(ready=tmp_path / "ready.json", initialization_gate=gate))


def test_finalization_separates_deployable_adapter_from_training_state(
    monkeypatch, tmp_path
):
    initialization = {"checkpoint": "round-1"}
    ready = {
        "round_index": 1,
        "initialization": initialization,
        "datum_manifest": str(tmp_path / "datum-manifest.json"),
        "model": str(tmp_path / "base-model"),
        "checkpoint_manifest": str(tmp_path / "base-manifest.json"),
        "rollout_gate": str(tmp_path / "rollout-gate.json"),
    }
    gate = tmp_path / "initialization.json"
    write_json(gate, initialization)
    write_json(tmp_path / "base-manifest.json", {"checkpoint": {"path": ready["model"]}})
    monkeypatch.setattr(run_psd_round, "load_ready", lambda _: ready)
    captured = {}

    def build_manifest(**kwargs):
        captured.update(kwargs)
        write_json(kwargs["output_path"], {"checkpoint": {"path": str(kwargs["checkpoint_dir"])}})
        return {"checkpoint": {"path": str(kwargs["checkpoint_dir"])}}

    monkeypatch.setattr(run_psd_round, "build_checkpoint_manifest", build_manifest)
    monkeypatch.setattr(run_psd_round, "frozen_base_binding", lambda *_: {})
    monkeypatch.setattr(run_psd_round, "complete_psd_round", lambda **_: {"passed": True})
    adapter = tmp_path / "adapter-export"
    state = tmp_path / "checkpoint-10"
    result = run_psd_round.finalize(SimpleNamespace(
        ready=tmp_path / "ready.json",
        checkpoint=adapter,
        state_checkpoint=state,
        training_profile=tmp_path / "profile.json",
        initialization_gate=gate,
        output=tmp_path / "round-output",
    ))
    assert result["status"] == "round_completed"
    assert captured["checkpoint_dir"] == adapter
    assert captured["state_checkpoint_dir"] == state
    assert result["next_round"]["adapter"] == str(adapter.resolve())


def _completed_bank_fixture(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    train_cases = tmp_path / "train-cases.jsonl"
    write_json(run_dir / "run_manifest.json", {"run_id": "round-1-rollout"})
    write_jsonl(run_dir / "post_rollout_rewards.jsonl", [{"case_id": "train-1"}])
    write_jsonl(run_dir / "rollout_groups.jsonl", [{"case_id": "train-1"}])
    write_jsonl(train_cases, [{"case_id": "train-1", "split": "train"}])
    gate = tmp_path / "rollout-gate.json"
    write_json(gate, {
        "schema_version": run_psd_round.PSD_ROLLOUT_GATE_SCHEMA_VERSION,
        "round_index": 1,
        "passed": True,
        "run": {
            "directory": str(run_dir.resolve()),
            "run_id": "round-1-rollout",
            "manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            "post_rollout_rewards_sha256": sha256_file(run_dir / "post_rollout_rewards.jsonl"),
            "rollout_groups_sha256": sha256_file(run_dir / "rollout_groups.jsonl"),
        },
        "train_cases": {
            "path": str(train_cases.resolve()),
            "sha256": sha256_file(train_cases),
        },
    })
    target_dir = tmp_path / "targets"
    targets = target_dir / "targets.jsonl"
    target = {
        "target_status": "complete",
        "source": {
            "psd_rollout_gate_sha256": sha256_file(gate),
            "psd_round_index": 1,
            "source_run_id": "round-1-rollout",
        },
    }
    write_jsonl(targets, [target])
    datum_dir = tmp_path / "datums"
    datums = datum_dir / "datums.jsonl"
    write_jsonl(datums, [{"target_id": "target-1"}])
    write_json(datum_dir / "manifest.json", {
        "source": {"targets": str(targets.resolve()), "targets_sha256": sha256_file(targets)},
    })
    snapshot = tmp_path / "snapshot"
    write_json(snapshot / "serving-profile.json", {
        "profile_id": "model", "model_path": str((tmp_path / "model").resolve())})
    write_json(snapshot / "checkpoint-manifest.json", {"checkpoint": {"path": "model"}})
    datum_gate = {"passed": True, "datums": {"rows": 1, "sha256": sha256_file(datums)}}
    initialization = {"schema_version": "ifv-psd-initialization-gate-v1", "passed": True}
    monkeypatch.setattr(run_psd_round, "verify_psd_training_input", lambda **_: datum_gate)
    monkeypatch.setattr(run_psd_round, "verify_initialization", lambda **_: initialization)
    return gate, datums, snapshot, target


def test_attest_completed_bank_binds_rollout_targets_and_initialization(monkeypatch, tmp_path):
    gate, datums, snapshot, _ = _completed_bank_fixture(tmp_path, monkeypatch)
    output = tmp_path / "attested"
    result = run_psd_round.attest(SimpleNamespace(
        rollout_gate=gate, datums=datums, datum_manifest=None,
        snapshot=snapshot, output=output))
    assert result["status"] == "ready_for_training"
    ready = run_psd_round.load_ready(output / "ready.json")
    assert ready["round_index"] == 1
    assert ready["preparation_mode"] == "attested_completed_bank_without_resampling"
    assert run_psd_round.load_json(output / "attestation.json")["target_count"] == 1


def test_attest_completed_bank_rejects_target_from_another_rollout(monkeypatch, tmp_path):
    gate, datums, snapshot, target = _completed_bank_fixture(tmp_path, monkeypatch)
    targets = tmp_path / "targets" / "targets.jsonl"
    target["source"]["psd_rollout_gate_sha256"] = "0" * 64
    write_jsonl(targets, [target])
    manifest = tmp_path / "datums" / "manifest.json"
    write_json(manifest, {"source": {
        "targets": str(targets.resolve()), "targets_sha256": sha256_file(targets)}})
    with pytest.raises(ValueError, match="target lineage differs"):
        run_psd_round.attest(SimpleNamespace(
            rollout_gate=gate, datums=datums, datum_manifest=manifest,
            snapshot=snapshot, output=tmp_path / "rejected"))

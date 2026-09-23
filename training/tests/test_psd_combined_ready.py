from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import psd_combined_ready as combined
from scripts.server.run_psd_dp4_resume_gate import ready_inputs
from ifv_training.io import write_json


def test_combined_completion_requires_native_gate_and_actual_profile(tmp_path):
    datums = tmp_path / "datums.jsonl"
    datums.write_text("fixture\n")
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"targets_by_kind": {"repair": 1598, "preserve": 1598}})
    resume = tmp_path / "resume.json"
    write_json(resume, {"adapter_bitwise_equal": True})
    init = {"passed": True, "checkpoint_manifest_sha256": combined.CHECKPOINT_SHA}
    init_path = tmp_path / "init.json"
    write_json(init_path, init)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    ready = {"schema_version": combined.SCHEMA, "initialization": init,
        "datums": str(datums), "datum_manifest": str(manifest),
        "native_resume_result": str(resume)}
    profile = tmp_path / "profile.json"
    write_json(profile, {"passed_production_gate": True, "smoke_only": False,
        "parallelism": {"world_size": 4, "data_parallel_size": 4,
            "unique_samples_per_optimizer_step": 32},
        "train_step_metric_rows": 500, "loss": {"count": 500, "last": 1.0},
        "raw_dataset_gate": {"passed": True, "verification": {"passed": True,
            "datums": {"path": str(datums), "rows": 3196}}}})
    checkpoint = tmp_path / "checkpoint.json"
    write_json(checkpoint, {"checkpoint": {"global_step": 500,
        "path": str(adapter), "optimizer_state_available": True,
        "scheduler_state_available": True, "rng_state_available": True},
        "training_dataset": {"manifest_sha256": combined.sha256_file(manifest)},
        "artifacts": [{"scope": "training_state"},
            {"scope": "model", "path": "adapter_model.safetensors"}]})
    args = dict(ready=ready, profile_path=profile,
        checkpoint_manifest_path=checkpoint, initialization_gate_path=init_path,
        output=tmp_path / "completion.json")
    assert combined.complete(**args)["passed"] is True
    write_json(resume, {"adapter_bitwise_equal": False})
    assert combined.complete(**args)["passed"] is False
    write_json(resume, {"adapter_bitwise_equal": True})
    bad = json.loads(profile.read_text())
    bad["loss"]["last"] = float("nan")
    write_json(profile, bad)
    assert combined.complete(**args)["passed"] is False


def test_combined_ready_rejects_synthetic_single_lineage(tmp_path, monkeypatch):
    root = tmp_path
    snapshot = root / "snapshot"
    snapshot.mkdir()
    model = root / "exports/h20-sft-merged4872-3epoch-step3084-20260915/model"
    model.mkdir(parents=True)
    write_json(snapshot / "serving-profile.json", {"profile_id": combined.MODEL,
        "base_url": "http://127.0.0.1:19025/v1", "engine_model_path": str(model)})
    write_json(snapshot / "checkpoint-manifest.json", {})
    scored = root / "scored"
    scored.mkdir()
    (scored / "targets.jsonl").write_text("fixture\n")
    source = root / "source.json"
    source.write_text("{}")
    st = source.stat()
    write_json(scored / "manifest.json", {"schema_version": "ifv-psd-combined-sft3-scored-v1",
        "targets": 3196, "repair_targets": 1598, "preservation_targets": 1598,
        "checkpoint_manifest_sha256": combined.CHECKPOINT_SHA,
        "teacher_logprob_semantics": "pre_grammar_unprocessed_v1",
        "source_stat_bindings": {str(source): {"device": st.st_dev, "inode": st.st_ino,
            "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}}})
    small = root / "small.json"
    old = root / "old.json"
    freeze = root / "freeze.json"
    selection = root / "selection.json"
    score = root / "score.json"
    write_json(small, {"counts": {"repair": 626, "preserve": 626}})
    write_json(old, {"counts": {"repair": 972, "preserve": 972}})
    write_json(freeze, {"accepted_repair_targets": 990})
    write_json(selection, {"metadata_coverage_gate_pass": True,
        "uses_gold_or_teacher_score": False,
        "status": "metadata_only_not_authorized_to_score_or_train"})
    write_json(score, {"phase": "raw_teacher_scoring_complete", "services_restored": True})
    datums = root / "datums.jsonl"
    datums.write_text("fixture\n")
    write_json(root / "manifest.json", {"source": {"targets": str(scored / "targets.jsonl")},
        "targets_by_kind": {"repair": 1598, "preserve": 1598},
        "counts": {"rejections": 0}})
    monkeypatch.setattr(combined, "verify_psd_training_input", lambda **_: {
        "passed": True, "datums": {"rows": 3196,
            "by_kind": {"repair": 1598, "preserve": 1598}}})
    monkeypatch.setattr(combined, "verify_initialization", lambda **_: {
        "passed": True, "targets": 3196,
        "checkpoint_manifest_sha256": combined.CHECKPOINT_SHA})
    args = dict(datums=datums, scored_manifest=scored / "manifest.json",
        small_manifest=small, old_manifest=old, selection_manifest=selection,
        freeze_manifest=freeze, score_state=score, snapshot=snapshot,
        output=root / "runs/combined", root=root)
    with pytest.raises(ValueError, match="source-stat"):
        combined.attest(**args)


def test_native_dp4_gate_accepts_two_bound_sources_without_synthetic_rollout(tmp_path):
    root = tmp_path
    run = root / "runs/combined"
    run.mkdir(parents=True)
    ready_file = run / "ready.json"
    ready_file.write_text("{}")
    datums = root / "datums/datums.jsonl"
    datums.parent.mkdir()
    datums.write_text("fixture\n")
    (datums.parent / "manifest.json").write_text("{}")
    snapshot = root / "snapshot"
    snapshot.mkdir()
    write_json(snapshot / "serving-profile.json", {"base_url": "http://127.0.0.1:19025/v1"})
    write_json(snapshot / "checkpoint-manifest.json", {})
    sources = [root / "small.json", root / "old.json"]
    for path in sources:
        path.write_text("{}")
    model = root / "exports/h20-sft-merged4872-3epoch-step3084-20260915/model"
    model.mkdir(parents=True)
    ready = {"schema_version": combined.SCHEMA, "datums": str(datums),
        "datum_manifest": str(datums.parent / "manifest.json"),
        "serving_profile": str(snapshot / "serving-profile.json"),
        "checkpoint_manifest": str(snapshot / "checkpoint-manifest.json"),
        "source_manifests": [str(path) for path in sources],
        "native_resume_result": str(run / "dp4-resume-gate-combined-v1/result.json"),
        "model": str(model), "adapter": None}
    selected = ready_inputs(ready_file, root=root, load_ready=lambda _: ready)
    assert selected["datums"] == datums
    ready["native_resume_result"] = str(run / "wrong/result.json")
    with pytest.raises(ValueError, match="native resume"):
        ready_inputs(ready_file, root=root, load_ready=lambda _: ready)

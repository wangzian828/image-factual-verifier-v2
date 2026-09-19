import json

from scripts.server.control_psd_posttrain_next1000 import diagnostics


def test_smallbank_diagnostics_require_real_training_progress(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"changed adapter")
    profile = tmp_path / "profile.json"
    completion = tmp_path / "completion.json"
    command = " ".join(("--lora_rank 32", "--lora_alpha 32", "--lora_dropout 0.0",
        "--learning_rate 4e-5", "--adam_beta1 0.9", "--adam_beta2 0.95",
        "--adam_epsilon 1e-12", "--weight_decay 0", "--num_train_epochs 5"))
    profile.write_text(json.dumps({"passed_production_gate": True,
        "steps": {"complete": True, "last_observed_epoch": 5.0},
        "loss": {"count": 10, "last": 2.0},
        "parallelism": {"world_size": 4, "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4},
        "resources": {"summary": {"selected_physical_gpu_ids": [0, 1, 2, 3]}},
        "launch": {"command": command}}))
    names = ("training_production_gate", "initialization_gate_passed",
        "optimizer_step_completed", "resumable_training_state", "output_artifacts_intact",
        "weight_artifacts_changed", "checkpoint_changed")
    completion.write_text(json.dumps({"checks": {name: {"passed": True} for name in names}}))
    state = {"training": {"profile": str(profile), "completion": str(completion),
        "adapter": str(adapter), "checkpoint": str(tmp_path / "checkpoint"),
        "formal_training": True}}
    report = diagnostics(state)
    assert report["passed"] and report["global_batch_32"] and report["five_epochs"]
    changed = json.loads(profile.read_text())
    changed["loss"]["last"] = float("nan")
    profile.write_text(json.dumps(changed))
    assert not diagnostics(state)["passed"]

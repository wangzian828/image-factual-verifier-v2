from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from scripts.server import run_psd_combined_cached_full_eval as flow


def original_command(root: Path) -> list[str]:
    return [
        "python", "vllm", "serve", str(root), "--served-model-name",
        flow.agent.MODEL_ALIAS, "--tool-call-parser", "ifv_psd_qwen3_single",
        "--mamba-cache-mode", "none", "--no-enable-prefix-caching",
    ]


def test_candidate_preserves_parser_and_enables_safe_cache(monkeypatch, tmp_path):
    sft = tmp_path / "sft"
    merged = tmp_path / "merged"
    monkeypatch.setattr(flow, "SFT_ROOT", sft)
    original = original_command(sft)
    candidate = flow.candidate_command(original, merged)
    assert original[3] == str(sft)
    assert candidate[3] == str(merged.resolve())
    assert candidate[candidate.index("--served-model-name") + 1] == flow.agent.MODEL_ALIAS
    assert candidate[candidate.index("--tool-call-parser") + 1] == "ifv_psd_qwen3_single"
    assert candidate[candidate.index("--mamba-cache-mode") + 1] == "align"
    assert "--enable-prefix-caching" in candidate
    assert "--no-enable-prefix-caching" not in candidate


def test_candidate_rejects_wrong_source_and_alias(monkeypatch, tmp_path):
    sft = tmp_path / "sft"
    monkeypatch.setattr(flow, "SFT_ROOT", sft)
    with pytest.raises(ValueError, match="frozen SFT3"):
        flow.candidate_command(original_command(tmp_path / "other"), tmp_path / "new")
    wrong = original_command(sft)
    wrong[wrong.index("--served-model-name") + 1] = "wrong-model"
    with pytest.raises(ValueError, match="unexpected internal alias"):
        flow.candidate_command(wrong, tmp_path / "new")


def test_replica_compilation_cache_is_private_and_fresh(tmp_path):
    original = {"VLLM_CACHE_ROOT": "/old/shared", "OTHER": "unchanged"}
    first = flow.candidate_environment(original, tmp_path, 0)
    second = flow.candidate_environment(original, tmp_path, 1)
    assert first["VLLM_CACHE_ROOT"] != second["VLLM_CACHE_ROOT"]
    assert first["VLLM_COMPUTE_NANS_IN_LOGITS"] == "1"
    assert first["OTHER"] == "unchanged"
    assert original["VLLM_CACHE_ROOT"] == "/old/shared"
    with pytest.raises(ValueError, match="replica index"):
        flow.candidate_environment(original, tmp_path, 4)


def test_gateway_is_case_isolated_with_new_public_identity():
    original = {"PYTHONPATH": "/old", "PSD_PUBLIC_MODEL_ALIAS": "old"}
    candidate = flow.gateway_environment(original)
    assert candidate["PSD_PUBLIC_MODEL_ALIAS"] == flow.PROFILE_MODEL
    assert candidate["IFV_PREFIX_CACHE_MODE"] == "case_isolated"
    assert candidate["IFV_PREFIX_CACHE_BLOCK_SIZE"] == "528"
    assert candidate["IFV_PREFIX_CACHE_UNSAFE_WINDOW"] == "16"
    assert candidate["PYTHONPATH"].startswith(str(flow.GATEWAY_CODE) + ":")
    assert original["PSD_PUBLIC_MODEL_ALIAS"] == "old"


def test_new_weight_cannot_borrow_an_old_cache_verdict():
    valid = {
        "passed": True, "block_size": 528, "unsafe_window": 16,
        "boundary_rows": [{"cache_behavior_passed": True} for _ in range(17)],
        "incremental_multimodal": {"cache_hit_observed": True},
        "stress": {"corrupted_delta": 0, "errors": [], "healthy": 32},
    }
    flow.attest_new_cache(valid)
    with pytest.raises(RuntimeError, match="real multimodal cache hit"):
        flow.attest_new_cache({**valid, "incremental_multimodal": {}})
    with pytest.raises(RuntimeError, match="corruption"):
        flow.attest_new_cache({**valid, "stress": {"corrupted_delta": 1}})
    with pytest.raises(RuntimeError, match="geometry"):
        flow.attest_new_cache({**valid, "block_size": 512})


def test_new_gateway_requires_corruption_rejection_and_new_identity():
    valid = {
        "prefix_cache_policy": "case_isolated",
        "public_model_alias": flow.PROFILE_MODEL,
        "reject_corrupted_responses": True,
        "prefix_cache_block_size": 528,
        "prefix_cache_unsafe_window": 16,
        "cache_corruption_quarantines": 0,
        "cache_corruption_metric_failures": 0,
    }
    flow.attest_new_gateway_health(valid)
    for key, value in (
        ("public_model_alias", "old"),
        ("reject_corrupted_responses", False),
        ("cache_corruption_metric_failures", 1),
        ("cache_corruption_metric_failures", None),
        ("cache_corruption_quarantines", 1),
    ):
        with pytest.raises(RuntimeError, match="gateway cache safety"):
            flow.attest_new_gateway_health({**valid, key: value})


def test_all_three_ablation_variants_are_distinct_and_manifest_bound(tmp_path):
    names = [name for name, _, _ in flow.ABLATIONS]
    assert names == ["no-web-search", "no-image-retrieval", "no-evidence-inspection"]
    assert len({flow.ablation_output(name) for name in names}) == 3
    for name, flag, config in flow.ABLATIONS:
        assert flag == "--disable-" + name.removeprefix("no-")
        manifest = tmp_path / (name + ".json")
        manifest.write_text(json.dumps({"agent": {"tool_families": config.to_manifest()}}))
        flow.attest_ablation_run_manifest(manifest, config)
        wrong = tmp_path / (name + "-wrong.json")
        wrong.write_text(json.dumps({"agent": {"tool_families": {}}}))
        with pytest.raises(RuntimeError, match="manifest mismatch"):
            flow.attest_ablation_run_manifest(wrong, config)


def test_ablation_flags_reach_frozen_case_runner(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(flow.agent.subprocess, "run", fake_run)
    monkeypatch.setattr(flow.agent, "BENCHMARK", tmp_path / "frozen.jsonl")
    flow.agent.run_attempt(
        deploy=tmp_path / "deploy", source_code=tmp_path / "code",
        output=tmp_path, name="smoke-0", cases=["case-1"], concurrency=1,
        base_seed=3903, environment={"QWEN35_LOCAL_MODEL": flow.PROFILE_MODEL},
        tool_ablation_flags=("--disable-web-search",),
    )
    assert "--disable-web-search" in commands[0]
    assert commands[0][commands[0].index("--case-list") + 1].endswith("smoke-0-cases.txt")


def test_case_isolated_runtime_opt_in_overrides_default_and_private_env(monkeypatch, tmp_path):
    for key in (
        "SERPER_API_KEY", "JINA_API_KEY", "GEMINI_API_KEY",
        "BAIDU_OCR_API_KEY", "BAIDU_OCR_SECRET_KEY",
    ):
        monkeypatch.setenv(key, "test-key")
    monkeypatch.setenv("IFV_PREFIX_CACHE_MODE", "request_isolated")
    monkeypatch.setenv("PERCEPTION_CACHE_ENABLED", "1")
    monkeypatch.setenv("TOOL_CACHE_ENABLED", "1")
    environment = flow.agent.runtime_environment(
        tmp_path, flow.PROFILE_MODEL, prefix_cache_mode="case_isolated",
    )
    assert environment["IFV_PREFIX_CACHE_MODE"] == "case_isolated"
    assert environment["QWEN35_LOCAL_MODEL"] == flow.PROFILE_MODEL
    assert environment["PERCEPTION_CACHE_ENABLED"] == "0"
    assert environment["TOOL_CACHE_ENABLED"] == "0"
    with pytest.raises(ValueError, match="only opts into case-isolated"):
        flow.agent.runtime_environment(
            tmp_path, flow.PROFILE_MODEL, prefix_cache_mode="request_isolated",
        )


def test_smoke_rejects_successful_trace_with_visual_gateway_400(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "case.json").write_text(json.dumps({"state": {"all_steps": [{
        "action_type": "tool_call", "tool_name": "perceive_scene",
        "tool_args": {}, "tool_result": "400 Bad Request",
        "metadata": {"tool_execution_status": "failed"},
    }]}}))
    selected = {"case": ({
        "fact_check_report": "A valid structured report.",
        "total_tool_calls": 1,
        "trace_path": "traces/case.json",
    }, tmp_path)}
    report = flow.agent.anomaly_report(selected, ["case"])
    assert report["passed"] is False
    assert "visual_tool_gateway_rejected" in report["records"][0]["issues"]


def test_smoke_rejects_cached_perception_even_when_tool_succeeded(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "case.json").write_text(json.dumps({"state": {"all_steps": [{
        "action_type": "tool_call", "tool_name": "perceive_scene",
        "tool_args": {}, "tool_result": '{"status":"success"}',
        "metadata": {"cache_hit": True, "tool_success": True},
    }]}}))
    selected = {"case": ({
        "fact_check_report": "A valid structured report.",
        "total_tool_calls": 1,
        "trace_path": "traces/case.json",
    }, tmp_path)}
    report = flow.agent.anomaly_report(selected, ["case"])
    assert report["passed"] is False
    assert "cached_perception_in_fresh_smoke" in report["records"][0]["issues"]


def test_existing_merged_model_requires_source_receipt_before_reuse(monkeypatch, tmp_path):
    sft = tmp_path / "sft"
    adapter = tmp_path / "adapter"
    merged = tmp_path / "merged"
    deploy = tmp_path / "deploy"
    for directory in (sft, adapter, merged, deploy):
        directory.mkdir()
    (sft / "config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (merged / "config.json").write_text("{}")
    monkeypatch.setattr(flow, "SFT_ROOT", sft)
    monkeypatch.setattr(flow, "MERGED_ROOT", merged)
    with pytest.raises(RuntimeError, match="no source receipt"):
        flow.merge_model(adapter, deploy)
    (merged / "merge-export.json").write_text(json.dumps({
        "passed": True, "large_payload_hashing": False,
        "source": {"base_model": str(sft), "adapter": str(adapter)},
        "adapter_parameter_count": 1,
    }))
    monkeypatch.setattr(flow.subprocess, "run", lambda *a, **k: pytest.fail("merge reran"))
    flow.merge_model(adapter, deploy)
    assert (deploy / "merge-reuse.json").is_file()
    assert (deploy / "merge-binding.json").is_file()

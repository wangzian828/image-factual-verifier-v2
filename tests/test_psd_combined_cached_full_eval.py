from pathlib import Path

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

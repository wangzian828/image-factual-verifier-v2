from scripts.server.probe_prefix_cache_hybrid_safety import (
    probe_passed,
    prometheus_counter,
    response_health,
    shape_text_for_exact_count,
)


def test_prometheus_counter_accepts_counter_suffix_and_sums_labels():
    text = """
# HELP vllm:prefix_cache_hits Prefix hits
vllm:prefix_cache_hits_total{model_name="a"} 528
vllm:prefix_cache_hits_total{model_name="b"} 1056
vllm:other_total 999
"""
    assert prometheus_counter(text, "vllm:prefix_cache_hits") == 1584


def test_exact_prompt_shaper_uses_tokenizer_observations():
    calls = []

    def count(text: str) -> int:
        calls.append(text)
        return 7 + text.count(" x")

    text, observed = shape_text_for_exact_count(count, 535)
    assert observed == 535
    assert count(text) == 535
    assert len(calls) < 20


def test_response_health_rejects_nan_and_repeated_zero_tokens():
    valid = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
                "token_ids": [1, 2, 3],
            }
        ]
    }
    assert response_health(valid)["healthy"] is True
    invalid = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"role": "assistant", "content": "!!!!"},
                "token_ids": [0, 0, 0, 0],
            }
        ],
        "nan": float("nan"),
    }
    observed = response_health(invalid)
    assert observed["healthy"] is False
    assert observed["longest_zero_run"] == 4
    assert observed["finite"] is False


def healthy_row(*, hit: bool, rotated: bool = False, replica: str = "r0"):
    health = {"healthy": True}
    return {
        "first_health": health,
        "second_health": health,
        "same_replica": True,
        "expected_rotation": rotated,
        "cache_hit_observed": hit,
        "cache_behavior_passed": not (rotated and hit),
        "replica": replica,
    }


def test_probe_allows_retryable_stress_errors_but_not_corruption():
    boundary = [healthy_row(hit=True), healthy_row(hit=False, rotated=True)]
    multimodal = healthy_row(hit=True)
    stress = [
        healthy_row(hit=True, replica=f"r{index % 4}") for index in range(31)
    ]
    errors = [{"error_type": "TimeoutError", "error": "timed out"}]
    assert probe_passed(
        boundary_rows=boundary,
        multimodal=multimodal,
        stress_rows=stress,
        stress_errors=errors,
        requested_stress_domains=32,
        corrupted_delta=0,
    )
    assert not probe_passed(
        boundary_rows=boundary,
        multimodal=multimodal,
        stress_rows=stress,
        stress_errors=errors,
        requested_stress_domains=32,
        corrupted_delta=1,
    )

from scripts.server.probe_prefix_cache_hybrid_safety import (
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
    def count(text: str) -> int:
        return 7 + text.count(" x")

    text, observed = shape_text_for_exact_count(count, 535)
    assert observed == 535
    assert count(text) == 535


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

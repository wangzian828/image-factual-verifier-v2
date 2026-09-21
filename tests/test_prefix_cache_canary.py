import json

from scripts.server.run_prefix_cache_canary import (
    cache_on_command,
    metric_delta,
    response_is_well_formed,
)


def test_cache_on_command_replaces_only_cache_flags():
    original = [
        "python",
        "vllm",
        "serve",
        "/model",
        "--port",
        "19002",
        "--no-enable-prefix-caching",
        "--mamba-cache-mode",
        "none",
    ]
    result = cache_on_command(original)
    assert "--no-enable-prefix-caching" not in result
    assert result.count("--enable-prefix-caching") == 1
    assert result[result.index("--mamba-cache-mode") + 1] == "align"
    assert result[:6] == original[:6]


def test_metric_delta_is_per_request():
    assert metric_delta(
        {"hits": 528.0, "queries": 1056.0, "mm_hits": 1.0},
        {"hits": 1056.0, "queries": 2112.0, "mm_hits": 2.0},
    ) == {"hits": 528.0, "mm_hits": 1.0, "queries": 1056.0}


def test_response_well_formed_rejects_nan_and_invalid_token_ids():
    valid = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
                "token_ids": [1, 2, 3],
            }
        ],
        "usage": {"completion_tokens": 3},
    }
    assert response_is_well_formed(valid)
    invalid_float = json.loads(json.dumps(valid))
    invalid_float["usage"]["score"] = float("nan")
    assert not response_is_well_formed(invalid_float)
    invalid_tokens = json.loads(json.dumps(valid))
    invalid_tokens["choices"][0]["token_ids"] = [1, -1]
    assert not response_is_well_formed(invalid_tokens)

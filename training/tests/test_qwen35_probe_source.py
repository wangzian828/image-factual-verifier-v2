from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe"
    / "vllm_qwen35_endpoint.py"
).read_text(encoding="utf-8")


def test_qwen35_probe_checks_hybrid_thinking_and_native_protocols() -> None:
    assert '"enable_thinking": thinking' in SOURCE
    assert "thinking_off: reasoning was unexpectedly returned" in SOURCE
    assert "thinking_on: reasoning was not separated or was empty" in SOURCE
    assert '"type": "image_url"' in SOURCE
    assert '"type": "json_schema"' in SOURCE
    assert '"tool_choice": "required"' in SOURCE
    assert '"role": "tool"' in SOURCE
    assert "--rounds" in SOURCE
    assert "expected max_model_len" in SOURCE
    assert "detail[:4000]" in SOURCE

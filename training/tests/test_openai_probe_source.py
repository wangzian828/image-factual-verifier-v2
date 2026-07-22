from __future__ import annotations

from pathlib import Path


def test_probe_covers_multimodal_schema_and_independent_tool_lifecycles() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "probe"
        / "openai_endpoint.py"
    ).read_text(encoding="utf-8")

    assert '"type": "image_url"' in source
    assert '"type": "json_schema"' in source
    assert '"tool_choice": "required"' in source
    assert '"role": "tool"' in source
    assert '"tool_choice": "none"' in source
    assert 'parser.add_argument("--max-tokens", type=int, default=2048)' in source
    assert '"max_tokens": args.max_tokens' in source
    assert "Each loop is a new lifecycle" in source
    assert '"lifecycle": "independent_tool_roundtrips"' in source

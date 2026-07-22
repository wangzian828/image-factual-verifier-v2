from __future__ import annotations

from pathlib import Path


def test_rl_probe_requires_tools_token_ids_logprobs_and_length_alignment() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "server"
        / "probe_rl_compatibility.py"
    ).read_text(encoding="utf-8")

    assert '"tool_choice": "required"' in source
    assert '"return_token_ids": True' in source
    assert '"logprobs": True' in source
    assert '"prompt_token_ids"' in source
    assert '"completion_token_ids"' in source
    assert '"completion_logprobs"' in source
    assert '"token_logprob_lengths_match"' in source
    assert "raise SystemExit(1)" in source

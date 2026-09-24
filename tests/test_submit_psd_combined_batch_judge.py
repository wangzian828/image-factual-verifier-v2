import base64
import json
from pathlib import Path

import pytest

from scripts.server import submit_psd_combined_batch_judge as batch


def test_request_keeps_image_prompt_model_settings_and_schema(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\ncontents")
    prepared = {
        "image": batch.stat_identity(image),
        "image_mime": "image/png",
        "prompt": "frozen judge prompt",
    }
    request = batch.request_for(prepared)
    parts = request["contents"][0]["parts"]
    assert base64.b64decode(parts[0]["inlineData"]["data"]) == image.read_bytes()
    assert parts[1]["text"] == "frozen judge prompt"
    assert request["generationConfig"]["maxOutputTokens"] == 32768
    assert request["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "LOW", "includeThoughts": True,
    }
    assert request["generationConfig"]["responseJsonSchema"] == batch.PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        batch.request_for(prepared)


def test_prepare_one_preserves_judge_prompt_and_verdict(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "test-manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    image = tmp_path / "original.jpg"
    image.write_bytes(b"image")
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"image_path": image.name}), encoding="utf-8")
    monkeypatch.setattr(batch, "build_agent_private_gold_candidate", lambda raw: {"frozen": raw["image_path"]})
    monkeypatch.setattr(batch, "agent_candidate_answer", lambda _: {"verdict": "real"})
    monkeypatch.setattr(batch, "_private_gold", lambda _: {"auditable": True, "expected_verdict": "real"})
    row = batch.prepare_one("case-1", trace, {"verdict": "real"}, {}, manifest.parent)
    expected = batch.PRIVATE_GOLD_JUDGE_PROMPT + "\n\nAUDIT INPUT:\n" + json.dumps({
        "private_gold": {"auditable": True, "expected_verdict": "real"},
        "candidate_material": {"mode": "agent_trace", "candidate_answer": {"verdict": "real"},
                               "supporting_trace_material": {"frozen": "original.jpg"}},
    }, ensure_ascii=False, indent=2)
    assert row["prompt"] == expected
    assert row["trace"] == batch.stat_identity(trace)
    assert row["image"] == batch.stat_identity(image)
    with pytest.raises(ValueError, match="verdict changed"):
        batch.prepare_one("case-1", trace, {"verdict": "fake"}, {}, manifest.parent)

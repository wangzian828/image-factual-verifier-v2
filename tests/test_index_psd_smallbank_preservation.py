import json

import pytest

from scripts.server.index_psd_smallbank_preservation import index_preservation


MODEL = "ifv-qwen3.5-9b-sft-3084"
SHA = "d" * 64
CHECKPOINT = "/frozen/sft3/model"


class Tokenizer:
    def decode(self, ids, **_kwargs):
        assert ids == [10]
        return '<think>look</think> [{"name": "text_search"}]<|im_end|>'


def _row():
    return {"case_id": "case-1", "episode_id": "episode-1",
            "verified_full_task": True, "strict_trace_audit_pass": True,
            "model_roles": {
                "frozen_self_teacher": {"provider": "qwen_local", "model": MODEL,
                                        "checkpoint_manifest_sha256": SHA,
                                        "round_start_checkpoint": CHECKPOINT},
                "trainable_student": {"provider": "qwen_local", "model": MODEL,
                                      "checkpoint_manifest_sha256": SHA,
                                      "initial_checkpoint": CHECKPOINT}},
            "preservation_steps": [
                {"step_id": "episode-1:react:1", "stage": "unified_react",
                 "student_prompt_ids": [1], "completion_ids": [10], "psd_media": None},
                {"step_id": "episode-1:judgment:2", "stage": "unified_judgment",
                 "student_prompt_ids": [1, 2], "completion_ids": [11], "psd_media": None}]}


def _run(tmp_path, row, *, steps=2):
    source = tmp_path / "assembled.jsonl"
    source.write_text(json.dumps(row) + "\n")
    return index_preservation(source=source, output_dir=tmp_path / "index",
        tokenizer=Tokenizer(), model=MODEL, checkpoint_sha=SHA,
        checkpoint=CHECKPOINT, expected_episodes=1, expected_steps=steps)


def test_builds_whole_episode_tool_metadata_without_payload(tmp_path):
    manifest = _run(tmp_path, _row())
    assert manifest["episodes"] == 1 and manifest["steps"] == 2
    assert manifest["native_tool_counts"] == {"text_search": 1}
    episode = json.loads((tmp_path / "index/episodes.jsonl").read_text())
    assert episode["step_count"] == 2 and episode["tool_sequence"] == ["text_search"]
    assert "completion_ids" not in episode and "student_prompt_ids" not in episode


def test_rejects_mixed_checkpoint_or_partial_episode(tmp_path):
    row = _row()
    row["model_roles"]["trainable_student"]["checkpoint_manifest_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="mixed"):
        _run(tmp_path, row)
    assert not (tmp_path / "index").exists()


def test_rejects_image_without_pixel_binding(tmp_path):
    row = _row()
    row["preservation_steps"][0]["student_prompt_ids"] = [248056, 1]
    with pytest.raises(ValueError, match="pixel binding"):
        _run(tmp_path, row)


def test_rejects_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="counts disagree"):
        _run(tmp_path, _row(), steps=3)

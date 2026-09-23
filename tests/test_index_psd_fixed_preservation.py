import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/server/index_psd_fixed_preservation.py"
SPEC = importlib.util.spec_from_file_location("index_psd_fixed_preservation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture(tmp_path):
    reviews = tmp_path / "reviews.json"
    candidates = tmp_path / "candidates.jsonl"
    manifest = tmp_path / "candidate-manifest.json"
    review_doc = {
        "status": "source_reviews_complete_with_unadopted",
        "selected_cases": 3, "completed_cases": 3, "case_errors": 0,
        "review_counts": {"pass": 2, "fail": 1},
        "cases": [
            {"case_id": "a", "reviews": [
                {"status": "fail", "episode_id": "a0", "path": "/review/a0"},
                {"status": "pass", "episode_id": "a1", "path": "/review/a1"}]},
            {"case_id": "b", "reviews": [
                {"status": "pass", "episode_id": "b0", "path": "/review/b0"}]},
            {"case_id": "c", "reviews": []},
        ],
    }
    reviews.write_text(json.dumps(review_doc))
    manifest.write_text(json.dumps({"status": "ready_for_privileged_localization",
                                    "counts": {"preservation_candidates": 2}}))
    def row(cid, episode):
        return {
            "case_id": cid, "episode_id": episode, "candidate_id": cid + "-candidate",
            "class": "base_pass_preserve", "candidate_status": "ready_for_teacher_collection",
            "verified_full_task": True, "strict_trace_audit_pass": True,
            "source": {"source_task_review": {"path": f"/review/{episode}"},
                       "source_trace_path": f"traces/{episode}.json"},
            "preservation_steps": [
                {"step_id": f"{episode}:react:1", "stage": "unified_react",
                 "action_type": "tool_call", "protocol_rejected": False,
                 "observed_policy_action": {"name": "search_web"},
                 "rollout_token_capture": {"status": "complete", "prompt_token_ids": [1, 2],
                                           "completion_token_ids": [3]}},
                {"step_id": f"{episode}:judgment:2", "stage": "judgment",
                 "action_type": "final", "protocol_rejected": False,
                 "observed_policy_action": {},
                 "rollout_token_capture": {"status": "complete", "prompt_token_ids": [1, 2, 3],
                                           "completion_token_ids": [4, 5]}},
            ],
        }
    rows = [row("a", "a1"), row("b", "b0")]
    candidates.write_text("".join(json.dumps(x) + "\n" for x in rows))
    return candidates, reviews, manifest, rows


def test_freezes_complete_first_successes_as_offsets_only(tmp_path):
    candidates, reviews, manifest, rows = _fixture(tmp_path)
    output = tmp_path / "fixed"
    result = MODULE.freeze_preservation(candidates=candidates, reviews=reviews,
                                        candidate_manifest=manifest, output=output)
    assert result["counts"]["preservation_cases"] == 2
    assert result["counts"]["preservation_steps"] == 4
    assert result["counts"]["completion_token_ids"] == 6
    assert result["tool_counts"] == {"search_web": 2}
    index = [json.loads(x) for x in (output / "episodes.jsonl").read_text().splitlines()]
    assert len(index) == 2
    assert all(len(x["step_ids"]) == 2 for x in index)
    with candidates.open("rb") as source:
        for entry, expected in zip(index, rows):
            source.seek(entry["source_offset"])
            assert json.loads(source.read(entry["source_length"])) == expected
    assert not (output / "targets.jsonl").exists()
    with pytest.raises(FileExistsError):
        MODULE.freeze_preservation(candidates=candidates, reviews=reviews,
                                   candidate_manifest=manifest, output=output)


@pytest.mark.parametrize("change", ["wrong_episode", "wrong_review", "missing_judgment",
                                    "unverified", "duplicate_case", "duplicate_candidate"])
def test_rejects_bad_preservation_without_final_manifest(tmp_path, change):
    candidates, reviews, manifest, rows = _fixture(tmp_path)
    if change == "wrong_episode":
        rows[0]["episode_id"] = "a0"
    elif change == "wrong_review":
        rows[0]["source"]["source_task_review"]["path"] = "/review/a0"
    elif change == "missing_judgment":
        rows[0]["preservation_steps"][1]["step_id"] = "a1:react:2"
    elif change == "unverified":
        rows[0]["strict_trace_audit_pass"] = False
    else:
        if change == "duplicate_case":
            rows[1] = rows[0]
        else:
            rows[1]["candidate_id"] = rows[0]["candidate_id"]
    candidates.write_text("".join(json.dumps(x) + "\n" for x in rows))
    output = tmp_path / "fixed"
    with pytest.raises(ValueError):
        MODULE.freeze_preservation(candidates=candidates, reviews=reviews,
                                   candidate_manifest=manifest, output=output)
    assert not (output / "manifest.json").exists()

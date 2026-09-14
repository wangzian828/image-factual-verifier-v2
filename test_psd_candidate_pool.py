import pytest

from scripts.prepare_psd_candidate_pool import alias_index, build_selection, group_cases, resolve, sample_groups


def fixture_rows():
    official = [{"unified_case_id": f"canonical-{i}", "record_id": f"short-{i}",
        "split": "train", "unified_image_path": f"images/{i}.jpg",
        "factual_status": "supported" if i % 2 else "refuted",
        "construction_subroute": "real_event" if i % 2 else "mutation"} for i in range(14)]
    gold = [{"case_id": r["unified_case_id"], "factual_status": r["factual_status"]} for r in official]
    splits = [{"case_id": r["record_id"], "split": "train", "image_sha256": str(i),
               "split_group_id": f"group-{i}"} for i, r in enumerate(official)]
    sft = [{"case_id": f"short-{i}", "split": "train", "tool_call_count": 4 + i} for i in range(4)]
    actions = [{"case_id": "short-4"}]
    return official, gold, splits, sft, actions


def test_aliases_reconcile_and_outcomes_are_not_invented():
    data = fixture_rows()
    result = build_selection(*data, [], dev_size=2, sft_size=2, seed="frozen")
    assert len(result["inventory"]) == 14
    assert len(result["hard_train"]) + len(result["development"]) == 9
    assert len(result["action_only_reserve"]) == 1
    assert all(r["prior_sft_exposure"] for r in result["sft_revisit"])
    assert all(r["teacher_failure_reason"] is None for r in result["hard_train"])
    reordered = [list(reversed(rows)) for rows in data]
    repeated = build_selection(*reordered, [], dev_size=2, sft_size=2, seed="frozen")
    for name in ("sft_revisit", "development"):
        assert result[name] == repeated[name]


def test_whole_related_groups_and_old_validation_stay_excluded():
    official, gold, splits, sft, actions = fixture_rows()
    official[10]["event_identity"] = official[11]["event_identity"] = "the same public event"
    splits[0]["split"] = "validation"  # Even if later SFT trained on it.
    splits[5]["image_sha256"] = splits[1]["image_sha256"]  # Hard case related to SFT.
    tests = [{"case_id": "external-test", "event_identity": "the same public event"}]
    result = build_selection(official, gold, splits, sft, actions, tests, dev_size=2, sft_size=2, seed="frozen")
    excluded = {r["case_id"] for r in result["excluded"]}
    assert {"canonical-0", "canonical-10", "canonical-11"} <= excluded
    assert "canonical-5" not in {r["case_id"] for r in result["development"]}
    assert not excluded & {r["case_id"] for r in result["sft_revisit"]}


def test_ambiguous_identity_rejected():
    official = [{"unified_case_id": "a", "record_id": "shared"},
                {"unified_case_id": "b", "record_id": "shared"}]
    with pytest.raises(ValueError, match="non-unique"):
        resolve("shared", alias_index(official))


def test_sampling_keeps_duplicates_and_transitive_groups_together():
    records = [{"case_id": str(i), "original_split_group_id": str(i), "label": "real"} for i in range(5)]
    records[0]["original_image_sha256"] = records[1]["original_image_sha256"] = "same-pixels"
    records[1]["event_keys"] = records[2]["event_keys"] = [("event", "shared-event")]
    group_cases(records)
    assert len({r["selection_group_id"] for r in records[:3]}) == 1
    sample = sample_groups(records, 3, "stable", ("label",))
    chosen = {r["case_id"] for r in sample}
    assert not chosen & {"0", "1", "2"} or {"0", "1", "2"} <= chosen


def test_gold_label_mismatch_is_quarantined():
    official, gold, splits, sft, actions = fixture_rows()
    gold[7]["factual_status"] = "refuted"
    result = build_selection(official, gold, splits, sft, actions, [], dev_size=2, sft_size=2, seed="frozen")
    rejected = next(r for r in result["excluded"] if r["case_id"] == "canonical-7")
    assert "private_label_missing_or_mismatched" in rejected["exclusion_reasons"]

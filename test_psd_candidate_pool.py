import pytest

from scripts.prepare_psd_candidate_pool import OrderedRangeReader, alias_index, build_selection, group_cases, resolve, sample_groups


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


def test_parallel_byte_stream_preserves_order_boundaries_and_final_short_range():
    source = bytes(range(251)) * 7
    reader = OrderedRangeReader(len(source), lambda a, b: source[a:b], workers=3, chunk_size=103)
    try:
        chunks = [reader.read(n) for n in (1, 102, 105, 0, 203, 5000)]
        assert b"".join(chunks) == source
        assert reader.read(100) == b""
    finally:
        reader.close()


def test_parallel_byte_stream_rejects_truncated_range():
    reader = OrderedRangeReader(100, lambda a, b: b"short", workers=1, chunk_size=20)
    try:
        with pytest.raises(ValueError, match="length mismatch"):
            reader.read(10)
    finally:
        reader.close()


@pytest.mark.parametrize("image_format", ["JPEG", "BMP", "TIFF"])
def test_stream_resolves_selected_hardlink_without_storing_full_archive(tmp_path, monkeypatch, image_format):
    import hashlib
    import io
    import tarfile
    from types import SimpleNamespace
    from PIL import Image
    from scripts import prepare_psd_candidate_pool as module

    pixels = io.BytesIO()
    Image.new("RGB", (20, 12), "red").save(pixels, format=image_format)
    data = pixels.getvalue()
    compressed = io.BytesIO()
    with tarfile.open(fileobj=compressed, mode="w:gz") as archive:
        regular = tarfile.TarInfo("images/a.jpg")
        regular.size = len(data)
        archive.addfile(regular, io.BytesIO(data))
        linked = tarfile.TarInfo("images/b.jpg")
        linked.type, linked.linkname = tarfile.LNKTYPE, "images/a.jpg"
        archive.addfile(linked)
    source = compressed.getvalue()
    monkeypatch.setattr(module, "SIZE", len(source))
    monkeypatch.setattr(module, "SHA256", hashlib.sha256(source).hexdigest())
    monkeypatch.setattr(module, "official_range", lambda a, b: source[a:b])
    result = module.stream_images(SimpleNamespace(output_dir=tmp_path),
        [{"official_image_member": "images/b.jpg"}],
        [{"unified_image_path": f"images/{name}.jpg"} for name in ("a", "b")])
    assert result["images/a.jpg"]["sha256"] == result["images/b.jpg"]["sha256"]
    assert result["images/b.jpg"]["normalized_image_sha256"]
    assert module.sha256_file(module.Path(result["images/b.jpg"]["path"])) == hashlib.sha256(data).hexdigest()
    assert len(list((tmp_path / "media").iterdir())) == 1
    assert module.load_json(tmp_path / "archive-verification.json")["hardlink_members"] == 1


@pytest.mark.parametrize("linkname", ["../../outside.jpg", "/outside.jpg", "images/forward.jpg"])
def test_stream_rejects_unsafe_or_forward_hardlinks(tmp_path, monkeypatch, linkname):
    import hashlib
    import io
    import tarfile
    from types import SimpleNamespace
    from scripts import prepare_psd_candidate_pool as module

    compressed = io.BytesIO()
    with tarfile.open(fileobj=compressed, mode="w:gz") as archive:
        linked = tarfile.TarInfo("images/b.jpg")
        linked.type, linked.linkname = tarfile.LNKTYPE, linkname
        archive.addfile(linked)
    source = compressed.getvalue()
    monkeypatch.setattr(module, "SIZE", len(source))
    monkeypatch.setattr(module, "SHA256", hashlib.sha256(source).hexdigest())
    monkeypatch.setattr(module, "official_range", lambda a, b: source[a:b])
    with pytest.raises(ValueError, match="unsafe or forward"):
        module.stream_images(SimpleNamespace(output_dir=tmp_path), [], [{"unified_image_path": "images/b.jpg"}])


def test_conflicting_image_labels_quarantine_both_groups_even_when_bytes_differ():
    from scripts.prepare_psd_candidate_pool import image_conflict_groups
    rows = [{"official_image_member": str(i), "label": label, "selection_group_id": str(i)}
            for i, label in enumerate(("real", "fake", "real"))]
    images = {str(i): {"sha256": str(i), "normalized_image_sha256": "same" if i < 2 else "other"}
              for i in range(3)}
    assert image_conflict_groups({"inventory": rows}, images) == {"0", "1"}


def test_final_public_releases_verify_and_development_guard_is_enforced(tmp_path):
    from PIL import Image
    from scripts import prepare_psd_candidate_pool as selector
    from scripts.verify_psd_candidate_pool import verify
    from src.orchestrator.source_access import benchmark_source_access_policy
    from src.tools.vision_utils import controlled_image_to_data_url

    images, gold, selections = {}, {}, {}
    for name, color in zip(("hard_train", "sft_revisit", "development"), ("red", "green", "blue")):
        path = tmp_path / (name + ".jpg")
        Image.new("RGB", (12, 8), color).save(path)
        sha = selector.sha256_file(path)
        _, metadata = controlled_image_to_data_url(str(path), max_long_edge=1024, jpeg_quality=95)
        images[name] = {"archive_member": name, "sha256": sha, "normalized_image_sha256": metadata["sha256"], "path": str(path)}
        selections[name] = [{"case_id": name, "official_image_member": name, "selection_group_id": name,
                             "label": "real", "prior_sft_exposure": name == "sft_revisit"}]
        gold[name] = {"case_id": name, "factual_status": "supported", "private_evidence": "not public"}
    selections["train"] = selections["hard_train"] + selections["sft_revisit"]
    releases = {}
    for name, rows in selections.items():
        releases[name] = selector.release(tmp_path, name, rows, images, gold, benchmark_source_access_policy([]))
        selector.write_jsonl(tmp_path / "selection" / (name + "-final.jsonl"), rows)
    selector.write_jsonl(tmp_path / "official-image-inventory.jsonl", list(images.values()))
    selector.write_json(tmp_path / "archive-verification.json", {"passed": True, "official_images_hashed": 3})
    selector.write_json(tmp_path / "selection-report.json", {"status": "ready_for_fresh_policy_rollouts",
                        "source_identity": {"files": {}}, "releases": releases})
    result = verify(tmp_path, tmp_path, wire_images=True)
    assert result["public_images_independently_hashed"] == 3
    assert result["runtime_wire_images_checked"] == 3
    assert result["pools"]["train"] == 2
    dev_manifest = tmp_path / "development/runtime-release/manifest.json"
    manifest = selector.load_json(dev_manifest)
    manifest["training_prohibited"] = False
    selector.write_json(dev_manifest, manifest)
    with pytest.raises(ValueError, match="development training guard"):
        verify(tmp_path, tmp_path)

"""Select a provenance-bound PSD case pool; never generate teacher targets.

Reconcile official case aliases with actual SFT membership. Split whole image/
event groups, quarantine prior validation/test groups, sample representative
SFT cases, and optionally stream the pinned official archive for original
pixels. Gold and selection metadata never enter public model inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT = Path(os.environ.get("IFV_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from scripts.prepare_psd_training_metadata import ARCHIVE, REPO, SHA256, SIZE, HashingReader

VERSION = "ifv-psd-candidate-pool-v1"
ALIAS_KEYS = ("case_id", "unified_case_id", "record_id", "candidate_id", "assignment_id")


def canonical(row):
    return str(row.get("unified_case_id") or row.get("case_id") or "")


def aliases(row):
    return {str(row[k]) for k in ALIAS_KEYS if row.get(k)} | set(row.get("case_id_aliases") or [])


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def alias_index(rows):
    result = defaultdict(set)
    for row in rows:
        if not canonical(row):
            raise ValueError("official case lacks canonical ID")
        for alias in aliases(row):
            result[alias].add(canonical(row))
    return result


def resolve(alias, index):
    matches = index.get(alias, set())
    if len(matches) != 1:
        raise ValueError(f"non-unique official mapping for {alias}: {len(matches)}")
    return next(iter(matches))


def event_keys(row):
    return {(key, str(row[key]).strip().casefold()) for key in ("event_key", "event_identity")
            if row.get(key) and len(str(row[key]).strip()) > 5}


def group_cases(records):
    """Transitive groups: old split, original hash, and explicit event identity."""
    parent = {r["case_id"]: r["case_id"] for r in records}
    seen = {}

    def root(case):
        while parent[case] != case:
            parent[case] = parent[parent[case]]
            case = parent[case]
        return case

    for row in records:
        keys = [("split_group", row.get("original_split_group_id")),
                ("image", row.get("original_image_sha256")), *row.get("event_keys", [])]
        for key in keys:
            if not key[1]:
                continue
            if key in seen:
                a, b = root(row["case_id"]), root(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = row["case_id"]
    for row in records:
        row["selection_group_id"] = digest(root(row["case_id"]))[:24]
    return records


def sample_groups(records, target, seed, fields):
    """Deterministic, proportional stratification with whole-group selection.

    A group may cross the target by its size; this is reported, never split.
    """
    if not 0 <= target <= len(records):
        raise ValueError("requested sample exceeds eligible cases")
    groups = defaultdict(list)
    for row in records:
        groups[row["selection_group_id"]].append(row)
    total = Counter(tuple(r.get(k) for k in fields) for r in records)
    quota = {s: target * n / len(records) for s, n in total.items()} if records else {}
    selected, counts = [], Counter()
    while len(selected) < target:
        def priority(item):
            group, members = item
            gain = sum(max(0, quota[tuple(r.get(k) for k in fields)] - counts[tuple(r.get(k) for k in fields)])
                       / max(1, total[tuple(r.get(k) for k in fields)]) for r in members) / len(members)
            return (-gain, digest(seed + ":" + group))
        group, members = min(groups.items(), key=priority)
        del groups[group]
        selected.extend(sorted(members, key=lambda r: r["case_id"]))
        counts.update(tuple(r.get(k) for k in fields) for r in members)
    return sorted(selected, key=lambda r: r["case_id"])


def describe(rows):
    return {"cases": len(rows), "groups": len({r["selection_group_id"] for r in rows}),
            "labels": dict(Counter(r["label"] for r in rows)),
            "routes": dict(Counter(r["route"] for r in rows)),
            "tool_count_bins": dict(Counter(r["teacher_tool_count_bin"] for r in rows))}


def build_selection(official, gold, splits, sft_index, action_only, tests, *, dev_size, sft_size, seed):
    index = alias_index(official)
    by_id = {canonical(r): r for r in official}
    if len(by_id) != len(official):
        raise ValueError("duplicate official ID")
    split = {resolve(r["case_id"], index): r for r in splits}
    if len(split) != len(splits) or set(split) != set(by_id):
        raise ValueError("frozen split must map one-to-one onto official train metadata")
    gold_map = {canonical(r): r for r in gold}
    if len(gold_map) != len(gold) or set(gold_map) != set(by_id):
        raise ValueError("private gold must cover each official case exactly once")
    sft = {resolve(r["case_id"], index): r for r in sft_index}
    if len(sft) != len(sft_index):
        raise ValueError("duplicate SFT case mapping")
    action = {resolve(r["case_id"], index) for r in action_only}
    if action & sft.keys():
        raise ValueError("reasoning and action-only source memberships overlap")
    test_aliases = set().union(*(aliases(r) for r in tests))
    test_events = set().union(*(event_keys(r) for r in tests))
    test_hashes = {r["image_sha256"] for r in tests if r.get("image_sha256")}
    records = []
    for case, row in by_id.items():
        sp, teacher = split[case], sft.get(case, {})
        reasons = []
        if row.get("split") != "train" or row.get("in_registered_test_set"):
            reasons.append("not_official_train")
        if aliases(row) & test_aliases:
            reasons.append("test_identity_overlap")
        if event_keys(row) & test_events:
            reasons.append("test_event_overlap")
        if sp.get("image_sha256") in test_hashes:
            reasons.append("test_image_overlap")
        if sp.get("split") != "train" or (teacher and teacher.get("split") != "train"):
            reasons.append("existing_validation")
        status = row.get("factual_status")
        if status not in ("supported", "refuted") or gold_map[case].get("factual_status") != status:
            reasons.append("private_label_missing_or_mismatched")
        count = teacher.get("tool_call_count")
        records.append({"case_id": case, "record_id": row.get("record_id"),
            "label": "real" if status == "supported" else "fake", "route": row.get("construction_subroute") or "unknown",
            "capability_cell": row.get("target_capability_cell") or "unknown",
            "official_image_member": row["unified_image_path"],
            "original_image_sha256": sp.get("image_sha256"),
            "original_split_group_id": sp.get("split_group_id"), "event_keys": sorted(event_keys(row)),
            "prior_sft_exposure": teacher.get("split") == "train",
            "teacher_delivery": "reasoning" if teacher else "action_only" if case in action else "not_in_success_delivery",
            "teacher_failure_reason": None,
            "teacher_tool_count": count,
            "teacher_tool_count_bin": "unknown" if count is None else "0-5" if count <= 5 else "6-12" if count <= 12 else "13+",
            "exclusion_reasons": reasons})
    group_cases(records)
    prohibited_groups = {r["selection_group_id"] for r in records if r["exclusion_reasons"]}
    for row in records:
        if row["selection_group_id"] in prohibited_groups and not row["exclusion_reasons"]:
            row["exclusion_reasons"].append("related_to_excluded_group")
    eligible = [r for r in records if not r["exclusion_reasons"]]
    hard = [r for r in eligible if r["teacher_delivery"] == "not_in_success_delivery"]
    seen_groups = {r["selection_group_id"] for r in records if r["teacher_delivery"] != "not_in_success_delivery"}
    dev_candidates = [r for r in hard if r["selection_group_id"] not in seen_groups]
    dev = sample_groups(dev_candidates, dev_size, seed + ":dev", ("label", "route"))
    dev_groups = {r["selection_group_id"] for r in dev}
    hard_train = [r for r in hard if r["selection_group_id"] not in dev_groups]
    sft_candidates = [r for r in eligible if r["prior_sft_exposure"] and r["selection_group_id"] not in dev_groups]
    sft_sample = sample_groups(sft_candidates, sft_size, seed + ":sft", ("label", "route", "teacher_tool_count_bin"))
    return {"hard_train": hard_train, "sft_revisit": sft_sample, "development": dev,
            "action_only_reserve": [r for r in eligible if r["teacher_delivery"] == "action_only"],
            "excluded": [r for r in records if r["exclusion_reasons"]],
            "inventory": records}


def scoped(path, root):
    path = path.resolve()
    path.relative_to(root.resolve())
    return path


def stream_images(args, selected, official):
    """One pass, no compressed archive copy; retain selected original pixels."""
    from modelscope_hub.api import HubApi
    from PIL import Image
    output = args.output_dir
    media = output / "media"
    media.mkdir(exist_ok=True)
    desired = {r["official_image_member"] for r in selected}
    known = {r["unified_image_path"] for r in official}
    inventory, seen = {}, set()
    response = HubApi().downloader._client.download_stream(repo_id=REPO, repo_type="dataset",
        file_path=ARCHIVE, revision="master", headers={"Accept-Encoding": "identity"})
    with response:
        response.raise_for_status()
        reader = HashingReader(response.raw)
        with tarfile.open(fileobj=reader, mode="r|gz", bufsize=1024**2) as archive:
            for member in archive:
                rel = PurePosixPath(member.name)
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("unsafe archive member")
                name = rel.as_posix()
                if name not in known:
                    continue
                if name in seen or not member.isfile() or member.size > 128 * 1024**2:
                    raise ValueError("duplicate/nonregular/oversized official image")
                seen.add(name)
                with archive.extractfile(member) as handle:
                    data = handle.read()
                sha = hashlib.sha256(data).hexdigest()
                record = {"archive_member": name, "sha256": sha, "bytes": len(data)}
                if name in desired:
                    with Image.open(io.BytesIO(data)) as im:
                        record.update(width=im.width, height=im.height, format=im.format)
                        extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}.get(im.format)
                        if not extension:
                            raise ValueError("unsupported original image format")
                        im.verify()
                    path = media / (sha + extension)
                    if path.exists():
                        if sha256_file(path) != sha:
                            raise ValueError("existing selected image bytes changed")
                    else:
                        with path.open("xb") as handle:
                            handle.write(data)
                    record["path"] = str(path)
                inventory[name] = record
        while reader.read(1024**2):
            pass
        if reader.count != SIZE or reader.digest.hexdigest() != SHA256:
            raise ValueError("official archive did not match pinned SHA256/size")
    if set(inventory) != known or not desired.issubset(inventory):
        raise ValueError("official archive image membership mismatch")
    write_json(output / "archive-verification.json", {"passed": True, "sha256": SHA256,
        "bytes": SIZE, "compressed_archive_stored": False, "official_images_hashed": len(inventory)})
    write_jsonl(output / "official-image-inventory.jsonl", sorted(inventory.values(), key=lambda r: r["archive_member"]))
    return inventory


def release(output, name, records, image_inventory, gold_map, policy):
    from src.eval.release_adapter import (RELEASE_SCHEMA_VERSION, RUNTIME_CONTRACT_VERSION,
        RUNTIME_CASE_KEYS, INPUT_MODE, DATA_PIPELINE_DECISION_POLICY_VERSION)
    from src.eval.public_release import load_public_release
    root = output / name
    runtime = root / "runtime-release"
    public = []
    for row in records:
        image = image_inventory[row["official_image_member"]]
        source = Path(image["path"])
        destination = runtime / "runtime_input/assets" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(source, destination)
        public.append({"case_id": row["case_id"], "image_path": "assets/" + source.name,
                       "image_sha256": image["sha256"]})
    benchmark = runtime / "runtime_input/cases.jsonl"
    write_jsonl(benchmark, public)
    write_jsonl(root / "evaluator_private/private_gold.jsonl", [dict(gold_map[r["case_id"]], case_id=r["case_id"]) for r in records])
    write_jsonl(root / "evaluator_private/case_split.jsonl", [{"case_id": r["case_id"],
        "split": "validation" if name == "development" else "train", "split_group_id": r["selection_group_id"],
        "image_sha256": image_inventory[r["official_image_member"]]["sha256"]} for r in records])
    policy_path = runtime / "evaluator_private/source_access_policy.json"
    write_json(policy_path, policy.to_dict())
    write_json(runtime / "manifest.json", {"schema_version": RELEASE_SCHEMA_VERSION,
        "release_id": VERSION + "-" + name, "release_stage": "psd_candidate_selection",
        "training_prohibited": name == "development", "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
        "input_mode": INPUT_MODE, "decision_policy_version": DATA_PIPELINE_DECISION_POLICY_VERSION,
        "runtime_contract": {"allowed_keys": sorted(RUNTIME_CASE_KEYS), "private_keys_absent": True},
        "artifacts": {"agent_input": "runtime_input/cases.jsonl"},
        "source_access_policy": {"active": True, "path": "evaluator_private/source_access_policy.json",
                                 "sha256": sha256_file(policy_path)}})
    load_public_release(benchmark)
    return {"cases": len(public), "benchmark": str(benchmark), "sha256": sha256_file(benchmark)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("storage-root", "metadata-root", "sft-root", "case-split", "test-manifest", "test-runtime", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dev-size", type=int, default=400)
    parser.add_argument("--sft-size", type=int, default=1000)
    parser.add_argument("--seed", default="psd-pool-20260914-v1")
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, scoped(value, args.storage_root))
    if args.output_dir == args.storage_root:
        raise ValueError("output must not be storage root")
    meta = load_json(args.metadata_root / "verified-metadata.json")
    if meta["archive_sha256"] != SHA256:
        raise ValueError("wrong official train archive")
    for name, sha in meta["files"].items():
        if sha256_file(args.metadata_root / name) != sha:
            raise ValueError("official metadata hash changed")
    paths = {"official": args.metadata_root / "train-manifest.jsonl",
             "gold": args.metadata_root / "evaluator_private/private-gold-v1/train-private-gold.jsonl",
             "sft_index": args.sft_root / "canonical-dataset/index.jsonl",
             "action_only": args.sft_root / "canonical-dataset/action_only.jsonl",
             "splits": args.case_split, "test_manifest": args.test_manifest, "test_runtime": args.test_runtime}
    inputs = {k: load_jsonl(p) for k, p in paths.items()}
    identity = {"version": VERSION, "files": {str(p): sha256_file(p) for p in paths.values()},
                "seed": args.seed, "dev_size": args.dev_size, "sft_size": args.sft_size}
    prior = args.output_dir / "identity.json"
    if prior.exists():
        if load_json(prior) != identity:
            raise ValueError("refusing to reuse output with changed selection inputs")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(prior, identity)
    selection = build_selection(inputs["official"], inputs["gold"], inputs["splits"], inputs["sft_index"],
        inputs["action_only"], inputs["test_manifest"] + inputs["test_runtime"],
        dev_size=args.dev_size, sft_size=args.sft_size, seed=args.seed)
    for name, rows in selection.items():
        write_jsonl(args.output_dir / "selection" / (name + ".jsonl"), rows)
    report = {"schema_version": VERSION, "status": "selected_images_pending", "official_cases": len(inputs["official"]),
        "pools": {k: describe(v) for k, v in selection.items()}, "source_identity": identity,
        "teacher_failure_evidence": "absence_from_three_success_delivery_bundles; individual attempt outcomes unavailable",
        "fresh_current_policy_rollout_required": True, "psd_targets_created": False,
        "quarantine_reasons": dict(Counter(x for r in selection["excluded"] for x in r["exclusion_reasons"]))}
    write_json(args.output_dir / "selection-report.json", report)
    print(json.dumps({"stage": "selected", "pools": report["pools"]}, ensure_ascii=False), flush=True)
    if not args.materialize:
        return
    selected = selection["hard_train"] + selection["sft_revisit"] + selection["development"]
    archive_record = args.output_dir / "archive-verification.json"
    if archive_record.exists():
        if load_json(archive_record).get("sha256") != SHA256:
            raise ValueError("wrong saved archive binding")
        images = {r["archive_member"]: r for r in load_jsonl(args.output_dir / "official-image-inventory.jsonl")}
    else:
        images = stream_images(args, selected, inputs["official"])
    test_hashes = {r["image_sha256"] for r in inputs["test_runtime"]}
    dev_hashes = {images[r["official_image_member"]]["sha256"] for r in selection["development"]}
    train_hashes = {images[r["official_image_member"]]["sha256"] for r in selection["hard_train"] + selection["sft_revisit"]}
    historical_hashes = {images[r["official_image_member"]]["sha256"] for r in selection["inventory"]
                         if r["teacher_delivery"] != "not_in_success_delivery" or r["exclusion_reasons"]}
    bad_hashes = test_hashes | (dev_hashes & (train_hashes | historical_hashes))
    bad_groups = {r["selection_group_id"] for r in selected if images[r["official_image_member"]]["sha256"] in bad_hashes}
    report["raw_image_quarantine"] = [r["case_id"] for r in selected if r["selection_group_id"] in bad_groups]
    final = {name: [r for r in selection[name] if r["selection_group_id"] not in bad_groups]
             for name in ("hard_train", "sft_revisit", "development")}
    final["train"] = sorted(final["hard_train"] + final["sft_revisit"], key=lambda r: r["case_id"])
    # Verify each selected byte artifact before any model-visible release.
    for row in selected:
        image = images[row["official_image_member"]]
        if sha256_file(scoped(Path(image["path"]), args.storage_root)) != image["sha256"]:
            raise ValueError("selected original image changed")
    from src.orchestrator.source_access import benchmark_source_access_policy
    policy = benchmark_source_access_policy((str(r.get("source_url") or "") for r in inputs["official"]), policy_id=VERSION)
    gold_map = {canonical(r): r for r in inputs["gold"]}
    report["releases"] = {name: release(args.output_dir, name, rows, images, gold_map, policy) for name, rows in final.items()}
    for name, rows in final.items():
        write_jsonl(args.output_dir / "selection" / (name + "-final.jsonl"), rows)
    report.update(status="ready_for_fresh_policy_rollouts", raw_image_test_overlap=0,
        raw_image_train_development_overlap=0, development_historical_sft_image_overlap=0,
        stored_original_image_bytes=sum(r["bytes"] for r in {r["sha256"]: r for r in images.values() if r.get("path")}.values()),
        final_pools={name: describe(rows) for name, rows in final.items()})
    write_json(args.output_dir / "selection-report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

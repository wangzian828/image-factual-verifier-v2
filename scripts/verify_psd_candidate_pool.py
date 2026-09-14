"""Independently verify finalized PSD public inputs and split isolation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("IFV_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_json, load_jsonl, sha256_file, write_json
from src.eval.public_release import load_public_release


def verify(root, storage_root, *, wire_images=False):
    root.resolve().relative_to(storage_root.resolve())
    report = load_json(root / "selection-report.json")
    if report["status"] != "ready_for_fresh_policy_rollouts":
        raise ValueError("candidate pool is not finalized")
    for path, expected in report["source_identity"]["files"].items():
        Path(path).resolve().relative_to(storage_root.resolve())
        if sha256_file(Path(path)) != expected:
            raise ValueError("selection source changed")
    archive = load_json(root / "archive-verification.json")
    if archive.get("passed") is not True:
        raise ValueError("official archive has not been verified")
    inventory = load_jsonl(root / "official-image-inventory.jsonl")
    images = {r["archive_member"]: r for r in inventory}
    if len(images) != len(inventory) or len(images) != archive["official_images_hashed"]:
        raise ValueError("official image inventory count mismatch")
    selections = {name: load_jsonl(root / "selection" / (name + "-final.jsonl"))
                  for name in ("hard_train", "sft_revisit", "development", "train")}
    final_ids, groups, raw, normalized, checked = {}, {}, {}, {}, set()
    wire_checks = {}
    for name, rows in selections.items():
        benchmark = root / name / "runtime-release/runtime_input/cases.jsonl"
        load_public_release(benchmark)
        public = load_jsonl(benchmark)
        expected_ids = [r["case_id"] for r in rows]
        if [r["case_id"] for r in public] != expected_ids or len(set(expected_ids)) != len(expected_ids):
            raise ValueError("public input membership is inconsistent")
        if sha256_file(benchmark) != report["releases"][name]["sha256"]:
            raise ValueError("public input checksum mismatch")
        private = load_jsonl(root / name / "evaluator_private/private_gold.jsonl")
        if [r["case_id"] for r in private] != expected_ids:
            raise ValueError("private gold membership is inconsistent")
        splits = load_jsonl(root / name / "evaluator_private/case_split.jsonl")
        if [r["case_id"] for r in splits] != expected_ids:
            raise ValueError("split membership is inconsistent")
        for source, item, split, gold in zip(rows, public, splits, private):
            if set(item) != {"case_id", "image_path", "image_sha256"}:
                raise ValueError("private data entered public input")
            path = (benchmark.parent / item["image_path"]).resolve()
            path.relative_to(storage_root.resolve())
            if item["image_sha256"] != images[source["official_image_member"]]["sha256"]:
                raise ValueError("public image mismatches pinned original")
            if split["split"] != ("validation" if name == "development" else "train"):
                raise ValueError("wrong split role")
            if source["label"] != ("real" if gold["factual_status"] == "supported" else "fake"):
                raise ValueError("selected label mismatches private gold")
            stat = path.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode not in checked:
                if sha256_file(path) != item["image_sha256"]:
                    raise ValueError("public image bytes changed")
                checked.add(inode)
                wire_checks[str(path)] = images[source["official_image_member"]]
        final_ids[name] = set(expected_ids)
        groups[name] = {r["selection_group_id"] for r in rows}
        raw[name] = {images[r["official_image_member"]]["sha256"] for r in rows}
        normalized[name] = {images[r["official_image_member"]]["normalized_image_sha256"] for r in rows}
        manifest = load_json(benchmark.parent.parent / "manifest.json")
        if manifest["training_prohibited"] != (name == "development"):
            raise ValueError("development training guard missing")
    if final_ids["hard_train"] & final_ids["sft_revisit"]:
        raise ValueError("hard and SFT revisit case membership overlap")
    if final_ids["train"] != final_ids["hard_train"] | final_ids["sft_revisit"]:
        raise ValueError("combined train membership mismatch")
    for values in (final_ids, groups, raw, normalized):
        if values["train"] & values["development"]:
            raise ValueError("PSD train/development overlap")
    if not all(r["prior_sft_exposure"] for r in selections["sft_revisit"]):
        raise ValueError("revisit pool includes unseen cases")
    if wire_images:
        from src.tools.vision_utils import controlled_image_to_data_url
        def check_wire(pair):
            path, expected = pair
            data, metadata = controlled_image_to_data_url(path, max_long_edge=1024, jpeg_quality=95)
            if (not data.startswith("data:image/jpeg;base64,")
                    or metadata["source_sha256"] != expected["sha256"]
                    or metadata["sha256"] != expected["normalized_image_sha256"]
                    or max(metadata["sent_size"]) > 1024):
                raise ValueError("runtime image wire conversion mismatches verified fingerprint")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(check_wire, wire_checks.items()))
    result = {"passed": True, "selection_report_sha256": sha256_file(root / "selection-report.json"),
        "official_image_inventory_sha256": sha256_file(root / "official-image-inventory.jsonl"),
        "public_images_independently_hashed": len(checked),
        "runtime_wire_images_checked": len(wire_checks) if wire_images else 0,
        "pools": {name: len(ids) for name, ids in final_ids.items()},
        "public_keys_checked": True, "private_gold_membership_checked": True,
        "train_development_case_group_raw_and_normalized_image_disjoint": True,
        "training_started": False, "fresh_policy_rollouts_still_required": True}
    write_json(root / "release-verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", required=True, type=Path)
    parser.add_argument("--storage-root", required=True, type=Path)
    parser.add_argument("--wire-images", action="store_true")
    args = parser.parse_args()
    print(verify(args.pool_root, args.storage_root, wire_images=args.wire_images), flush=True)


if __name__ == "__main__":
    main()

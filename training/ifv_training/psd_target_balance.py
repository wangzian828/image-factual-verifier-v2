"""Select a smaller, case-diverse PSD target bank without changing target payloads.

Run this before frozen-teacher top-K scoring to avoid scoring discarded targets.
It also accepts already-scored targets for a derived training-only ablation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def _identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path.resolve()), "device": stat.st_dev, "inode": stat.st_ino,
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if not 0 < count <= length:
        raise ValueError("invalid preservation selection count")
    return [((2 * index + 1) * length) // (2 * count) for index in range(count)]


def balance_psd_targets(
    *, source: Path, output_dir: Path, preserve_target_cap: int | None = None,
    seed: str = "psd-target-balance-v1",
) -> dict:
    """Keep every repair and sample preservation steps across successful cases.

    Only small row metadata and byte offsets are retained in memory. The source
    is read once; selected original JSON lines are then copied by seek. There
    are no model/provider calls, payload rewrites, or large-file hashes.
    """
    if not seed:
        raise ValueError("selection seed must be nonempty")
    source = source.resolve()
    output_dir = output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    repair_offsets: list[int] = []
    preserve_by_case: dict[str, list[int]] = defaultdict(list)
    seen_ids: set[str] = set()
    with source.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            row = json.loads(line)
            target_id = row.get("target_id")
            kind = row.get("kind")
            case_id = row.get("case_id")
            if not isinstance(target_id, str) or not target_id or target_id in seen_ids:
                raise ValueError("PSD target IDs must be present and unique")
            if kind not in ("repair", "preserve"):
                raise ValueError(f"unknown PSD target kind: {kind!r}")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError("PSD targets require case_id")
            seen_ids.add(target_id)
            if kind == "repair":
                repair_offsets.append(offset)
            else:
                preserve_by_case[case_id].append(offset)
    repair_count = len(repair_offsets)
    available_preserve = sum(map(len, preserve_by_case.values()))
    if not repair_count or not available_preserve:
        raise ValueError("balanced PSD training requires both source kinds")
    cap = repair_count if preserve_target_cap is None else preserve_target_cap
    if isinstance(cap, bool) or not isinstance(cap, int) or not 0 < cap <= available_preserve:
        raise ValueError("preservation cap must be positive and available")

    case_order = sorted(preserve_by_case, key=lambda case_id: (
        hashlib.sha256(f"{seed}\0{case_id}".encode()).digest(), case_id))
    quotas = Counter({case_id: 0 for case_id in case_order})
    remaining = cap
    while remaining:
        progress = False
        for case_id in case_order:
            if quotas[case_id] < len(preserve_by_case[case_id]):
                quotas[case_id] += 1
                remaining -= 1
                progress = True
                if not remaining:
                    break
        if not progress:
            raise RuntimeError("preservation selection could not meet the cap")
    selected_offsets = set(repair_offsets)
    for case_id, offsets in preserve_by_case.items():
        selected_offsets.update(offsets[index] for index in
                                _evenly_spaced_indices(len(offsets), quotas[case_id])
                                if quotas[case_id])
    if len(selected_offsets) != repair_count + cap:
        raise RuntimeError("balanced PSD target selection is not unique")

    output_dir.mkdir(parents=True, exist_ok=False)
    destination = output_dir / "targets.jsonl"
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        for offset in sorted(selected_offsets):
            input_handle.seek(offset)
            output_handle.write(input_handle.readline())
    quota_histogram = dict(sorted(Counter(quotas.values()).items()))
    manifest = {
        "schema_version": "ifv-psd-balanced-target-selection-v1",
        "selection_policy": "all_verified_repairs_plus_case_round_robin_evenly_spaced_preservation",
        "seed": seed,
        "source_identity": _identity(source),
        "output_identity": _identity(destination),
        "counts": {"repair_targets": repair_count,
                   "source_preservation_targets": available_preserve,
                   "source_preservation_cases": len(preserve_by_case),
                   "selected_preservation_targets": cap,
                   "selected_targets": repair_count + cap,
                   "excluded_preservation_targets": available_preserve - cap},
        "preservation_targets_per_case": quota_histogram,
        "per_target_weights_unchanged": True,
        "source_targets_unchanged": True,
        "provider_calls": 0,
        "large_payload_hashing": False,
        "status": "ready_for_teacher_topk_or_datum_build",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest

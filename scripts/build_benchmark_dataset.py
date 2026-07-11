from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_REAL_EVENT_CARDS = "data/benchmark_candidates/recent_diverse/event_cards/real_photo_event_cards.jsonl"
DEFAULT_SYNTHETIC_REAL_PROMPTS = "data/benchmarks/v0.1/prompt_prep/synthetic_real_prompts.json"
DEFAULT_SYNTHETIC_REAL_MANIFEST = "data/benchmarks/v0.1/generated/synthetic_real/manifest.jsonl"
DEFAULT_SYNTHETIC_FAKE_PROMPTS = "data/benchmarks/v0.1/prompt_prep/synthetic_fake_prompts.json"
DEFAULT_SYNTHETIC_FAKE_MANIFEST = "data/benchmarks/v0.1/generated/synthetic_fake/manifest.jsonl"
DEFAULT_UNVERIFIABLE = "data/benchmarks/v0.1/prompt_prep/unverifiable_candidates.json"
DEFAULT_OUTPUT_DIR = "data/benchmarks/v0.1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a benchmark dataset from real cards, generated images, and unverifiable samples."
    )
    parser.add_argument("--real-event-cards", default=DEFAULT_REAL_EVENT_CARDS)
    parser.add_argument("--synthetic-real-prompts", default=DEFAULT_SYNTHETIC_REAL_PROMPTS)
    parser.add_argument("--synthetic-real-manifest", default=DEFAULT_SYNTHETIC_REAL_MANIFEST)
    parser.add_argument("--synthetic-fake-prompts", default=DEFAULT_SYNTHETIC_FAKE_PROMPTS)
    parser.add_argument("--synthetic-fake-manifest", default=DEFAULT_SYNTHETIC_FAKE_MANIFEST)
    parser.add_argument("--unverifiable", default=DEFAULT_UNVERIFIABLE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-real", type=int, default=12)
    parser.add_argument("--max-synthetic-real", type=int, default=10)
    parser.add_argument("--max-synthetic-fake", type=int, default=10)
    parser.add_argument("--max-unverifiable", type=int, default=8)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _by_sample_id(rows: Iterable[Dict[str, Any]], key: str = "sample_id") -> Dict[str, Dict[str, Any]]:
    return {row[key]: row for row in rows if row.get(key)}


def _real_entry(card: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": card["event_id"],
        "bucket": "real-photo-real",
        "ground_truth": "real",
        "image_path": card.get("image_path"),
        "title": card.get("title"),
        "primary_claim": (card.get("depicted_claims") or [""])[0],
        "depicted_claims": card.get("depicted_claims", []),
        "visible_cues": card.get("visible_cues", []),
        "evidence_urls": card.get("evidence_urls", []),
        "event_metadata": {
            "event_id": card.get("event_id"),
            "source_id": card.get("source_id"),
            "source_type": card.get("source_type"),
            "publish_date": card.get("publish_date"),
            "location_hints": card.get("location_hints", []),
            "event_type": card.get("event_type"),
            "entities": card.get("entities", []),
        },
        "generation_metadata": None,
    }


def _synthetic_entry(
    *,
    prompt_row: Dict[str, Any],
    image_row: Dict[str, Any],
    bucket: str,
    ground_truth: str,
) -> Dict[str, Any]:
    return {
        "sample_id": prompt_row["sample_id"],
        "bucket": bucket,
        "ground_truth": ground_truth,
        "image_path": image_row.get("image_path"),
        "title": prompt_row.get("title") or prompt_row.get("sample_id"),
        "primary_claim": (prompt_row.get("expected_claims") or [""])[0],
        "depicted_claims": prompt_row.get("expected_claims", []),
        "visible_cues": prompt_row.get("visible_cues", []),
        "evidence_urls": prompt_row.get("evidence_urls", []),
        "event_metadata": {
            "source_event_id": prompt_row.get("source_event_id"),
            "source_image_path": prompt_row.get("source_image_path"),
            "real_event_claims": prompt_row.get("real_event_claims", []),
            "why_fake": prompt_row.get("why_fake"),
        },
        "generation_metadata": {
            "provider": image_row.get("provider"),
            "model": image_row.get("model"),
            "size": image_row.get("size"),
            "prompt": image_row.get("prompt") or prompt_row.get("prompt"),
            "response_path": image_row.get("response_path"),
            "metadata": image_row.get("metadata", {}),
        },
    }


def _unverifiable_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "bucket": "unverifiable",
        "ground_truth": "unverifiable",
        "image_path": row.get("image_path"),
        "title": row.get("title"),
        "primary_claim": (row.get("depicted_claims") or [""])[0],
        "depicted_claims": row.get("depicted_claims", []),
        "visible_cues": row.get("visible_cues", []),
        "evidence_urls": [row.get("source_url")] if row.get("source_url") else [],
        "event_metadata": {
            "source_sample_id": row.get("source_sample_id"),
            "reason": row.get("reason"),
        },
        "generation_metadata": None,
    }


def _existing_only(
    prompt_rows: List[Dict[str, Any]],
    manifest_rows: List[Dict[str, Any]],
) -> List[tuple[Dict[str, Any], Dict[str, Any]]]:
    manifest_index = _by_sample_id(manifest_rows)
    pairs: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
    for prompt_row in prompt_rows:
        sample_id = prompt_row.get("sample_id")
        image_row = manifest_index.get(sample_id)
        if image_row and image_row.get("image_path") and Path(image_row["image_path"]).exists():
            pairs.append((prompt_row, image_row))
    return pairs


def build_benchmark_dataset(
    *,
    real_event_cards_path: Path,
    synthetic_real_prompts_path: Path,
    synthetic_real_manifest_path: Path,
    synthetic_fake_prompts_path: Path,
    synthetic_fake_manifest_path: Path,
    unverifiable_path: Path,
    output_dir: Path,
    max_real: int,
    max_synthetic_real: int,
    max_synthetic_fake: int,
    max_unverifiable: int,
) -> Dict[str, Any]:
    real_cards = _load_jsonl(real_event_cards_path)[:max_real]
    synthetic_real_prompts = _load_json(synthetic_real_prompts_path) if synthetic_real_prompts_path.exists() else []
    synthetic_real_manifest = _load_jsonl(synthetic_real_manifest_path)
    synthetic_fake_prompts = _load_json(synthetic_fake_prompts_path) if synthetic_fake_prompts_path.exists() else []
    synthetic_fake_manifest = _load_jsonl(synthetic_fake_manifest_path)
    unverifiable_rows = _load_json(unverifiable_path) if unverifiable_path.exists() else []

    dataset: List[Dict[str, Any]] = []
    real_rows = [_real_entry(card) for card in real_cards if card.get("image_path")]
    dataset.extend(real_rows)

    synthetic_real_pairs = _existing_only(synthetic_real_prompts, synthetic_real_manifest)[:max_synthetic_real]
    synthetic_real_rows = [
        _synthetic_entry(
            prompt_row=prompt_row,
            image_row=image_row,
            bucket="synthetic-real",
            ground_truth="real",
        )
        for prompt_row, image_row in synthetic_real_pairs
    ]
    dataset.extend(synthetic_real_rows)

    synthetic_fake_pairs = _existing_only(synthetic_fake_prompts, synthetic_fake_manifest)[:max_synthetic_fake]
    synthetic_fake_rows = [
        _synthetic_entry(
            prompt_row=prompt_row,
            image_row=image_row,
            bucket="synthetic-fake",
            ground_truth="fake",
        )
        for prompt_row, image_row in synthetic_fake_pairs
    ]
    dataset.extend(synthetic_fake_rows)

    unverifiable_final = [
        _unverifiable_entry(row)
        for row in unverifiable_rows[:max_unverifiable]
        if row.get("image_path") and Path(row["image_path"]).exists()
    ]
    dataset.extend(unverifiable_final)

    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_json_path = output_dir / "benchmark_v0.1.json"
    benchmark_jsonl_path = output_dir / "benchmark_v0.1.jsonl"
    summary_path = output_dir / "summary.json"

    _write_json(benchmark_json_path, dataset)
    _write_jsonl(benchmark_jsonl_path, dataset)

    summary = {
        "output_dir": str(output_dir),
        "num_total": len(dataset),
        "num_real_photo_real": len(real_rows),
        "num_synthetic_real": len(synthetic_real_rows),
        "num_synthetic_fake": len(synthetic_fake_rows),
        "num_unverifiable": len(unverifiable_final),
        "benchmark_json_path": str(benchmark_json_path),
        "benchmark_jsonl_path": str(benchmark_jsonl_path),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    args = _parse_args()
    report = build_benchmark_dataset(
        real_event_cards_path=Path(args.real_event_cards),
        synthetic_real_prompts_path=Path(args.synthetic_real_prompts),
        synthetic_real_manifest_path=Path(args.synthetic_real_manifest),
        synthetic_fake_prompts_path=Path(args.synthetic_fake_prompts),
        synthetic_fake_manifest_path=Path(args.synthetic_fake_manifest),
        unverifiable_path=Path(args.unverifiable),
        output_dir=Path(args.output_dir),
        max_real=args.max_real,
        max_synthetic_real=args.max_synthetic_real,
        max_synthetic_fake=args.max_synthetic_fake,
        max_unverifiable=args.max_unverifiable,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

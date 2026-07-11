from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from src.storage import data_path


DEFAULT_EVENT_CARDS = str(
    data_path(
        "benchmarks/candidates/recent_diverse/event_cards/event_cards.jsonl",
        "data/benchmark_candidates/recent_diverse/event_cards/event_cards.jsonl",
    )
)
DEFAULT_SCORED_MANIFEST = str(
    data_path(
        "artifacts/source_materials/recent_diverse_scored_all/scored_manifest.jsonl",
        "data/source_materials/recent_diverse_scored_all/scored_manifest.jsonl",
    )
)
DEFAULT_OUTPUT_DIR = str(
    data_path(
        "generated/legacy_v0.1/prompt_prep",
        "data/benchmarks/v0.1/prompt_prep",
    )
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare synthetic-real, synthetic-fake, and unverifiable prompt packs."
    )
    parser.add_argument(
        "--event-cards",
        default=DEFAULT_EVENT_CARDS,
        help="JSONL produced by build_event_cards.py",
    )
    parser.add_argument(
        "--scored-manifest",
        default=DEFAULT_SCORED_MANIFEST,
        help="Scored manifest from score_source_materials_with_gemini.py",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store prompt-prep outputs.",
    )
    parser.add_argument(
        "--max-synthetic-real",
        type=int,
        default=10,
        help="Maximum synthetic-real prompts to prepare.",
    )
    parser.add_argument(
        "--max-synthetic-fake",
        type=int,
        default=10,
        help="Maximum synthetic-fake prompts to prepare.",
    )
    parser.add_argument(
        "--max-unverifiable",
        type=int,
        default=8,
        help="Maximum unverifiable prompt candidates to prepare.",
    )
    return parser.parse_args()


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


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _truncate(value: str, limit: int) -> str:
    text = _clean_text(value)
    return text[:limit] if len(text) > limit else text


def _dedupe_by_key(rows: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = set()
    result: List[Dict[str, Any]] = []
    for row in rows:
        value = row.get(key)
        if value in seen:
            continue
        seen.add(value)
        result.append(row)
    return result


def _sort_for_synthetic_real(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -int(row.get("image_scores", {}).get("synthetic_real_potential", 0) or 0),
            row.get("candidate_type") != "synthetic_real_candidate",
            -int(row.get("image_scores", {}).get("image_information_score", 0) or 0),
        ),
    )


def _build_synthetic_real_prompt(card: Dict[str, Any]) -> str:
    title = card.get("title", "")
    claims = card.get("depicted_claims", []) or []
    location_hints = card.get("location_hints", []) or []
    visible_cues = card.get("visible_cues", []) or []
    prompt_seed = card.get("synthetic_prompt_hint") or ""

    lines = [
        "Use case: photorealistic-natural",
        "Asset type: benchmark image for factual verification",
        f"Primary request: Create a realistic documentary-style image for this real event: {title}",
    ]
    if prompt_seed:
        lines.append(f"Scene/backdrop: {prompt_seed}")
    if location_hints:
        lines.append(f"Location anchor: {', '.join(location_hints[:3])}")
    if claims:
        lines.append(f"Facts the image should support: {'; '.join(claims[:3])}")
    if visible_cues:
        lines.append(f"Must-show cues: {'; '.join(visible_cues[:5])}")
    lines.extend(
        [
            "Style/medium: high-resolution news photography",
            "Composition/framing: single-frame image with multiple strong factual cues visible at once",
            "Lighting/mood: natural and believable",
            "Constraints: keep the depicted event factually consistent with the source event; no watermark; no collage; no split panels",
            "Avoid: fantasy elements, exaggerated cinematic effects, extra unrelated text, obvious AI artifacts",
        ]
    )
    return "\n".join(lines)


def _mutate_claim_for_fake(claim: str, card: Dict[str, Any]) -> str:
    claim = _clean_text(claim.rstrip("."))
    replacements = [
        ("NASA", "ESA"),
        ("ESA", "NASA"),
        ("NOAA", "NASA"),
        ("French", "German"),
        ("Lebanon", "Jordan"),
        ("International Space Station", "lunar gateway station"),
        ("Johns Hopkins Applied Physics Laboratory", "Jet Propulsion Laboratory"),
        ("Washington Monument", "U.S. Capitol"),
        ("National Mall", "Times Square"),
        ("European Commission", "United Nations"),
    ]
    for source, target in replacements:
        if source in claim:
            return claim.replace(source, target)

    location_hints = card.get("location_hints", []) or []
    if location_hints:
        hint = str(location_hints[0])
        return f"{claim} in {hint} during a public ceremony attended by officials"

    entities = card.get("entities", []) or []
    if entities:
        return f"{claim} featuring {entities[0]} branding prominently displayed"

    return f"{claim} at a different high-profile location than the real event"


def _build_synthetic_fake_prompt(card: Dict[str, Any]) -> Dict[str, Any]:
    real_claims = card.get("depicted_claims", []) or []
    fake_claims = [_mutate_claim_for_fake(claim, card) for claim in real_claims[:2]]
    title = card.get("title", "")
    location_hints = card.get("location_hints", []) or []
    visible_cues = card.get("visible_cues", []) or []

    prompt_lines = [
        "Use case: photorealistic-natural",
        "Asset type: benchmark image for factual verification",
        f"Primary request: Create a highly realistic image that appears to document this event headline: {title}",
        f"Scene/backdrop: {_clean_text(card.get('synthetic_prompt_hint') or title)}",
        f"False facts the image should imply: {'; '.join(fake_claims)}",
    ]
    if location_hints:
        prompt_lines.append(f"Location anchor: {', '.join(location_hints[:3])}")
    if visible_cues:
        prompt_lines.append(f"Carry over visual style cues: {'; '.join(visible_cues[:4])}")
    prompt_lines.extend(
        [
            "Style/medium: realistic breaking-news photography",
            "Composition/framing: clear factual cues and legible context details in one frame",
            "Lighting/mood: natural and credible",
            "Constraints: make the image look authentic and newsworthy; no watermark; no collage; no obvious synthetic defects",
            "Avoid: surreal effects, extra panels, meme formatting, stylized illustration",
        ]
    )

    why_fake = (
        "Constructed from a real event card with a deliberate factual alteration. "
        f"Original claims: {'; '.join(real_claims[:2])}. "
        f"Fake claims: {'; '.join(fake_claims[:2])}."
    )
    return {
        "sample_id": f"{card['event_id']}_synthetic_fake",
        "title": card.get("title"),
        "prompt": "\n".join(prompt_lines),
        "expected_claims": fake_claims,
        "visible_cues": card.get("visible_cues", [])[:5],
        "ground_truth": "fake",
        "source_event_id": card["event_id"],
        "bucket": "synthetic-fake",
        "why_fake": why_fake,
        "real_event_claims": real_claims[:2],
    }


def _looks_unverifiable(row: Dict[str, Any]) -> bool:
    image_score = int(row.get("image_information_score", 0) or 0)
    synthetic_score = int(row.get("synthetic_real_potential", 0) or 0)
    route = str(row.get("recommended_route", ""))
    claims = row.get("candidate_claims", []) or []
    visible_cues = row.get("visible_cues", []) or []
    return (
        route == "discard_low_signal"
        and image_score <= 45
        and synthetic_score <= 50
        and len(claims) <= 2
        and len(visible_cues) <= 4
    )


def _build_unverifiable_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    title = row.get("title", "")
    cues = row.get("visible_cues", []) or []
    claims = row.get("candidate_claims", []) or []
    return {
        "sample_id": f"{row['sample_id']}_unverifiable",
        "source_sample_id": row["sample_id"],
        "image_path": row.get("local_image_path"),
        "bucket": "unverifiable",
        "ground_truth": "unverifiable",
        "title": title,
        "depicted_claims": claims[:2],
        "visible_cues": cues[:5],
        "reason": _truncate(
            row.get("rationale")
            or "Low-signal image with insufficient event-specific cues for reliable verification.",
            320,
        ),
        "source_url": row.get("source_url"),
    }


def prepare_prompt_packs(
    *,
    event_cards_path: Path,
    scored_manifest_path: Path,
    output_dir: Path,
    max_synthetic_real: int,
    max_synthetic_fake: int,
    max_unverifiable: int,
) -> Dict[str, Any]:
    event_cards = _dedupe_by_key(_load_jsonl(event_cards_path), "event_id")
    scored_rows = _load_jsonl(scored_manifest_path)

    synthetic_real_cards = _sort_for_synthetic_real(event_cards)[:max_synthetic_real]
    synthetic_real_prompts = []
    for card in synthetic_real_cards:
        synthetic_real_prompts.append(
            {
                "sample_id": f"{card['event_id']}_synthetic_real",
                "title": card.get("title"),
                "prompt": _build_synthetic_real_prompt(card),
                "expected_claims": card.get("depicted_claims", [])[:3],
                "visible_cues": card.get("visible_cues", [])[:5],
                "ground_truth": "real",
                "bucket": "synthetic-real",
                "source_event_id": card["event_id"],
                "source_image_path": card.get("image_path"),
                "evidence_urls": card.get("evidence_urls", []),
            }
        )

    fake_source_cards = _sort_for_synthetic_real(event_cards)[:max_synthetic_fake]
    synthetic_fake_prompts = [_build_synthetic_fake_prompt(card) for card in fake_source_cards]

    unverifiable_rows = [
        row for row in scored_rows
        if _looks_unverifiable(row)
    ]
    unverifiable_rows = sorted(
        unverifiable_rows,
        key=lambda row: (
            int(row.get("image_information_score", 0) or 0),
            int(row.get("synthetic_real_potential", 0) or 0),
        ),
    )[:max_unverifiable]
    unverifiable_entries = [_build_unverifiable_entry(row) for row in unverifiable_rows]

    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_real_path = output_dir / "synthetic_real_prompts.json"
    synthetic_fake_path = output_dir / "synthetic_fake_prompts.json"
    unverifiable_path = output_dir / "unverifiable_candidates.json"
    summary_path = output_dir / "summary.json"

    _write_json(synthetic_real_path, synthetic_real_prompts)
    _write_json(synthetic_fake_path, synthetic_fake_prompts)
    _write_json(unverifiable_path, unverifiable_entries)

    summary = {
        "event_cards_path": str(event_cards_path),
        "scored_manifest_path": str(scored_manifest_path),
        "output_dir": str(output_dir),
        "num_synthetic_real_prompts": len(synthetic_real_prompts),
        "num_synthetic_fake_prompts": len(synthetic_fake_prompts),
        "num_unverifiable_candidates": len(unverifiable_entries),
        "synthetic_real_path": str(synthetic_real_path),
        "synthetic_fake_path": str(synthetic_fake_path),
        "unverifiable_path": str(unverifiable_path),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    args = _parse_args()
    report = prepare_prompt_packs(
        event_cards_path=Path(args.event_cards),
        scored_manifest_path=Path(args.scored_manifest),
        output_dir=Path(args.output_dir),
        max_synthetic_real=args.max_synthetic_real,
        max_synthetic_fake=args.max_synthetic_fake,
        max_unverifiable=args.max_unverifiable,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

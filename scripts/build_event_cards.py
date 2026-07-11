from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from src.storage import data_path


DEFAULT_REAL_CANDIDATES = str(
    data_path(
        "benchmarks/candidates/recent_diverse/real_photo_candidates.jsonl",
        "data/benchmark_candidates/recent_diverse/real_photo_candidates.jsonl",
    )
)
DEFAULT_SYNTHETIC_CANDIDATES = str(
    data_path(
        "benchmarks/candidates/recent_diverse/synthetic_real_candidates.jsonl",
        "data/benchmark_candidates/recent_diverse/synthetic_real_candidates.jsonl",
    )
)
DEFAULT_SOURCE_MANIFEST = str(
    data_path(
        "artifacts/source_materials/recent_diverse/manifest.jsonl",
        "data/source_materials/recent_diverse/manifest.jsonl",
    )
)
DEFAULT_OUTPUT_DIR = str(
    data_path(
        "benchmarks/candidates/recent_diverse/event_cards",
        "data/benchmark_candidates/recent_diverse/event_cards",
    )
)

STOP_ENTITY_WORDS = {
    "A",
    "An",
    "And",
    "At",
    "By",
    "For",
    "From",
    "In",
    "Into",
    "Of",
    "On",
    "Or",
    "The",
    "To",
    "With",
}

EVENT_TYPE_RULES = [
    ("space_hardware", ("satellite", "imager", "cleanroom", "spacecraft")),
    ("space_mission_operations", ("mission control", "new horizons", "flight controllers")),
    ("human_spaceflight", ("astronaut", "iss", "space station", "columbus")),
    ("ceremonial_event", ("moon tree", "tree-planting", "ceremony", "planting")),
    ("aviation_flyover", ("flyover", "fighter jets", "f-5 tiger", "national mall")),
    ("robotics_field_test", ("rover", "desert", "field test", "jpl")),
    ("maritime_vessel", ("ship", "vessel", "stern", "hull")),
    ("weather_disaster", ("wildfire", "flash flood", "storm", "hurricane")),
]

LOCATION_KEYWORDS = (
    "National Mall",
    "Washington Monument",
    "International Space Station",
    "ISS Columbus laboratory",
    "Johns Hopkins Applied Physics Laboratory",
    "Mission Operations center",
    "cleanroom facility",
    "desert environment",
    "space station module",
)

GENERIC_ENTITY_PHRASES = {
    "Astronaut",
    "Astronauts",
    "Complex",
    "Digital",
    "European",
    "Flight",
    "Four",
    "Good Health",
    "Hibernation",
    "Installed",
    "Moon",
    "New European",
    "Operations",
    "Personnel",
    "Spacecraft",
    "The Artemis II",
    "The French",
    "Unpacking Europe",
    "While",
    "White Charlatte",
    "Grassy",
}

LOCATION_MARKERS = (
    "Center",
    "Centre",
    "City",
    "Columbus",
    "Complex",
    "Desert",
    "Facility",
    "Field",
    "Guiana",
    "ISS",
    "Laboratory",
    "Lab",
    "Mall",
    "Maryland",
    "Module",
    "Monument",
    "Park",
    "Plaster City",
    "Space Center",
    "Space Station",
    "Spaceport",
    "Washington",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build structured event cards from routed benchmark candidates."
    )
    parser.add_argument(
        "--real-candidates",
        default=DEFAULT_REAL_CANDIDATES,
        help="JSONL of real-photo candidates from route_benchmark_candidates.py",
    )
    parser.add_argument(
        "--synthetic-candidates",
        default=DEFAULT_SYNTHETIC_CANDIDATES,
        help="JSONL of synthetic-real candidates from route_benchmark_candidates.py",
    )
    parser.add_argument(
        "--source-manifest",
        default=DEFAULT_SOURCE_MANIFEST,
        help="Original source-material manifest for enriching context fields.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store event-card outputs.",
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


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        item = _clean_text(value)
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _index_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = _load_jsonl(path)
    return {row.get("sample_id", ""): row for row in rows if row.get("sample_id")}


def _collect_text_blobs(candidate_row: Dict[str, Any], source_row: Optional[Dict[str, Any]]) -> List[str]:
    blobs = [
        candidate_row.get("title", ""),
        " ".join(candidate_row.get("candidate_claims", []) or []),
        " ".join(candidate_row.get("visible_cues", []) or []),
        candidate_row.get("rationale", ""),
        candidate_row.get("synthetic_prompt_hint", "") or "",
    ]
    if source_row:
        blobs.extend(
            [
                source_row.get("page_text", ""),
                source_row.get("feed_description_text", ""),
            ]
        )
    return [_clean_text(blob) for blob in blobs if _clean_text(blob)]


def _infer_event_type(text_blobs: Iterable[str], source_id: str) -> str:
    text = " ".join(text_blobs).lower()
    for event_type, keywords in EVENT_TYPE_RULES:
        if any(_keyword_present(text, keyword) for keyword in keywords):
            return event_type
    if source_id.startswith("nasa") or source_id.startswith("esa"):
        return "space_event"
    if source_id.startswith("noaa"):
        return "environment_event"
    if source_id.startswith("un_"):
        return "international_news"
    return "general_event"


def _keyword_present(text: str, keyword: str) -> bool:
    normalized = keyword.lower().strip()
    if not normalized:
        return False
    if " " in normalized or "-" in normalized:
        return normalized in text
    return re.search(rf"\b{re.escape(normalized)}\b", text) is not None


def _extract_named_phrases(text: str) -> List[str]:
    pattern = re.compile(
        r"\b(?:[A-Z]{2,}(?:\s+[A-Z]{2,})*|[A-Z][a-z]+(?:['-][A-Z]?[a-z]+)?"
        r"(?:\s+(?:[A-Z]{2,}|[A-Z][a-z]+(?:['-][A-Z]?[a-z]+)?|[0-9]{1,4})){0,5})\b"
    )
    results: List[str] = []
    for match in pattern.finditer(text):
        phrase = _clean_text(match.group(0))
        if len(phrase) < 3:
            continue
        tokens = phrase.split()
        if all(token in STOP_ENTITY_WORDS for token in tokens):
            continue
        if phrase.lower() in {"a", "an", "the", "iss"}:
            continue
        results.append(phrase)
    return results


def _normalize_entity_phrase(value: str) -> Optional[str]:
    phrase = _clean_text(value)
    phrase = re.sub(r"^(?:A|An|The)\s+", "", phrase)
    phrase = phrase.strip(" .,:;!-")
    if not phrase or len(phrase) < 3:
        return None
    if phrase in GENERIC_ENTITY_PHRASES:
        return None
    words = phrase.split()
    if len(words) == 1:
        token = words[0]
        if token.lower() in {"crew", "moon", "personnel", "while"}:
            return None
        if token[0].isupper() and token[1:].islower() and token not in {"NASA", "ESA", "ISS", "JPL", "APL"}:
            return None
    if any(char.isdigit() for char in phrase) or any(token.isupper() for token in words):
        return phrase
    if len(words) >= 2 and all(part[:1].isupper() for part in words if part):
        return phrase
    return None


def _sentence_split(text: str) -> List[str]:
    return [
        _clean_text(part)
        for part in re.split(r"(?<=[.!?])\s+", text or "")
        if _clean_text(part)
    ]


def _infer_entities(candidate_row: Dict[str, Any], source_row: Optional[Dict[str, Any]]) -> List[str]:
    phrases: List[str] = []

    preferred_blobs = [
        candidate_row.get("title", ""),
        " ".join(candidate_row.get("candidate_claims", []) or []),
        " ".join(candidate_row.get("visible_cues", []) or []),
        candidate_row.get("synthetic_prompt_hint", "") or "",
    ]
    if source_row:
        page_text = _clean_text(source_row.get("page_text", ""))
        preferred_blobs.extend(_sentence_split(page_text)[:2])

    for blob in preferred_blobs:
        phrases.extend(_extract_named_phrases(blob))

    normalized = []
    for phrase in phrases:
        cleaned = _normalize_entity_phrase(phrase)
        if cleaned:
            normalized.append(cleaned)
    return _dedupe_keep_order(normalized)[:12]


def _normalize_location_hint(value: str) -> Optional[str]:
    hint = _clean_text(value)
    hint = re.sub(r"^(?:in|at|inside|over|near|aboard|within)\s+", "", hint, flags=re.IGNORECASE)
    hint = hint.strip(" .,:;!-")
    if not hint or len(hint) < 4:
        return None
    if any(token in hint for token in LOCATION_MARKERS):
        return hint
    return None


def _infer_location_hints(candidate_row: Dict[str, Any], source_row: Optional[Dict[str, Any]]) -> List[str]:
    hints: List[str] = []
    combined = _collect_text_blobs(candidate_row, source_row)
    for keyword in LOCATION_KEYWORDS:
        if any(keyword.lower() in blob.lower() for blob in combined):
            hints.append(keyword)

    location_pattern = re.compile(
        r"\b(?:in|at|inside|over|near|aboard|within)\s+([A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,6})"
    )
    location_blobs = [
        candidate_row.get("synthetic_prompt_hint", "") or "",
        " ".join(candidate_row.get("candidate_claims", []) or []),
    ]
    if source_row:
        location_blobs.extend(_sentence_split(source_row.get("page_text", ""))[:3])

    for blob in location_blobs:
        for match in location_pattern.finditer(blob):
            cleaned = _normalize_location_hint(match.group(1))
            if cleaned:
                hints.append(cleaned)

    normalized = []
    for hint in hints:
        cleaned = _normalize_location_hint(hint)
        if cleaned:
            normalized.append(cleaned)
    return _dedupe_keep_order(normalized)[:8]


def _build_visual_requirements(
    candidate_row: Dict[str, Any],
    source_row: Optional[Dict[str, Any]],
    *,
    candidate_type: str,
) -> List[str]:
    requirements: List[str] = []
    visible_cues = candidate_row.get("visible_cues", []) or []
    if visible_cues:
        requirements.extend(
            f"Keep visible cue: {cue}" for cue in visible_cues[:5]
        )

    dimension_scores = candidate_row.get("dimension_scores", {}) or {}
    if int(dimension_scores.get("readable_text_signal", 0) or 0) < 3:
        requirements.append("Strengthen readable text, signage, or equipment labels.")
    if int(dimension_scores.get("logo_or_signage_signal", 0) or 0) < 3:
        requirements.append("Include explicit organization or mission branding where it exists in source context.")
    if int(dimension_scores.get("place_or_landmark_signal", 0) or 0) < 3:
        requirements.append("Add a location anchor or distinctive setting element.")
    if int(dimension_scores.get("people_identity_signal", 0) or 0) < 3:
        requirements.append("If people are central to the claim, make identity cues more recognizable.")

    if candidate_type == "synthetic_real_candidate":
        prompt_hint = candidate_row.get("synthetic_prompt_hint")
        if prompt_hint:
            requirements.append(f"Generation seed: {prompt_hint}")
        requirements.append("Synthetic regeneration should increase factual density without changing the underlying event.")
    else:
        requirements.append("Preserve the source photo composition closely; this stays in the real-photo-real pool.")

    if source_row and source_row.get("page_text"):
        excerpt = _clean_text(source_row["page_text"])[:220]
        if excerpt:
            requirements.append(f"Context anchor: {excerpt}")

    return _dedupe_keep_order(requirements)


def _extract_domains(urls: Iterable[str]) -> List[str]:
    domains: List[str] = []
    for url in urls:
        if not url:
            continue
        parsed = urlparse(url)
        if parsed.netloc:
            domains.append(parsed.netloc)
    return _dedupe_keep_order(domains)


def _build_event_card(
    candidate_row: Dict[str, Any],
    source_row: Optional[Dict[str, Any]],
    *,
    candidate_type: str,
) -> Dict[str, Any]:
    text_blobs = _collect_text_blobs(candidate_row, source_row)
    evidence_urls = _dedupe_keep_order(
        [
            candidate_row.get("source_url", ""),
            (source_row or {}).get("final_url", ""),
            (source_row or {}).get("feed_url", ""),
            (source_row or {}).get("image_url", ""),
        ]
    )
    benchmark_bucket = (
        "real-photo-real"
        if candidate_type == "real_photo_candidate"
        else "synthetic-real"
    )

    page_text = _clean_text((source_row or {}).get("page_text", ""))
    feed_desc = _clean_text((source_row or {}).get("feed_description_text", ""))
    context_excerpt = page_text[:600] if page_text else feed_desc[:400]

    return {
        "event_id": candidate_row.get("sample_id"),
        "sample_id": candidate_row.get("sample_id"),
        "candidate_type": candidate_type,
        "benchmark_bucket": benchmark_bucket,
        "ground_truth": "real",
        "source_id": candidate_row.get("source_id"),
        "source_type": (source_row or {}).get("source_type", "unknown"),
        "title": candidate_row.get("title"),
        "publish_date": candidate_row.get("publish_date"),
        "time_anchor": {
            "publish_date": candidate_row.get("publish_date"),
            "granularity": "source_publish_time",
        },
        "location_hints": _infer_location_hints(candidate_row, source_row),
        "event_type": _infer_event_type(text_blobs, candidate_row.get("source_id", "")),
        "entities": _infer_entities(candidate_row, source_row),
        "depicted_claims": candidate_row.get("candidate_claims", []) or [],
        "visible_cues": candidate_row.get("visible_cues", []) or [],
        "visual_requirements": _build_visual_requirements(
            candidate_row,
            source_row,
            candidate_type=candidate_type,
        ),
        "image_path": candidate_row.get("local_image_path"),
        "image_origin": "real_photo" if candidate_type == "real_photo_candidate" else "synthetic_target",
        "image_scores": {
            "image_information_score": candidate_row.get("image_information_score"),
            "synthetic_real_potential": candidate_row.get("synthetic_real_potential"),
            "direct_benchmark_fit": candidate_row.get("direct_benchmark_fit"),
        },
        "source_context_excerpt": context_excerpt,
        "source_page_text_path": (source_row or {}).get("raw_html_path"),
        "evidence_urls": evidence_urls,
        "evidence_domains": _extract_domains(evidence_urls),
        "synthetic_prompt_hint": candidate_row.get("synthetic_prompt_hint"),
        "review_notes": [
            "Heuristic extraction only; confirm named entities and location before benchmark finalization."
        ],
    }


def build_event_cards(
    *,
    real_candidates_path: Path,
    synthetic_candidates_path: Path,
    source_manifest_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    source_index = _index_manifest(source_manifest_path)
    real_rows = _load_jsonl(real_candidates_path)
    synthetic_rows = _load_jsonl(synthetic_candidates_path)

    all_cards: List[Dict[str, Any]] = []
    real_cards: List[Dict[str, Any]] = []
    synthetic_cards: List[Dict[str, Any]] = []

    for row in real_rows:
        card = _build_event_card(
            row,
            source_index.get(row.get("sample_id", "")),
            candidate_type="real_photo_candidate",
        )
        real_cards.append(card)
        all_cards.append(card)

    for row in synthetic_rows:
        card = _build_event_card(
            row,
            source_index.get(row.get("sample_id", "")),
            candidate_type="synthetic_real_candidate",
        )
        synthetic_cards.append(card)
        all_cards.append(card)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_cards_path = output_dir / "event_cards.jsonl"
    real_cards_path = output_dir / "real_photo_event_cards.jsonl"
    synthetic_cards_path = output_dir / "synthetic_real_event_cards.jsonl"
    generation_queue_path = output_dir / "synthetic_generation_queue.jsonl"
    summary_path = output_dir / "summary.json"

    _write_jsonl(all_cards_path, all_cards)
    _write_jsonl(real_cards_path, real_cards)
    _write_jsonl(synthetic_cards_path, synthetic_cards)

    generation_queue = [
        {
            "event_id": card["event_id"],
            "title": card["title"],
            "ground_truth": card["ground_truth"],
            "benchmark_bucket": card["benchmark_bucket"],
            "expected_claims": card["depicted_claims"],
            "prompt_seed": card.get("synthetic_prompt_hint"),
            "visual_requirements": card.get("visual_requirements", []),
            "evidence_urls": card.get("evidence_urls", []),
        }
        for card in synthetic_cards
    ]
    _write_jsonl(generation_queue_path, generation_queue)

    summary = {
        "real_candidates_path": str(real_candidates_path),
        "synthetic_candidates_path": str(synthetic_candidates_path),
        "source_manifest_path": str(source_manifest_path),
        "output_dir": str(output_dir),
        "num_event_cards": len(all_cards),
        "num_real_photo_event_cards": len(real_cards),
        "num_synthetic_real_event_cards": len(synthetic_cards),
        "all_cards_path": str(all_cards_path),
        "real_cards_path": str(real_cards_path),
        "synthetic_cards_path": str(synthetic_cards_path),
        "synthetic_generation_queue_path": str(generation_queue_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = _parse_args()
    report = build_event_cards(
        real_candidates_path=Path(args.real_candidates),
        synthetic_candidates_path=Path(args.synthetic_candidates),
        source_manifest_path=Path(args.source_manifest),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

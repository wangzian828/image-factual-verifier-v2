from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import random
from io import BytesIO
from pathlib import Path
from time import sleep
from threading import local
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from PIL import Image
from src.storage import data_path


load_dotenv()


DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_IMAGE_EDGE = 1024
DEFAULT_MAX_WORKERS = 3
DEFAULT_WORKER_DELAY_MS = 350
DEFAULT_INPUT_MANIFEST = data_path(
    "artifacts/source_materials/recent_official/manifest.jsonl",
    "data/source_materials/recent_official/manifest.jsonl",
)
DEFAULT_OUTPUT_DIR = data_path(
    "artifacts/source_materials/recent_official_scored",
    "data/source_materials/recent_official_scored",
)

_THREAD_LOCAL = local()


SCORING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "image_information_score": {
            "type": "integer",
            "description": "0-100. How much verifiable information the image itself contains.",
        },
        "synthetic_real_potential": {
            "type": "integer",
            "description": "0-100. How useful the event/context is for generating a better synthetic-real benchmark image.",
        },
        "direct_benchmark_fit": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "recommended_route": {
            "type": "string",
            "enum": [
                "real_photo_candidate",
                "synthetic_real_candidate",
                "discard_low_signal",
            ],
        },
        "dimension_scores": {
            "type": "object",
            "properties": {
                "people_identity_signal": {"type": "integer"},
                "readable_text_signal": {"type": "integer"},
                "logo_or_signage_signal": {"type": "integer"},
                "place_or_landmark_signal": {"type": "integer"},
                "event_action_signal": {"type": "integer"},
                "ui_or_screenshot_signal": {"type": "integer"},
                "standalone_claim_richness": {"type": "integer"},
            },
            "required": [
                "people_identity_signal",
                "readable_text_signal",
                "logo_or_signage_signal",
                "place_or_landmark_signal",
                "event_action_signal",
                "ui_or_screenshot_signal",
                "standalone_claim_richness",
            ],
        },
        "visible_cues": {
            "type": "array",
            "items": {"type": "string"},
        },
        "candidate_claims": {
            "type": "array",
            "items": {"type": "string"},
        },
        "rationale": {
            "type": "string",
        },
        "synthetic_prompt_hint": {
            "type": ["string", "null"],
        },
    },
    "required": [
        "image_information_score",
        "synthetic_real_potential",
        "direct_benchmark_fit",
        "recommended_route",
        "dimension_scores",
        "visible_cues",
        "candidate_claims",
        "rationale",
        "synthetic_prompt_hint",
    ],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score recent source materials for benchmark suitability with Gemini."
    )
    parser.add_argument(
        "--input-manifest",
        default=str(DEFAULT_INPUT_MANIFEST),
        help="Manifest created by collect_recent_source_materials.py",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store scoring results.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini multimodal model to use for scoring.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of items to score.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip items that already have a score file.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum number of concurrent Gemini scoring workers.",
    )
    parser.add_argument(
        "--worker-delay-ms",
        type=int,
        default=DEFAULT_WORKER_DELAY_MS,
        help="Base delay in milliseconds before each worker sends a Gemini request.",
    )
    return parser.parse_args()


def _resolve_api_key() -> str:
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required in the environment.")
    return api_key


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "image-factual-verifier/score-materials"})
    return session


def _get_thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = _build_session()
        _THREAD_LOCAL.session = session
    return session


class ScoringRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


def _image_to_inline_part(image_path: Path, max_edge: int = DEFAULT_MAX_IMAGE_EDGE) -> Dict[str, Any]:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if longest > max_edge:
            scale = max_edge / float(longest)
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = image.resize(new_size)

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return {
        "inlineData": {
            "mimeType": "image/jpeg" if mime_type.startswith("image/") else mime_type,
            "data": encoded,
        }
    }


def _build_prompt(row: Dict[str, Any]) -> str:
    title = row.get("title", "")
    source_id = row.get("source_id", "")
    publish_date = row.get("publish_date", "")
    page_text = (row.get("page_text") or "")[:1200]
    feed_desc = (row.get("feed_description_text") or "")[:500]

    return f"""
You are evaluating whether a single image is suitable for an image factual verification benchmark.

Task:
1. Focus first on the image itself.
2. Use the article title and text only as background context for the event.
3. Distinguish between:
   - what is directly visible and verifiable from the image
   - what is only known from the article/context
4. Be conservative. A beautiful image with little factual content should score low.

Benchmark goal:
- We want images that are worth fact-checking from the image alone.
- High-signal images usually contain identifiable people, readable text, logos, signage, landmarks, UI, or a clear event action.
- Low-signal images are generic scenic shots, decorative images, abstract science imagery, or images that need the article text to become meaningful.

Source metadata:
- source_id: {source_id}
- publish_date: {publish_date}
- title: {title}

Feed description:
{feed_desc}

Page text excerpt:
{page_text}

Scoring rubric:
- Each dimension score is 0-5.
- image_information_score is 0-100 and should mainly reflect image-contained evidence.
- synthetic_real_potential is 0-100 and should reflect whether the event/context is valuable enough to regenerate as a richer synthetic-real image.

Recommended route:
- real_photo_candidate: image itself is strong enough to enter real-photo-real candidate pool.
- synthetic_real_candidate: article/event is useful, but the current image is weak; better to regenerate a richer synthetic-real image.
- discard_low_signal: both image and event value are low for benchmark purposes.
""".strip()


def _extract_response_text(payload: Dict[str, Any]) -> str:
    candidates = payload.get("candidates", []) or []
    for candidate in candidates:
        content = candidate.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            text = part.get("text")
            if text:
                return text
    raise ValueError("Gemini response did not contain text output.")


def _score_one(
    *,
    api_key: str,
    model: str,
    row: Dict[str, Any],
    worker_delay_ms: int,
) -> Dict[str, Any]:
    session = _get_thread_session()
    image_path = Path(row["local_image_path"])
    prompt = _build_prompt(row)
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    _image_to_inline_part(image_path),
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseJsonSchema": SCORING_SCHEMA,
        },
    }

    endpoint = os.getenv("GEMINI_TEXT_API_URL") or DEFAULT_ENDPOINT_TEMPLATE.format(model=model)
    last_error: Optional[Exception] = None
    payload = None
    for attempt in range(3):
        try:
            if worker_delay_ms > 0:
                sleep((worker_delay_ms / 1000.0) + random.uniform(0.0, 0.2))
            response = session.post(
                endpoint,
                params={"key": api_key},
                headers={"Content-Type": "application/json"},
                json=body,
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            break
        except requests.exceptions.RequestException as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            response_text = None
            if exc.response is not None:
                try:
                    response_text = exc.response.text[:4000]
                except Exception:
                    response_text = None
            last_error = ScoringRequestError(
                repr(exc),
                status_code=status_code,
                response_text=response_text,
            )
            if attempt == 2:
                raise last_error
            sleep((2 * (attempt + 1)) + random.uniform(0.0, 0.5))
    if payload is None:
        raise RuntimeError(f"Gemini scoring failed: {last_error!r}")
    text = _extract_response_text(payload)
    parsed = json.loads(text)
    return {
        "model": model,
        "score": parsed,
        "raw_response": payload,
    }


def _score_row_task(
    row: Dict[str, Any],
    *,
    api_key: str,
    model: str,
    worker_delay_ms: int,
) -> Dict[str, Any]:
    result = _score_one(
        api_key=api_key,
        model=model,
        row=row,
        worker_delay_ms=worker_delay_ms,
    )
    return {
        "row": row,
        "result": result,
    }


def _rewrite_scored_manifest(scored_dir: Path, destination: Path) -> int:
    rows: List[Dict[str, Any]] = []
    for path in sorted(scored_dir.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def score_manifest(
    input_manifest: Path,
    output_dir: Path,
    *,
    model: str,
    limit: Optional[int],
    skip_existing: bool,
    max_workers: int,
    worker_delay_ms: int,
) -> Dict[str, Any]:
    api_key = _resolve_api_key()
    rows = _load_manifest(input_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_dir = output_dir / "scores"
    raw_dir = output_dir / "raw_responses"
    scored_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    todo_rows: List[Dict[str, Any]] = []
    skipped = 0
    errors: List[Dict[str, Any]] = []

    for row in rows:
        if not row.get("local_image_path"):
            skipped += 1
            continue

        sample_id = row["sample_id"]
        score_path = scored_dir / f"{sample_id}.json"
        raw_path = raw_dir / f"{sample_id}.json"

        if skip_existing and score_path.exists():
            skipped += 1
            continue

        todo_rows.append(row)

    if limit is not None:
        todo_rows = todo_rows[:limit]

    scored = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(
                _score_row_task,
                row,
                api_key=api_key,
                model=model,
                worker_delay_ms=worker_delay_ms,
            ): row
            for row in todo_rows
        }

        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            sample_id = row["sample_id"]
            score_path = scored_dir / f"{sample_id}.json"
            raw_path = raw_dir / f"{sample_id}.json"

            try:
                task_payload = future.result()
            except Exception as exc:
                status_code = exc.status_code if isinstance(exc, ScoringRequestError) else None
                response_text = exc.response_text if isinstance(exc, ScoringRequestError) else None
                errors.append(
                    {
                        "sample_id": sample_id,
                        "source_id": row.get("source_id"),
                        "title": row.get("title"),
                        "source_url": row.get("source_url"),
                        "local_image_path": row.get("local_image_path"),
                        "status_code": status_code,
                        "error": repr(exc),
                        "response_text": response_text,
                    }
                )
                continue

            result = task_payload["result"]
            score_payload = {
                "sample_id": sample_id,
                "source_id": row.get("source_id"),
                "title": row.get("title"),
                "publish_date": row.get("publish_date"),
                "source_url": row.get("source_url"),
                "local_image_path": row.get("local_image_path"),
                "model": result["model"],
                **result["score"],
            }
            score_path.write_text(
                json.dumps(score_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raw_path.write_text(
                json.dumps(result["raw_response"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            scored += 1

    scored_manifest_path = output_dir / "scored_manifest.jsonl"
    total_scored_rows = _rewrite_scored_manifest(scored_dir, scored_manifest_path)

    errors_path = output_dir / "errors.jsonl"
    with errors_path.open("w", encoding="utf-8") as handle:
        for row in errors:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_manifest": str(input_manifest),
        "output_dir": str(output_dir),
        "model": model,
        "scored": scored,
        "skipped": skipped,
        "errors": len(errors),
        "max_workers": max(1, max_workers),
        "worker_delay_ms": max(0, worker_delay_ms),
        "total_scored_rows": total_scored_rows,
        "scored_manifest_path": str(scored_manifest_path),
        "errors_path": str(errors_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = _parse_args()
    report = score_manifest(
        Path(args.input_manifest),
        Path(args.output_dir),
        model=args.model,
        limit=args.limit,
        skip_existing=args.skip_existing,
        max_workers=args.max_workers,
        worker_delay_ms=args.worker_delay_ms,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

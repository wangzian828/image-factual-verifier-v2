import json
from pathlib import Path

from PIL import Image

from scripts.benchmark.prefilter_real_seed import prefilter


def _asset(path: Path, sha256: str) -> dict:
    return {"image_path": str(path), "sha256": sha256}


def test_prefilter_routes_and_groups_without_freezing_cases(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    Image.new("RGB", (400, 300), color="white").save(image_a)
    Image.new("RGB", (400, 300), color="white").save(image_b)

    candidates = [
        {
            "sample_id": "identity",
            "ground_truth": "fake",
            "original_label": "Refuted",
            "claim_assets": [_asset(image_a, "a" * 64)],
            "image_question_count": 1,
            "evidence_urls": ["https://example.com/evidence"],
            "source_article_url": "https://example.com/article",
            "source_metadata": {"image_misuse_types": []},
            "benchmark_routing": {"person_identity_review_required": True},
        },
        {
            "sample_id": "visual",
            "ground_truth": "real",
            "original_label": "Supported",
            "claim_assets": [_asset(image_b, "b" * 64)],
            "image_question_count": 1,
            "evidence_urls": ["https://example.com/source"],
            "source_article_url": "https://example.com/factcheck",
            "source_metadata": {"image_misuse_types": ["Out-of-context"]},
            "benchmark_routing": {"person_identity_review_required": False},
        },
    ]
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text(
        "".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8"
    )

    summary = prefilter(
        candidates_path=candidates_path,
        output_dir=tmp_path / "out",
        run_ocr=False,
        ocr_gpu=False,
        check_urls=False,
        url_workers=1,
        url_timeout=1.0,
        limit=None,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "prefilter.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_id = {row["sample_id"]: row for row in rows}

    assert summary["core_cases_frozen"] == 0
    assert summary["routes_are_suggestions_only"] is True
    assert by_id["identity"]["suggested_route"] == "excluded"
    assert "requires_person_identity_review" in by_id["identity"]["reasons"]
    assert by_id["visual"]["suggested_route"] == "needs_review"
    assert by_id["visual"]["duplicate_group"] == by_id["identity"]["duplicate_group"]
    assert sum(bool(row.get("duplicate_representative")) for row in rows) == 1

import json
from pathlib import Path

from PIL import Image

from scripts.benchmark.render_real_seed_review import render_review


def test_review_page_escapes_content_and_exports_decisions(tmp_path: Path) -> None:
    image = tmp_path / "claim image.png"
    Image.new("RGB", (20, 10), "white").save(image)
    candidate = {
        "sample_id": "case-<script>",
        "ground_truth": "fake",
        "original_label": "Refuted",
        "gold_claim_text": "<script>alert(1)</script>",
        "primary_image_path": str(image),
        "source_article_url": "https://example.com/fact",
        "evidence_urls": ["https://example.com/?a=1&b=2"],
        "image_question_count": 1,
        "claim_assets": [
            {
                "image_path": str(image),
                "filename": image.name,
                "width": 20,
                "height": 10,
                "sha256": "a" * 64,
            }
        ],
        "questions": [{"question": "What <thing>?", "question_type": ["Image-related"], "answer_method": "Image-search"}],
        "license": {"status": "review"},
    }
    prefilter = {
        "sample_id": candidate["sample_id"],
        "suggested_route": "needs_review",
        "review_priority": 80,
        "reasons": ["visible_text_signal_present"],
        "ocr": [{"status": "success", "full_text": "visible <text>", "region_count": 1}],
    }
    candidates = tmp_path / "candidates.jsonl"
    filters = tmp_path / "prefilter.jsonl"
    candidates.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    filters.write_text(json.dumps(prefilter) + "\n", encoding="utf-8")

    result = render_review(
        candidates_path=candidates,
        prefilter_path=filters,
        output_dir=tmp_path / "review",
    )
    rendered = Path(result["review_html"]).read_text(encoding="utf-8")
    queue = json.loads(Path(result["review_queue"]).read_text(encoding="utf-8"))

    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    for decision in ("core", "external_claim_transfer", "excluded", "needs_review"):
        assert f'value="{decision}"' in rendered
    assert "localStorage" in rendered
    assert "Export decisions.json" in rendered
    assert 'id="label-filter"' in rendered
    assert 'id="route-filter"' in rendered
    assert queue["sample_id"] == candidate["sample_id"]
    assert queue["review_priority"] == 80

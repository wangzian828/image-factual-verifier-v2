import json
from pathlib import Path

from PIL import Image

from scripts.benchmark.acquire_averimatec import build_candidate_manifest


def test_candidate_manifest_keeps_gold_out_of_core_runtime_contract(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    extraction_root = dataset_root / "extracted" / "images-fixture"
    extraction_root.mkdir(parents=True)
    image_path = extraction_root / "claim.jpg"
    Image.new("RGB", (24, 12), color="white").save(image_path)

    row = {
        "article": "https://factcheck.example/case",
        "date": "2024-01-01",
        "label": "Refuted",
        "location": "US",
        "questions": [
            {
                "question": "Where was this image first published?",
                "question_type": ["Image-related"],
                "answer_method": "Image-search",
                "input_images": [],
                "answers": [
                    {
                        "source_url": "https://source.example/original",
                        "answer_text": "An older publication.",
                    }
                ],
            }
        ],
        "justification": "The image predates the claimed event.",
        "claim_text": "This image depicts a recent event.",
        "claim_images": ["claim.jpg"],
        "metadata": {
            "transcription": "",
            "modality": "Image-text",
            "image_misuse_types": ["Out-of-context"],
        },
    }

    summary = build_candidate_manifest(
        dataset_root=dataset_root,
        extraction_root=extraction_root,
        split_rows={"train": [row]},
        output_dir=tmp_path / "candidates",
        acquisition_files=[],
    )
    candidate = json.loads(
        (tmp_path / "candidates" / "candidates.jsonl").read_text(encoding="utf-8")
    )

    assert summary["total_candidates"] == 1
    assert candidate["ground_truth"] == "fake"
    assert candidate["gold_claim_text"] == row["claim_text"]
    assert candidate["benchmark_routing"]["core_single_image_status"] == (
        "pending_claim_surface_audit"
    )
    assert candidate["benchmark_routing"][
        "gold_claim_must_not_enter_core_runtime_context"
    ] is True
    assert candidate["claim_assets"][0]["width"] == 24
    assert candidate["claim_assets"][0]["height"] == 12

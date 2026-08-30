from __future__ import annotations

from src.eval.evaluator_private_gold import (
    build_evaluator_private_gold_records,
    case_alias_rows,
    private_gold_index,
)


def _archive_row(case_id: str) -> dict[str, object]:
    return {
        "archive_source_version_id": case_id,
        "candidate_id": f"candidate-{case_id}",
        "assignment_id": f"assignment-{case_id}",
        "factual_status": "refuted",
        "target_claim": f"The image says A won {case_id}.",
        "claim_atom": {
            "subject": case_id,
            "event_or_context": f"final {case_id}",
            "relation_slot": "winner",
            "depicted_value": "A",
        },
        "decisive_visual_atom": "A is displayed as the winner.",
        "automatic_qa": {
            "target_visual_atom_observation": "A is displayed as the winner."
        },
        "evidence": {
            "binding": {
                "source_evidence_span": "B won the final.",
                "source_url": f"https://example.org/{case_id}",
            }
        },
    }


def test_private_sidecar_fills_manifest_gaps_from_archive_for_both_splits() -> None:
    records, summary = build_evaluator_private_gold_records(
        split_rows={
            "test": [
                {
                    "archive_source_version_id": "archive-test",
                    "candidate_id": "reused-candidate",
                    "factual_status": "refuted",
                }
            ],
            "train": [
                {
                    "candidate_id": "train-only",
                    "factual_status": "supported",
                    "target_claim": "The image correctly depicts C.",
                    "claim_atom": {
                        "subject": "C",
                        "event_or_context": "event",
                        "relation_slot": "identity",
                        "depicted_value": "C",
                    },
                    "decisive_visual_atom": "C is visible.",
                    "evidence": {
                        "binding": {
                            "source_evidence_span": "C is correctly identified."
                        }
                    },
                }
            ],
        },
        archive_rows=[_archive_row("archive-test")],
    )

    assert summary["record_count"] == 2
    assert summary["split_counts"] == {"test": 1, "train": 1}
    test_record = records[0]
    assert test_record["case_id"] == "archive-test"
    assert test_record["target_claim"] == "The image says A won archive-test."
    assert test_record["private_target"]["expected_verdict"] == "fake"
    assert records[1]["case_id"] == "train-only"


def test_private_sidecar_index_keeps_only_unambiguous_aliases() -> None:
    records, _ = build_evaluator_private_gold_records(
        split_rows={
            "test": [
                {
                    **_archive_row("archive-a"),
                    "candidate_id": "shared-candidate",
                }
            ],
            "train": [
                {
                    **_archive_row("archive-b"),
                    "candidate_id": "shared-candidate",
                }
            ],
        },
        archive_rows=[],
    )

    indexed = private_gold_index(records)
    assert indexed["archive-a"]["case_id"] == "archive-a"
    assert indexed["archive-b"]["case_id"] == "archive-b"
    assert "shared-candidate" not in indexed
    aliases = case_alias_rows(records)
    assert {"alias": "shared-candidate", "case_id": "archive-a"} not in aliases

from src.eval.private_gold_metrics import (
    PRIVATE_GOLD_BUCKETS,
    annotate_private_gold_category,
    private_gold_audit_summary,
    private_gold_category,
    private_gold_category_counts,
)


def test_private_gold_categories_use_one_definition_for_all_producers() -> None:
    rows = [
        {
            "status": "completed",
            "private_gold_auditable": True,
            "verdict_matches_gold": True,
            "quality_bucket": "strong",
        },
        {
            "status": "completed",
            "private_gold_auditable": True,
            "verdict_matches_gold": True,
            "quality_bucket": "usable",
        },
        {
            "status": "completed",
            "private_gold_auditable": True,
            "verdict_matches_gold": False,
            "quality_bucket": "strong",
        },
    ]

    assert private_gold_category_counts(rows) == {
        PRIVATE_GOLD_BUCKETS[0]: 1,
        PRIVATE_GOLD_BUCKETS[1]: 1,
        PRIVATE_GOLD_BUCKETS[2]: 1,
    }


def test_private_gold_excludes_non_auditable_and_failed_rows() -> None:
    assert private_gold_category(
        {
            "status": "not_auditable",
            "private_gold_auditable": False,
            "verdict_matches_gold": True,
            "quality_bucket": "strong",
        }
    ) is None
    assert private_gold_category(
        {
            "status": "error",
            "private_gold_auditable": True,
            "verdict_matches_gold": False,
        }
    ) is None
    assert private_gold_category(
        {
            "private_gold_auditable": True,
            "verdict_matches_gold": True,
            "quality_bucket": "strong",
        }
    ) is None
    assert private_gold_category(
        {
            "status": "completed",
            "private_gold_auditable": True,
            "verdict_matches_gold": True,
        }
    ) is None


def test_annotation_is_non_destructive() -> None:
    row = {
        "status": "completed",
        "private_gold_auditable": True,
        "verdict_matches_gold": True,
        "quality_bucket": "strong",
    }
    annotated = annotate_private_gold_category(row)
    assert "private_gold_category" in annotated
    assert "private_gold_category" not in row


def test_audit_summary_keeps_engineering_rows_out_of_three_categories() -> None:
    summary = private_gold_audit_summary(
        [
            {
                "status": "completed",
                "private_gold_auditable": True,
                "verdict_matches_gold": True,
                "quality_bucket": "strong",
            },
            {"status": "error", "error_type": "TimeoutError"},
        ]
    )
    assert summary["private_gold_categories"][PRIVATE_GOLD_BUCKETS[0]] == 1
    assert sum(summary["private_gold_categories"].values()) == 1
    assert summary["uncategorized_count"] == 1

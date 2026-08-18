from __future__ import annotations

from src.orchestrator.investigation_models import (
    BootstrapInvestigation,
    InvestigationBrief,
    RetrievalAnchor,
)


def test_bootstrap_investigation_truncates_excess_retrieval_anchors() -> None:
    anchors = [
        RetrievalAnchor(
            anchor_id=f"anchor-{index:02d}",
            kind="text",
            value=f"query {index}",
        ).model_dump()
        for index in range(25)
    ]

    bootstrap = BootstrapInvestigation(
        brief=InvestigationBrief(brief_id="brief-1", case_id="case-1"),
        retrieval_anchors=anchors,
    )

    assert len(bootstrap.retrieval_anchors) == 24
    assert bootstrap.retrieval_anchors[-1].anchor_id == "anchor-23"

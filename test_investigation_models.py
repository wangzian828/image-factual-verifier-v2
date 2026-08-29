from __future__ import annotations

from src.orchestrator.investigation_models import (
    InvestigationBrief,
    RetrievalAnchor,
    VisualBootstrap,
)


def test_visual_bootstrap_keeps_anchor_graph_up_to_state_limit() -> None:
    anchors = [
        RetrievalAnchor(
            anchor_id=f"anchor-{index:02d}",
            kind="text",
            value=f"query {index}",
        ).model_dump()
        for index in range(33)
    ]

    bootstrap = VisualBootstrap(
        brief=InvestigationBrief(brief_id="brief-1", case_id="case-1"),
        retrieval_anchors=anchors,
    )

    assert len(bootstrap.retrieval_anchors) == 32
    assert bootstrap.retrieval_anchors[-1].anchor_id == "anchor-31"

from __future__ import annotations

from scripts.audit_reviewed52_candidates import audit_manifest


def _candidate(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "case_id": "case-a",
        "snapshot": "/frozen/case-a/snapshot-12.json",
        "snapshot_number": 12,
        "evidence_id": "evidence-a",
        "task_id": "task-a",
        "claim_ids": ["claim-a"],
        "claim_statements": ["The animal has a white tail."],
        "relation_scope": "same_relation",
        "relation_stance": "contradicts",
        "directness": "direct",
        "quality": "moderate",
        "reviewed_by_prior_decision": False,
        "source_text": "The source describes the animal as having a white tail.",
        "source_visible_property_hint": "white tail",
        "binding_status": "qualified",
    }
    row.update(overrides)
    return row


def test_offline_audit_classifies_every_candidate_and_links_replay() -> None:
    manifest = {
        "schema_version": "ifv-reviewed52-replay-manifest-v1",
        "run_root": "/frozen",
        "case_count_scanned": 2,
        "snapshot_count_scanned": 20,
        "candidate_evidence_count": 4,
        "candidates": [
            _candidate(),
            _candidate(
                evidence_id="evidence-empty",
                source_visible_property_hint="",
                binding_status="no_concrete_visible_property",
            ),
            _candidate(
                evidence_id="evidence-reviewed",
                reviewed_by_prior_decision=True,
                binding_status="hint_only",
            ),
            _candidate(
                case_id="case-b",
                evidence_id="evidence-long",
                claim_statements=["A person is standing at a podium."],
                source_visible_property_hint=(
                    "red coat and white gloves while holding a book and wearing "
                    "a blue hat; another person stands nearby"
                ),
                relation_scope="partial_relation",
                relation_stance="background",
                directness="indirect",
                quality="weak",
                binding_status="hint_only",
            ),
        ],
    }
    replay = {
        "summaries": [
            {
                "case_id": "case-a",
                "source_evidence_id": "evidence-a",
                "engineering_error": "",
                "engineering_failure_codes": [],
                "decision_2_visual_consumption_mode": "consumed",
                "decision_2_deterministic_visual_consumption_fallback": False,
                "decision_2_deterministic_exhaustion_fallback": False,
                "source_only_follow_up_blocked": False,
            }
        ]
    }

    report = audit_manifest(manifest, replay_batch=replay)

    assert report["candidate_evidence_count"] == 4
    assert report["unique_case_evidence_count"] == 4
    assert report["classified_candidate_count"] == 4
    assert report["classification_complete"] is True
    assert report["primary_status_counts"] == {
        "no_concrete_visible_property": 1,
        "prior_decision_reviewed": 1,
        "relation_unqualified": 1,
        "strictly_qualified": 1,
        "hint_only_unqualified": 0,
    }
    assert report["replay_linked_candidate_count"] == 1
    assert report["replay_matched_summary_count"] == 1
    assert report["replay_unmatched_summary_count"] == 0
    rows = {
        row["evidence_id"]: row
        for row in report["candidates"]
    }
    assert rows["evidence-a"]["replay"]["visual_consumption_mode"] == "consumed"
    assert "replay_unavailable" in rows["evidence-empty"]["flags"]
    assert "reviewed_by_prior_decision" in rows["evidence-reviewed"]["flags"]
    assert "hint_multi_property" in rows["evidence-long"]["flags"]
    assert "relation_scope_not_same_relation" in rows["evidence-long"]["flags"]


def test_offline_audit_rejects_manifest_count_drift() -> None:
    manifest = {
        "candidate_evidence_count": 2,
        "candidates": [_candidate()],
    }

    try:
        audit_manifest(manifest)
    except ValueError as exc:
        assert "candidate_evidence_count" in str(exc)
    else:
        raise AssertionError("manifest count drift must fail closed")

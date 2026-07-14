from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_real_trace import audit_trace
from test_image_only_v2_trajectory import (
    test_scripted_image_only_v2_complete_trajectory,
)


def _scripted_trace(tmp_path: Path) -> Path:
    test_scripted_image_only_v2_complete_trajectory(tmp_path)
    return tmp_path / "traces" / "case_scripted_v2.json"


def test_strict_audit_accepts_complete_image_only_v2_trace(
    tmp_path: Path,
) -> None:
    report = audit_trace(_scripted_trace(tmp_path))

    assert not report.failures(strict_scheduler=True)
    assert report.stats["image_only_actions"] == 4
    assert report.stats["reflections"] == 1
    assert report.stats["decisive_facts"] == 3


def test_strict_audit_rejects_discovery_as_verdict_evidence(
    tmp_path: Path,
) -> None:
    trace_path = _scripted_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    investigation = trace["state"]["investigation_state"]
    discovery_id = investigation["discoveries"][0]["discovery_id"]
    trace["verdict_basis"]["evidence_ids"].append(discovery_id)
    investigation["verdict_basis"]["evidence_ids"].append(discovery_id)
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)
    codes = {
        issue.code
        for issue in report.failures(strict_scheduler=True)
    }

    assert "DISCOVERY_USED_AS_VERDICT_EVIDENCE" in codes
    assert "VERDICT_BASIS_EVIDENCE_UNKNOWN" in codes

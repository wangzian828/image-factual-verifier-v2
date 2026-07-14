from __future__ import annotations

import tempfile
import json
from pathlib import Path

from src.trace_viewer import save_trace_html
from test_image_only_v2_trajectory import (
    test_scripted_image_only_v2_complete_trajectory,
)


def test_trace_viewer_renders_iterative_agent_state() -> None:
    trace = {
        "image_id": "unsafe<script>",
        "verdict": "unverifiable",
        "confidence": 0.42,
        "termination": "success",
        "total_tool_calls": 2,
        "llm_api_calls": 6,
        "state": {
            "image_path": "missing.png",
            "plan_history": [
                {"revision": 0, "revision_reason": "", "questions": [{"question_id": "q0", "priority": 1, "question": "First?", "suggested_tools": ["text_search"]}]},
                {"revision": 1, "revision_reason": "q1 unresolved", "questions": [{"question_id": "q1", "priority": 1, "question": "Second?", "suggested_tools": ["visit"]}]},
            ],
            "coverage_audits": [
                {"iteration": 1, "complete": False, "reason": "q1 unresolved", "question_resolutions": [{"question_id": "q1", "status": "in_progress", "tool_attempts": 1, "evidence_count": 0, "remaining_gap": "direct source"}]},
                {"iteration": 2, "complete": True, "reason": "covered", "question_resolutions": [{"question_id": "q1", "status": "resolved", "tool_attempts": 2, "evidence_count": 1, "remaining_gap": ""}]},
            ],
            "all_steps": [
                {"round": 1, "stage": "verification", "action_type": "tool_call", "tool_name": "text_search", "tool_args": {"__question_id": "q1"}, "tool_result": "source", "metadata": {"verification_iteration": 1, "native_interactions": True, "previous_interaction_id": None, "interaction_id": "interaction-root-with-a-long-readable-id"}},
                {"round": 1, "stage": "replanning", "action_type": "output", "output": {"revision": 1}, "metadata": {}},
                {"round": 1, "stage": "verification", "action_type": "output_rejected", "metadata": {"verification_iteration": 2, "native_interactions": True, "forced_output": True, "previous_interaction_id": "interaction-root-with-a-long-readable-id", "interaction_id": "interaction-forced-child", "rejection_reason": "coverage missing <x>"}},
            ],
            "judgment": {"verdict": "unverifiable", "confidence": 0.42, "reasoning_chain": "Evidence remained incomplete.", "key_evidence": [], "anomalies": [], "overall_assessment": "Not enough evidence."},
            "stage_timings": {"verification": 1.2},
        },
    }
    path = Path(tempfile.mkdtemp(prefix="trace-viewer-test-")) / "trace.html"
    save_trace_html(trace, str(path))
    rendered = path.read_text(encoding="utf-8")
    assert "Plan revision 1" in rendered
    assert "Coverage Audits" in rendered
    assert "Iteration 1: replanning required" in rendered
    assert "Iteration 2: complete" in rendered
    assert "Verification iteration 2" in rendered
    assert "replanning" in rendered
    assert "Gemini Interactions" in rendered
    assert rendered.count('class="interaction-chain"') == 2
    assert "previous_interaction_id" in rendered
    assert "interaction_id" in rendered
    assert "null (root)" in rendered
    assert "interaction-root-with-a-long-readable-id" in rendered
    assert "interaction-forced-child" in rendered
    assert "Forced output" in rendered
    assert "overflow-wrap:anywhere" in rendered
    assert "grid-template-columns:minmax(0,1fr)" in rendered
    assert "coverage missing &lt;x&gt;" in rendered
    assert "coverage missing <x>" not in rendered


def test_trace_viewer_renders_image_only_visual_fact_sections(
    tmp_path: Path,
) -> None:
    test_scripted_image_only_v2_complete_trajectory(tmp_path)
    trace_path = tmp_path / "traces" / "case_scripted_v2.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    output = tmp_path / "trace.html"

    save_trace_html(trace, str(output))

    rendered = output.read_text(encoding="utf-8")
    assert "VisualFact Bootstrap" in rendered
    assert "Visual Facts" in rendered
    assert "Research Tasks" in rendered
    assert "Findings &amp; Evidence" in rendered
    assert "Reflection Checkpoints" in rendered
    assert "Verdict Basis" in rendered


if __name__ == "__main__":
    test_trace_viewer_renders_iterative_agent_state()
    print("Trace viewer test passed.")

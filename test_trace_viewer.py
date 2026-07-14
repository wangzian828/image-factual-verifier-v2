from __future__ import annotations

import json
from pathlib import Path

from src.trace_viewer import save_trace_html
from test_image_only_trajectory import (
    test_scripted_image_only_complete_trajectory,
)

def test_trace_viewer_renders_image_only_visual_fact_sections(
    tmp_path: Path,
) -> None:
    test_scripted_image_only_complete_trajectory(tmp_path)
    trace_path = tmp_path / "traces" / "case_scripted_v3.json"
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

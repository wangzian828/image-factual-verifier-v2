from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.tools.focused_visual_inspection import FocusedVisualInspectionTool


class RecordingMultiViewClient:
    def __init__(self) -> None:
        self.calls = []

    def create_images_json(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "answer_status": "observed",
            "summary": "The anchored subject and object are visibly adjacent.",
            "observations": [
                {
                    "view_index": 1,
                    "statement": "The relation context contains both anchors.",
                    "property_status": "observed",
                    "confidence": 0.94,
                }
            ],
            "limitations": [],
        }


def test_focused_visual_inspection_uses_original_and_anchor_views(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (400, 240), color=(120, 140, 160)).save(image_path)
    client = RecordingMultiViewClient()
    tool = FocusedVisualInspectionTool(
        client=client,
        provider="controlled",
        model_name="controlled-vlm",
    )

    result = tool.call(
        {
            "image_input": str(image_path),
            "visual_question_id": "visual-question-1",
            "question": "Are the subject and object visibly adjacent?",
            "expected_property": "Both anchors appear in one coherent scene.",
            "scope": "relation",
            "anchor_regions": [
                [0.1, 0.2, 0.3, 0.7],
                [0.6, 0.25, 0.85, 0.75],
            ],
            "active_fact": "The image depicts the subject next to the object.",
            "evidence_context": "A source introduced this visible relation.",
        }
    )

    assert result["status"] == "success"
    assert result["answer_status"] == "observed"
    assert result["views"][0] == {
        "view_index": 0,
        "kind": "original",
        "region": [0.0, 0.0, 1.0, 1.0],
    }
    assert result["views"][1]["kind"] == "relation_context"
    assert result["observations"][0]["view_kind"] == "relation_context"
    assert len(client.calls) == 1
    supplied_images = client.calls[0]["image_inputs"]
    assert supplied_images[0] == str(image_path)
    assert len(supplied_images) >= 3
    assert all(not Path(path).exists() for path in supplied_images[1:])


def test_focused_visual_inspection_keeps_global_question_on_original(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "global.png"
    Image.new("RGB", (120, 120), color=(20, 40, 60)).save(image_path)
    client = RecordingMultiViewClient()
    tool = FocusedVisualInspectionTool(
        client=client,
        provider="controlled",
        model_name="controlled-vlm",
    )

    result = tool.call(
        {
            "image_input": str(image_path),
            "visual_question_id": "visual-question-global",
            "question": "Does the complete scene contain the expected cue?",
            "expected_property": "A coherent scene-level cue is visible.",
            "scope": "scene",
            "anchor_regions": [],
            "active_fact": "The image depicts the proposed scene.",
            "evidence_context": "The source introduced a scene hypothesis.",
        }
    )

    assert result["status"] == "success"
    assert result["views"] == [
        {
            "view_index": 0,
            "kind": "original",
            "region": [0.0, 0.0, 1.0, 1.0],
        }
    ]
    assert client.calls[0]["image_inputs"] == [str(image_path)]

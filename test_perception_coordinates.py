from __future__ import annotations

from src.tools.perceive_scene import PerceiveSceneTool, normalize_entity_bbox


class FakePerceptionClient:
    def create_image_json(self, **_kwargs):
        return {
            "entities": [
                {
                    "name": "Elon Musk",
                    "entity_type": "person",
                    "bbox": [195, 88, 904, 560],
                    "confidence": 0.9,
                    "attributes": {},
                }
            ],
            "scene_description": "Two people stand behind a car.",
            "image_type": "photo",
        }


def test_gemini_1000_yxyx_bbox_is_converted_at_perception_boundary() -> None:
    tool = PerceiveSceneTool(client=FakePerceptionClient())

    result = tool.call({"image_input": "not-read-by-fake.jpg"})

    assert result["status"] == "success"
    assert result["entities"][0]["bbox"] == [0.088, 0.195, 0.56, 0.904]


def test_normalized_xyxy_is_preserved_and_invalid_boxes_are_rejected() -> None:
    assert normalize_entity_bbox([0.1, 0.2, 0.8, 0.9]) == [0.1, 0.2, 0.8, 0.9]

    try:
        normalize_entity_bbox([0.8, 0.2, 0.1, 0.9])
    except ValueError as exc:
        assert "ordered" in str(exc)
    else:
        raise AssertionError("invalid bbox was accepted")

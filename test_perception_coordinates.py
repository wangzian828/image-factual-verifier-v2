from __future__ import annotations

from src.tools.perceive_scene import (
    PERCEIVE_SCENE_SCHEMA,
    PerceiveSceneTool,
    normalize_entity_bbox,
)


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


class MixedScalePerceptionClient:
    def create_image_json(self, **_kwargs):
        return {
            "entities": [
                {
                    "name": "Brandenburg Gate",
                    "entity_type": "building",
                    "bbox": [0.158, 226, 0.788, 776],
                    "confidence": 0.99,
                },
                {
                    "name": "Berlin Wall",
                    "entity_type": "scene_element",
                    "bbox": [0.77, 0.0, 1.0, 1.0],
                    "confidence": 0.98,
                },
            ],
            "scene_description": (
                "A crowd stands on the Berlin Wall in front of the "
                "Brandenburg Gate."
            ),
            "image_type": "photo",
        }


def test_one_invalid_bbox_does_not_discard_the_scene_report() -> None:
    tool = PerceiveSceneTool(client=MixedScalePerceptionClient())

    result = tool.call({"image_input": "not-read-by-fake.jpg"})

    assert result["status"] == "success"
    assert result["scene_description"].startswith("A crowd stands")
    assert result["entities"][0]["name"] == "Brandenburg Gate"
    assert result["entities"][0]["bbox"] == []
    assert result["entities"][1]["bbox"] == [0.77, 0.0, 1.0, 1.0]
    assert result["bbox_warnings"] == [
        {
            "entity_index": 0,
            "entity_name": "Brandenburg Gate",
            "error": (
                "bbox mixes normalized and Gemini 0..1000 coordinate scales"
            ),
        }
    ]


def test_perception_schema_has_no_unbounded_entity_payload() -> None:
    entities = PERCEIVE_SCENE_SCHEMA["properties"]["entities"]
    entity_properties = entities["items"]["properties"]

    assert entities["maxItems"] == 8
    assert entity_properties["name"]["maxLength"] == 100
    assert "attributes" not in entity_properties
    assert PERCEIVE_SCENE_SCHEMA["properties"]["scene_description"]["maxLength"] == 280


class VerboseFakePerceptionClient:
    def create_image_json(self, **_kwargs):
        return {
            "entities": [
                {
                    "name": f"entity-{index}",
                    "entity_type": "object",
                    "bbox": [],
                    "confidence": 0.8,
                    "attributes": {"unbounded": "must be discarded"},
                }
                for index in range(12)
            ],
            "scene_description": "A bounded visual inventory.",
            "image_type": "photo",
        }


def test_perception_discards_attributes_and_caps_entities() -> None:
    tool = PerceiveSceneTool(client=VerboseFakePerceptionClient())

    result = tool.call({"image_input": "not-read-by-fake.jpg"})

    assert result["status"] == "success"
    assert result["total_entities"] == 8
    assert len(result["entities"]) == 8
    assert all(entity["attributes"] == {} for entity in result["entities"])

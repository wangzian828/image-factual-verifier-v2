from src.orchestrator.stage_runner import StageRunner


def test_hidden_image_input_is_bound_into_visual_cache_key():
    first = StageRunner.__new__(StageRunner)
    first.image_path = "/frozen/images/00001.jpg"
    second = StageRunner.__new__(StageRunner)
    second.image_path = "/frozen/images/00027.jpg"

    for tool in (
        "perceive_scene", "ocr_with_position", "reverse_image_search",
        "focused_visual_inspection", "crop_and_inspect", "count_objects",
        "check_consistency", "compare_with_reference",
        "analyze_visual_anomalies",
    ):
        one = first._build_cache_args(tool, {})
        two = second._build_cache_args(tool, {})
        assert one["__image_input__"] == first.image_path
        assert two["__image_input__"] == second.image_path
        assert one != two


def test_nonvisual_search_cache_key_does_not_gain_image_identity():
    runner = StageRunner.__new__(StageRunner)
    runner.image_path = "/frozen/images/00001.jpg"
    assert runner._build_cache_args("text_image_search", {"query": "event"}) == {
        "query": "event"
    }

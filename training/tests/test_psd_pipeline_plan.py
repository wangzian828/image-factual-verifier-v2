from ifv_training.psd_pipeline_plan import runnable, validate_plan


def test_pipeline_plan_is_acyclic():
    validate_plan()


def test_cpu_and_gemini_can_overlap_qwen_collection():
    # Per-case streaming workers are allowed to consume sealed collection rows.
    assert runnable({"collect"}, set())[:2] == ["source_review"]


def test_repair_qwen_does_not_compete_with_collection():
    completed = {"collect", "source_review", "postprocess", "candidate_extract", "repair_propose"}
    assert "repair_rollout" not in runnable(completed - {"collect"}, {"collect"})
    assert "repair_rollout" in runnable(completed, set())


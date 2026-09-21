import asyncio
from types import SimpleNamespace

from src.workflow import VerificationWorkflow


def test_run_batch_emits_results_as_each_case_finishes():
    workflow = VerificationWorkflow.__new__(VerificationWorkflow)
    workflow.config = SimpleNamespace(sampling_seed=7)

    async def run_single(path, image_id, *, runtime_case=None):
        await asyncio.sleep(0.02 if image_id == "slow" else 0)
        return {"image_id": image_id, "image_path": path, "verdict": "real"}

    workflow.run_single = run_single
    seen = []

    async def exercise():
        result = await workflow.run_batch(
            ["slow.jpg", "fast.jpg"],
            ["slow", "fast"],
            concurrency=2,
            on_result=lambda index, value: seen.append((index, value["image_id"])),
        )
        assert [row["image_id"] for row in result] == ["slow", "fast"]

    asyncio.run(exercise())
    assert seen == [(1, "fast"), (0, "slow")]

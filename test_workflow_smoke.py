# -*- coding: utf-8 -*-
"""Workflow-level smoke test using the active pipeline contract."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile

from PIL import Image

from src.workflow import VerificationWorkflow, WorkflowConfig
from test_unit import FakeOrchestrator


async def main() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="workflow-smoke-")
    image_path = os.path.join(tmp_dir, "sample.png")
    output_dir = os.path.join(tmp_dir, "outputs")
    Image.new("RGB", (48, 48), color=(200, 200, 200)).save(image_path)

    try:
        workflow = VerificationWorkflow(
            WorkflowConfig(
                provider="lmdeploy",
                model_name="fake-model",
                output_dir=output_dir,
                save_traces=True,
            )
        )
        workflow._orchestrator = FakeOrchestrator(image_path)
        result = await workflow.run_single(image_path, "workflow-sample")

        trace_json = os.path.join(output_dir, "workflow-sample.json")
        trace_html = os.path.join(output_dir, "workflow-sample.html")
        assert os.path.exists(trace_json)
        assert os.path.exists(trace_html)

        with open(trace_json, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["verdict"] == "real"
        assert any(step["stage"] == "verification" for step in saved["state"]["all_steps"])

        print("Workflow smoke test passed.")
        print(trace_json)
        print(trace_html)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())

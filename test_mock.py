# -*- coding: utf-8 -*-
"""Deterministic smoke runner for the active orchestrator pipeline."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile

from PIL import Image

from test_unit import FakeOrchestrator
from src.trace_viewer import save_trace_html


async def main() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="verifier-mock-")
    image_path = os.path.join(tmp_dir, "sample.png")
    Image.new("RGB", (64, 64), color=(240, 240, 240)).save(image_path)

    try:
        orchestrator = FakeOrchestrator(image_path)
        result = await orchestrator.run(image_path, "mock-sample")

        with open("test_mock_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        save_trace_html(result, "test_mock_result.html")

        print("=" * 72)
        print("Mock Pipeline Smoke Test")
        print("=" * 72)
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Termination: {result['termination']}")
        print(f"Tool calls: {result['total_tool_calls']}")
        print(f"LLM calls: {result['llm_api_calls']}")
        print("\nRecorded stages/tools:")
        for step in result["state"]["all_steps"]:
            if step["action_type"] == "tool_call":
                print(f"  - {step['stage']}: {step['tool_name']}")
        print("\nArtifacts:")
        print("  - test_mock_result.json")
        print("  - test_mock_result.html")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())

# -*- coding: utf-8 -*-
"""CLI entry point for running image verification."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from src.workflow import VerificationWorkflow, WorkflowConfig


def main():
    parser = argparse.ArgumentParser(description="Image Factual Verifier v2")
    parser.add_argument("image_path", help="Path to the image to verify")
    parser.add_argument("--provider", default="lmdeploy", help="LLM provider")
    parser.add_argument("--model", default="/gsdata/home/wza/models/Qwen3-VL-8B-Thinking", help="Model name")
    parser.add_argument("--output-dir", default="outputs/traces", help="Trace output directory")
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout in seconds")
    parser.add_argument("--no-trace", action="store_true", help="Don't save trace files")
    args = parser.parse_args()

    config = WorkflowConfig(
        provider=args.provider,
        model_name=args.model,
        output_dir=args.output_dir,
        timeout=args.timeout,
        save_traces=not args.no_trace,
    )

    workflow = VerificationWorkflow(config)
    result = asyncio.run(workflow.run_single(args.image_path))

    # Print summary
    print(f"\n{'='*60}")
    print(f"Image: {args.image_path}")
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Assessment: {result['overall_assessment']}")
    print(f"Time: {result['time_taken']:.1f}s")
    print(f"Tool calls: {result['total_tool_calls']}")
    print(f"Tokens: {result['token_usage']}")
    print(f"{'='*60}\n")

    # Print full judgment
    if result.get("judgment"):
        print(json.dumps(result["judgment"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

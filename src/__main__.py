# -*- coding: utf-8 -*-
"""CLI entry point for running image verification."""
from __future__ import annotations

import argparse
import asyncio
import json

from dotenv import load_dotenv
load_dotenv()

from src.workflow import VerificationWorkflow, WorkflowConfig
from src.storage import default_trace_dir


def main():
    parser = argparse.ArgumentParser(description="Image Factual Verifier v3")
    parser.add_argument("image_path", help="Path to the image to verify")
    parser.add_argument(
        "--claim",
        default=None,
        help="Optional external factual claim. Without it, the claim is recovered from visible pixels/OCR.",
    )
    parser.add_argument("--provider", default="gemini", help="LLM provider")
    parser.add_argument("--model", default="gemini-3.5-flash", help="Model name")
    parser.add_argument(
        "--llm-wire-api",
        default=None,
        choices=["interactions", "responses", "chat_completions"],
        help="Wire protocol. Gemini accepts interactions only.",
    )
    parser.add_argument(
        "--vlm-wire-api",
        default=None,
        choices=["interactions", "responses", "chat_completions"],
        help="Vision wire protocol. Gemini accepts interactions only.",
    )
    parser.add_argument(
        "--output-dir",
        default=default_trace_dir(),
        help="Trace output directory",
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="Timeout in seconds")
    parser.add_argument("--no-trace", action="store_true", help="Don't save trace files")
    args = parser.parse_args()

    config = WorkflowConfig(
        provider=args.provider,
        model_name=args.model,
        llm_wire_api=args.llm_wire_api,
        vlm_wire_api=args.vlm_wire_api,
        output_dir=args.output_dir,
        timeout=args.timeout,
        save_traces=not args.no_trace,
    )

    workflow = VerificationWorkflow(config)
    result = asyncio.run(workflow.run_single(args.image_path, user_claim=args.claim))

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

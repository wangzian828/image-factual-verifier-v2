from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integrations.gemini import GeminiInteractionsClient, extract_function_calls


load_dotenv()


def _schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }


def _tool(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "interactions_probe",
        "description": "Return the integer used by the protocol probe.",
        "parameters": schema,
    }


async def probe_once(model: str, timeout: float) -> Dict[str, Any]:
    schema = _schema()
    tool = _tool(schema)
    response_format = {
        "type": "text",
        "mime_type": "application/json",
        "schema": schema,
    }
    generation_config = {
        "max_output_tokens": 128,
        "thinking_level": "minimal",
    }
    async with GeminiInteractionsClient(timeout=timeout, max_retries=0) as client:
        structured = await client.create(
            model=model,
            input="Return the integer 7 in the required JSON schema.",
            response_format=response_format,
            generation_config=generation_config,
            store=True,
        )
        action = await client.create(
            model=model,
            input=(
                "Call interactions_probe once with value 7. After receiving its "
                "function result, return that value as JSON without another tool call."
            ),
            tools=[tool],
            generation_config=generation_config,
            store=True,
        )
        calls = extract_function_calls(action)
        if len(calls) != 1 or not str(calls[0].get("id", "")).strip():
            raise RuntimeError("Gemini did not return exactly one native function call with an id.")
        call = calls[0]
        continuation = await client.create(
            model=model,
            input=[
                {
                    "type": "function_result",
                    "name": "interactions_probe",
                    "call_id": call["id"],
                    "result": [{"type": "text", "text": '{"value":7}'}],
                }
            ],
            previous_interaction_id=action["id"],
            tools=[tool],
            response_format=response_format,
            generation_config=generation_config,
            store=True,
        )
    if continuation.get("status") != "completed":
        raise RuntimeError(
            "Function-result continuation did not complete: "
            + json.dumps(
                {
                    "id": continuation.get("id"),
                    "status": continuation.get("status"),
                    "function_calls": extract_function_calls(continuation),
                },
                ensure_ascii=True,
            )
        )
    return {
        "structured_interaction_id": structured["id"],
        "structured_status": structured["status"],
        "function_interaction_id": action["id"],
        "function_status": action["status"],
        "function_call_id": call["id"],
        "function_name": call.get("name"),
        "continuation_interaction_id": continuation["id"],
        "continuation_status": continuation["status"],
        "previous_interaction_id": action["id"],
    }


async def main_async(model: str, timeout: float, repeat: int) -> None:
    results = []
    for sequence in range(1, repeat + 1):
        result = await probe_once(model, timeout)
        results.append({"sequence": sequence, **result})
    print(json.dumps({"model": model, "probes": results}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe the complete Gemini Interactions structured/tool continuation contract."
    )
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(main_async(args.model, max(1.0, args.timeout), max(1, args.repeat)))


if __name__ == "__main__":
    main()

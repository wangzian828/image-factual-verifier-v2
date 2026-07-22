from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer none",
        },
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def _completion(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = _request(f"{base_url.rstrip('/')}/chat/completions", payload)
    choices = result.get("choices") or []
    if len(choices) != 1:
        raise SystemExit(f"expected one completion choice, got {len(choices)}")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise SystemExit("completion did not contain an assistant message")
    return message


def _json_content(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, str):
        raise SystemExit("structured completion content is not text")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"structured completion is not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("structured completion is not a JSON object")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8899/v1")
    parser.add_argument("--model")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.rounds < 1 or args.rounds > 8:
        raise SystemExit("--rounds must be between 1 and 8")
    if args.max_tokens < 512:
        raise SystemExit("--max-tokens must be at least 512 for Thinking checkpoints")

    base_url = args.base_url.rstrip("/")
    models = _request(f"{base_url}/models")
    model = args.model or models["data"][0]["id"]

    text_message = _completion(
        base_url,
        {
            "model": model,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "messages": [{"role": "user", "content": "Reply with READY."}],
        },
    )
    if not str(text_message.get("content") or "").strip():
        raise SystemExit("text completion was empty")

    image_message = _completion(
        base_url,
        {
            "model": model,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe one visible property."},
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(args.image)},
                        },
                    ],
                }
            ],
        },
    )
    if not str(image_message.get("content") or "").strip():
        raise SystemExit("image completion was empty")

    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ready"]},
            "count": {"type": "integer"},
        },
        "required": ["status", "count"],
        "additionalProperties": False,
    }
    structured = _json_content(
        _completion(
            base_url,
            {
                "model": model,
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": "Return status ready and count 1.",
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "probe_status",
                        "strict": True,
                        "schema": schema,
                    },
                },
            },
        )
    )
    if structured.get("status") != "ready" or structured.get("count") != 1:
        raise SystemExit(f"JSON-schema output mismatch: {structured}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect",
                "description": "Inspect one property from the explicit state packet.",
                "parameters": {
                    "type": "object",
                    "properties": {"property": {"type": "string"}},
                    "required": ["property"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    tool_roundtrips: list[dict[str, Any]] = []
    for round_index in range(args.rounds):
        # Each loop is a new lifecycle. Only the explicit state packet crosses rounds.
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "Use the available tool, then return the requested JSON.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "inspect one property",
                        "round": round_index + 1,
                        "completed_rounds": round_index,
                    }
                ),
            },
        ]
        call_message = _completion(
            base_url,
            {
                "model": model,
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "messages": messages,
                "tools": tools,
                "tool_choice": "required",
            },
        )
        tool_calls = call_message.get("tool_calls") or []
        if len(tool_calls) != 1:
            raise SystemExit(
                f"round {round_index + 1}: expected one tool call, got {len(tool_calls)}"
            )
        call = tool_calls[0]
        call_id = call.get("id")
        function = call.get("function") or {}
        if not call_id or function.get("name") != "inspect":
            raise SystemExit(f"round {round_index + 1}: malformed tool call {call}")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"round {round_index + 1}: tool arguments are not JSON"
            ) from exc

        messages.extend(
            [
                call_message,
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        {
                            "status": "success",
                            "round": round_index + 1,
                            "observation": "recorded",
                        }
                    ),
                },
                {
                    "role": "user",
                    "content": "Return one JSON object with round and accepted.",
                },
            ]
        )
        output = _json_content(
            _completion(
                base_url,
                {
                    "model": model,
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "none",
                    "response_format": {"type": "json_object"},
                },
            )
        )
        if output.get("round") != round_index + 1 or output.get("accepted") is not True:
            raise SystemExit(f"round {round_index + 1}: output mismatch {output}")
        tool_roundtrips.append(
            {"round": round_index + 1, "arguments": arguments, "output": output}
        )

    print(
        json.dumps(
            {
                "model": model,
                "text": str(text_message.get("content")),
                "image": str(image_message.get("content")),
                "structured": structured,
                "tool_roundtrips": tool_roundtrips,
                "lifecycle": "independent_tool_roundtrips",
                "max_tokens": args.max_tokens,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

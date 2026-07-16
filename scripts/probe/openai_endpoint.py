from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from urllib.request import Request, urlopen


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _request(url: str, payload: dict | None = None) -> dict:
    request = Request(
        url,
        data=(
            json.dumps(payload).encode("utf-8")
            if payload is not None
            else None
        ),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer none",
        },
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8899/v1")
    parser.add_argument("--model")
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    models = _request(f"{args.base_url.rstrip('/')}/models")
    model = args.model or models["data"][0]["id"]
    image_url = _data_url(args.image)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect",
                "description": "Inspect one visible property.",
                "parameters": {
                    "type": "object",
                    "properties": {"property": {"type": "string"}},
                    "required": ["property"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Call inspect exactly once for the most decisive visible property."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "tools": tools,
        "tool_choice": "required",
    }
    result = _request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        payload,
    )
    message = result["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    if len(tool_calls) != 1:
        raise SystemExit(f"expected exactly one tool call, got {len(tool_calls)}")
    continuation = {
        "model": model,
        "temperature": 0,
        "messages": [
            *payload["messages"],
            message,
            {
                "role": "tool",
                "tool_call_id": tool_calls[0]["id"],
                "content": json.dumps({"status": "ok", "property": "observed"}),
            },
            {
                "role": "user",
                "content": "Return exactly one JSON object with keys summary and ready.",
            },
        ],
        "response_format": {"type": "json_object"},
    }
    final = _request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        continuation,
    )
    content = final["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if set(parsed) != {"summary", "ready"}:
        raise SystemExit(f"structured output keys mismatch: {sorted(parsed)}")
    print(
        json.dumps(
            {
                "model": model,
                "tool_call": tool_calls[0],
                "continuation": parsed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

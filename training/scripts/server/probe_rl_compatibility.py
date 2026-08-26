from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer none",
        },
    )
    try:
        with urlopen(request, timeout=180) as response:
            return {"status": response.status, "body": json.load(response)}
    except HTTPError as exc:
        return {
            "status": exc.code,
            "body": exc.read().decode("utf-8", errors="replace"),
        }


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8899/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect",
                "description": "Inspect one property.",
                "parameters": {
                    "type": "object",
                    "properties": {"property": {"type": "string"}},
                    "required": ["property"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    completion = _request(
        f"{base_url}/chat/completions",
        {
            "model": args.model,
            "temperature": 0,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "Call inspect for color."}],
            "tools": tools,
            "tool_choice": "required",
            "logprobs": True,
            "top_logprobs": 20,
            "prompt_logprobs": 20,
            "return_token_ids": True,
        },
    )
    body = completion.get("body")
    choice = (
        (body.get("choices") or [{}])[0]
        if isinstance(body, dict)
        else {}
    )
    message = choice.get("message") or {}
    gen_tokens = message.get("gen_tokens") or choice.get("token_ids") or []
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        logprobs = logprobs.get("content") or logprobs.get("token_logprobs")
    prompt_logprobs = (
        body.get("prompt_logprobs")
        if isinstance(body, dict)
        else None
    )
    if isinstance(prompt_logprobs, dict):
        prompt_logprobs = (
            prompt_logprobs.get("content")
            or prompt_logprobs.get("token_logprobs")
        )
    result = {
        "schema_version": "ifv-rl-serving-compatibility-v1",
        "python": sys.version,
        "packages": {
            name: _version(name)
            for name in (
                "torch",
                "transformers",
                "vllm",
                "rllm",
                "rllm-model-gateway",
                "verl",
            )
        },
        "endpoint": {
            "base_url": base_url,
            "model": args.model,
            "status": completion["status"],
        },
        "checks": {
            "native_tool_call": bool(message.get("tool_calls")),
            "prompt_token_ids": bool(
                body.get("prompt_token_ids")
                if isinstance(body, dict)
                else False
            ),
            "completion_token_ids": bool(gen_tokens),
            "completion_logprobs": bool(logprobs),
            "token_logprob_lengths_match": bool(gen_tokens)
            and bool(logprobs)
            and len(gen_tokens) == len(logprobs),
            "prompt_top20_logprobs": bool(prompt_logprobs),
        },
        "response_excerpt": body,
    }
    required = (
        "native_tool_call",
        "prompt_token_ids",
        "completion_token_ids",
        "completion_logprobs",
        "token_logprob_lengths_match",
        "prompt_top20_logprobs",
    )
    result["passed"] = completion["status"] == 200 and all(
        result["checks"][key] for key in required
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

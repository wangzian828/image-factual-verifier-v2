from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    ProxyHandler,
    Request,
    build_opener,
)

from jsonschema import Draft202012Validator


_LOCAL_OPENER = build_opener(ProxyHandler({}))


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _request(url: str, payload: dict[str, Any] | None = None) -> Any:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
    )
    try:
        with _LOCAL_OPENER.open(request, timeout=600) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:4000]}") from exc
    except URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc
    return json.loads(raw) if raw else None


def _completion(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = _request(f"{base_url}/chat/completions", payload)
    choices = result.get("choices") or []
    if len(choices) != 1:
        raise RuntimeError(f"expected one completion choice, got {len(choices)}")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("completion did not contain an assistant message")
    return {
        "message": message,
        "finish_reason": choices[0].get("finish_reason"),
        "usage": result.get("usage") or {},
    }


def _content(message: dict[str, Any], gate: str) -> str:
    value = message.get("content")
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{gate}: final content was empty")
    if "<think>" in value or "</think>" in value:
        raise RuntimeError(f"{gate}: reasoning leaked into final content")
    return value.strip()


def _reasoning(message: dict[str, Any]) -> str:
    value = message.get("reasoning") or message.get("reasoning_content")
    return value.strip() if isinstance(value, str) else ""


def _body(model: str, messages: list[dict[str, Any]], *, thinking: bool) -> dict[str, Any]:
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 4096,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if thinking:
        body.update(
            {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
                "thinking_token_budget": 1024,
            }
        )
    return body


def _schema_body(
    model: str,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    *,
    thinking: bool,
) -> dict[str, Any]:
    body = _body(model, messages, thinking=thinking)
    body["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": "probe_status", "strict": True, "schema": schema},
    }
    return body


def _structured_with_bounded_correction(
    base_url: str,
    model: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Exercise the production thinking->direct-schema recovery boundary.

    Qwen3.5 may finish a thinking+schema request with the candidate only in
    ``reasoning`` and no visible ``content``.  That is not source support and
    must not be accepted as structured output.  The runtime's bounded repair
    retries the same schema once with thinking disabled.
    """

    initial = _completion(
        base_url,
        _schema_body(
            model,
            [{"role": "user", "content": "Return status ready and count 1."}],
            schema,
            thinking=True,
        ),
    )
    initial_message = initial["message"]
    initial_content = initial_message.get("content")
    initial_reasoning = _reasoning(initial_message)
    initial_meta = {
        "finish_reason": initial["finish_reason"],
        "content_chars": len(initial_content.strip())
        if isinstance(initial_content, str)
        else 0,
        "reasoning_chars": len(initial_reasoning),
        "usage": initial["usage"],
    }

    if isinstance(initial_content, str) and initial_content.strip():
        parsed = json.loads(initial_content)
        Draft202012Validator(schema).validate(parsed)
        return {
            "output": parsed,
            "accepted_mode": "thinking_schema",
            "initial": initial_meta,
            "correction": None,
        }

    if not initial_reasoning:
        raise RuntimeError(
            "structured: empty content was not accompanied by separated reasoning"
        )

    correction = _completion(
        base_url,
        _schema_body(
            model,
            [{"role": "user", "content": "Return status ready and count 1."}],
            schema,
            thinking=False,
        ),
    )
    correction_content = _content(correction["message"], "structured correction")
    parsed = json.loads(correction_content)
    Draft202012Validator(schema).validate(parsed)
    return {
        "output": parsed,
        "accepted_mode": "direct_schema_correction",
        "initial": initial_meta,
        "correction": {
            "finish_reason": correction["finish_reason"],
            "content_chars": len(correction_content),
            "reasoning_chars": len(_reasoning(correction["message"])),
            "usage": correction["usage"],
        },
    }


def _origin(base_url: str) -> str:
    split = urlsplit(base_url)
    return f"{split.scheme}://{split.netloc}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.image.is_file():
        raise RuntimeError(f"probe image does not exist: {args.image}")
    if args.rounds < 1 or args.rounds > 8:
        raise RuntimeError("--rounds must be between 1 and 8")
    base_url = args.base_url.rstrip("/")
    origin = _origin(base_url)

    _request(f"{origin}/health")
    models = _request(f"{base_url}/models")
    cards = models.get("data") or []
    if not cards:
        raise RuntimeError("/v1/models returned no models")
    model = args.model or cards[0]["id"]
    card = next((item for item in cards if item.get("id") == model), None)
    if card is None:
        raise RuntimeError(f"served model is absent from /v1/models: {model}")
    if card.get("max_model_len") != args.expected_context:
        raise RuntimeError(
            f"expected max_model_len={args.expected_context}, got {card.get('max_model_len')}"
        )
    tokenizer_info = _request(f"{origin}/tokenizer_info")
    template = tokenizer_info.get("chat_template") or ""
    if "enable_thinking" not in template:
        raise RuntimeError("active chat template lacks enable_thinking")

    direct = _completion(
        base_url,
        _body(
            model,
            [{"role": "user", "content": "Reply with exactly READY."}],
            thinking=False,
        ),
    )
    direct_content = _content(direct["message"], "thinking_off")
    if _reasoning(direct["message"]):
        raise RuntimeError("thinking_off: reasoning was unexpectedly returned")

    thinking = _completion(
        base_url,
        _body(
            model,
            [{"role": "user", "content": "Work out 17 + 25, then give only the number."}],
            thinking=True,
        ),
    )
    thinking_content = _content(thinking["message"], "thinking_on")
    thinking_reasoning = _reasoning(thinking["message"])
    if not thinking_reasoning:
        raise RuntimeError("thinking_on: reasoning was not separated or was empty")

    image = _completion(
        base_url,
        _body(
            model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "State one directly visible property."},
                        {"type": "image_url", "image_url": {"url": _data_url(args.image)}},
                    ],
                }
            ],
            thinking=False,
        ),
    )
    image_content = _content(image["message"], "image")

    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ready"]},
            "count": {"type": "integer", "const": 1},
        },
        "required": ["status", "count"],
        "additionalProperties": False,
    }
    structured = _structured_with_bounded_correction(base_url, model, schema)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect",
                "description": "Inspect one requested property.",
                "parameters": {
                    "type": "object",
                    "properties": {"property": {"type": "string"}},
                    "required": ["property"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    round_results = []
    for index in range(args.rounds):
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Use the available tool."},
            {
                "role": "user",
                "content": json.dumps(
                    {"task": "inspect", "round": index + 1, "completed_rounds": index}
                ),
            },
        ]
        call_body = _body(model, messages, thinking=False)
        call_body.update({"tools": tools, "tool_choice": "required"})
        call = _completion(base_url, call_body)
        call_message = call["message"]
        if _reasoning(call_message):
            raise RuntimeError(f"tool round {index + 1}: direct mode returned reasoning")
        calls = call_message.get("tool_calls") or []
        if len(calls) != 1:
            raise RuntimeError(f"tool round {index + 1}: expected one call, got {len(calls)}")
        tool_call = calls[0]
        function = tool_call.get("function") or {}
        if not tool_call.get("id") or function.get("name") != "inspect":
            raise RuntimeError(f"tool round {index + 1}: malformed tool call {tool_call}")
        json.loads(function.get("arguments") or "{}")
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": call_message.get("content"),
                    "tool_calls": calls,
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps({"status": "success", "round": index + 1}),
                },
                {"role": "user", "content": "Confirm the tool result in one sentence."},
            ]
        )
        continuation_body = _body(model, messages, thinking=False)
        continuation_body.update({"tools": tools, "tool_choice": "none"})
        continuation = _completion(base_url, continuation_body)
        continuation_content = _content(
            continuation["message"], f"continuation round {index + 1}"
        )
        round_results.append(
            {
                "round": index + 1,
                "continuation": continuation_content,
                "call_usage": call["usage"],
                "continuation_usage": continuation["usage"],
            }
        )

    return {
        "schema_version": "ifv-vllm-qwen35-endpoint-gates-v1",
        "created_at_epoch": int(time.time()),
        "passed": True,
        "model": model,
        "max_model_len": card["max_model_len"],
        "tokenizer_class": tokenizer_info.get("tokenizer_class"),
        "gates": {
            "health": True,
            "models": True,
            "context_configuration": True,
            "thinking_off": {"content": direct_content, "usage": direct["usage"]},
            "thinking_on": {
                "content": thinking_content,
                "reasoning_chars": len(thinking_reasoning),
                "usage": thinking["usage"],
            },
            "image": {"content": image_content, "usage": image["usage"]},
            "reasoning_then_schema": structured,
            "independent_tool_roundtrips": round_results,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8901/v1")
    parser.add_argument("--model", default="ifv-qwen3.5-9b")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--expected-context", type=int, default=131072)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

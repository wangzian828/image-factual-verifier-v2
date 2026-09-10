"""Verify IFV Qwen/ms-swift rows with the real processor and train template.

This is intentionally a runtime probe rather than a JSON-only validator. It
checks that supervised thought tokens survive template encoding, tool calls and
observations are present in the encoded conversation, and multimodal rows keep
their images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _template_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_length": args.max_context,
        "truncation_strategy": args.truncation_strategy,
        "max_pixels": args.max_pixels,
        "padding_free": args.padding_free,
        "sequence_parallel_size": args.sequence_parallel_size,
        "loss_scale": args.loss_scale,
        "enable_thinking": args.enable_thinking,
        "add_non_thinking_prefix": args.add_non_thinking_prefix,
    }


def _rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{index + 1} is not an object")
        yield index, value


def _as_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        value = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        value = tokenizer.encode(text)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def _contains_subsequence(sequence: list[int], needle: list[int]) -> bool:
    if not needle or len(needle) > len(sequence):
        return False
    width = len(needle)
    return any(sequence[index : index + width] == needle for index in range(len(sequence) - width + 1))


def _contains_supervised_subsequence(
    input_ids: list[int],
    labels: list[int],
    needle: list[int],
) -> bool:
    if not needle or len(needle) > len(input_ids):
        return False
    width = len(needle)
    for index in range(len(input_ids) - width + 1):
        if input_ids[index : index + width] == needle and all(
            labels[index + offset] != -100 for offset in range(width)
        ):
            return True
    return False


def _text_for_presence_check(content: str) -> str:
    # In multimodal rows ms-swift consumes each <image> marker and replaces it
    # with image tokens. Check the surrounding public text separately.
    return content.replace("<image>", " ").strip()


def _tool_call_contract(content: str) -> tuple[str, list[str]]:
    """Return the function name and public parameter names for one call."""

    payload = json.loads(content)
    if not isinstance(payload, Mapping):
        raise ValueError("tool_call content must be a JSON object")
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError("tool_call content has no function name")
    raw_arguments = payload.get("arguments", "{}")
    if not isinstance(raw_arguments, str):
        raise ValueError("tool_call arguments must be a JSON string")
    arguments = json.loads(raw_arguments)
    if not isinstance(arguments, Mapping):
        raise ValueError("tool_call arguments must decode to an object")
    return name, [str(key) for key in arguments]


def _verify_rendered_tool_calls(
    messages: list[Mapping[str, Any]],
    decoded_text: str,
) -> int:
    """Verify calls after ms-swift renders JSON into Qwen function XML.

    The source dataset stores ``tool_call`` content as JSON, while the Qwen
    template serializes it as ``<function=...>`` and ``<parameter=...>``.
    Searching for the source JSON verbatim therefore produces false failures.
    Function tags are not used by the tool-schema preamble, so their counts
    prove that the actual calls reached the encoded conversation.
    """

    expected_functions: Counter[str] = Counter()
    expected_parameters: Counter[tuple[str, str]] = Counter()
    for message in messages:
        if str(message.get("role", "")) != "tool_call":
            continue
        name, parameter_names = _tool_call_contract(
            str(message.get("content", ""))
        )
        expected_functions[name] += 1
        for parameter_name in parameter_names:
            expected_parameters[(name, parameter_name)] += 1

    for name, count in expected_functions.items():
        rendered = len(
            re.findall(
                rf"<function={re.escape(name)}>",
                decoded_text,
            )
        )
        if rendered < count:
            raise ValueError(
                f"encoded input preserves only {rendered}/{count} "
                f"calls to {name}"
            )

    for (name, parameter_name), count in expected_parameters.items():
        function_blocks = re.findall(
            rf"<function={re.escape(name)}>(.*?)</function>",
            decoded_text,
            flags=re.DOTALL,
        )
        rendered = sum(
            block.count(f"<parameter={parameter_name}>")
            for block in function_blocks
        )
        if rendered < count:
            raise ValueError(
                f"encoded input preserves only {rendered}/{count} "
                f"{name}.{parameter_name} parameters"
            )
    return sum(expected_functions.values())


def _validate_roles(messages: Any, *, kind: str) -> list[str]:
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("messages must contain at least system, user, assistant")
    roles = [str(item.get("role", "")) for item in messages if isinstance(item, Mapping)]
    if len(roles) != len(messages):
        raise ValueError("every message must be an object")
    if roles[:2] != ["system", "user"] or roles[-1] != "assistant":
        raise ValueError("rows must start system/user and end assistant")
    if "user" in roles[2:]:
        raise ValueError("rows cannot contain a later user message")
    index = 2
    while index < len(roles):
        if roles[index] == "assistant":
            if index == len(roles) - 1:
                break
            if roles[index + 1] != "tool_call":
                raise ValueError("non-final assistant must be followed by tool_call")
            index += 1
            continue
        if roles[index] == "tool_call":
            if index + 1 >= len(roles) or roles[index + 1] != "tool_response":
                raise ValueError("tool_call must be followed by tool_response")
            index += 2
            continue
        if roles[index] == "tool_response":
            raise ValueError("tool_response must follow tool_call")
        raise ValueError(f"message {index} has invalid turn role {roles[index]!r}")
    if kind == "policy" and not any(
        "<think>" in str(item.get("content", ""))
        for item in messages
        if isinstance(item, Mapping) and item.get("role") == "assistant"
    ):
        raise ValueError("policy row has no native <think> target")
    return roles


def _row_kind(path: Path) -> str:
    return "perception" if "perception" in path.name.lower() else "policy"


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p95": 0, "max": 0, "mean": 0.0}
    ordered = sorted(values)
    p50 = ordered[max(0, (len(ordered) + 1) // 2 - 1)]
    p95 = ordered[max(0, (len(ordered) * 95 + 99) // 100 - 1)]
    return {
        "count": len(values),
        "min": min(values),
        "p50": p50,
        "p95": p95,
        "max": max(values),
        "mean": statistics.mean(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode IFV agent/perception rows with the real ms-swift processor."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--perception-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-context", type=int, default=131072)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument(
        "--truncation-strategy",
        choices=("raise", "left", "right", "split"),
        default="raise",
    )
    parser.add_argument("--padding-free", type=_boolean, default=False)
    parser.add_argument("--sequence-parallel-size", type=int, default=1)
    parser.add_argument("--loss-scale", default="ignore_empty_think")
    parser.add_argument("--enable-thinking", type=_boolean, default=False)
    parser.add_argument(
        "--add-non-thinking-prefix",
        type=_boolean,
        default=False,
    )
    parser.add_argument("--image-max-token-num", type=int, default=1024)
    args = parser.parse_args()
    if args.max_context < 1:
        parser.error("--max-context must be positive")
    if args.max_pixels is not None and args.max_pixels < 1:
        parser.error("--max-pixels must be positive")
    if args.sequence_parallel_size < 1:
        parser.error("--sequence-parallel-size must be positive")
    if args.image_max_token_num < 1:
        parser.error("--image-max-token-num must be positive")

    os.environ["IMAGE_MAX_TOKEN_NUM"] = str(args.image_max_token_num)

    from swift import get_processor, get_template

    processor = get_processor(args.model)
    template_contract = _template_kwargs(args)
    template = get_template(processor, **template_contract)
    template.set_mode("train")
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("ms-swift processor does not expose a tokenizer")

    datasets: list[tuple[str, Path]] = []
    for split in ("train", "validation", "test"):
        policy_path = args.policy_dir / f"{split}.jsonl"
        if policy_path.is_file():
            datasets.append(("policy", policy_path))
        if args.perception_dir is not None:
            perception_path = args.perception_dir / f"{split}.jsonl"
            if perception_path.is_file():
                datasets.append(("perception", perception_path))

    errors: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    lengths: list[int] = []
    trainable_lengths: list[int] = []
    by_kind: dict[str, list[int]] = {"policy": [], "perception": []}
    checks = Counter()
    longest: list[dict[str, Any]] = []
    dataset_files: list[dict[str, Any]] = []

    for kind, path in datasets:
        dataset_files.append({"kind": kind, **_file_record(path)})
        for row_index, row in _rows(path):
            location = f"{path.name}[{row_index}]"
            try:
                messages = row.get("messages")
                roles = _validate_roles(messages, kind=kind)
                if not isinstance(row.get("images"), list):
                    raise ValueError("images must be a list, including for text-only rows")
                encoded = template.encode(row, return_template_inputs=True)
                input_ids = _as_list(encoded["input_ids"])
                labels = _as_list(encoded["labels"])
                decoded_text = tokenizer.decode(
                    input_ids,
                    skip_special_tokens=False,
                )
                if len(input_ids) != len(labels):
                    raise ValueError("input_ids and labels have different lengths")
                if not input_ids:
                    raise ValueError("processor returned empty input_ids")
                if len(input_ids) > args.max_context:
                    raise ValueError(
                        f"encoded length {len(input_ids)} exceeds {args.max_context}"
                    )
                loss_positions = [index for index, value in enumerate(labels) if value != -100]
                if not loss_positions:
                    raise ValueError("row has no trainable assistant tokens")

                template_inputs = encoded.get("template_inputs")
                encoded_images = list(getattr(template_inputs, "images", []) or [])
                expected_images = list(row.get("images") or [])
                if expected_images and not encoded_images:
                    raise ValueError("processor dropped the row images")
                if not expected_images and encoded_images:
                    raise ValueError("text-only row unexpectedly gained images")
                checks["rows_encoded"] += 1
                if kind == "policy":
                    checks["tool_call_preserved"] += _verify_rendered_tool_calls(
                        messages,
                        decoded_text,
                    )

                for message in messages:
                    if not isinstance(message, Mapping):
                        continue
                    role = str(message.get("role", ""))
                    content = str(message.get("content", ""))
                    if role == "assistant" and "<think>" in content:
                        marker_ids = _encode(tokenizer, "<think>")
                        if not _contains_supervised_subsequence(input_ids, labels, marker_ids):
                            raise ValueError("native <think> marker is not supervised")
                        checks["thought_targets"] += 1
                    if role == "tool_response":
                        presence_text = _text_for_presence_check(content)
                        content_ids = _encode(tokenizer, presence_text)
                        if content_ids and not _contains_subsequence(
                            input_ids, content_ids
                        ):
                            raise ValueError(f"{role} content is absent from encoded input")
                        checks[f"{role}_preserved"] += 1

                counts[f"{kind}:{path.stem}"] += 1
                lengths.append(len(input_ids))
                trainable_lengths.append(len(loss_positions))
                by_kind.setdefault(kind, []).append(len(input_ids))
                longest.append(
                    {
                        "kind": kind,
                        "path": path.name,
                        "row_index": row_index,
                        "input_tokens": len(input_ids),
                        "trainable_tokens": len(loss_positions),
                        "message_count": len(messages),
                        "tool_call_count": roles.count("tool_call"),
                        "image_count": len(encoded_images),
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "kind": kind,
                        "path": path.name,
                        "row_index": row_index,
                        "error": str(exc),
                    }
                )

    report = {
        "schema_version": "ifv-ms-swift-agent-processor-verification-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(Path(args.model).expanduser().resolve()),
        "processor_class": type(processor).__name__,
        "template_class": type(template).__name__,
        "template_contract": {
            **template_contract,
            "image_max_token_num": args.image_max_token_num,
        },
        "max_context": args.max_context,
        "dataset_files": dataset_files,
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "counts": dict(sorted(counts.items())),
        "checks": dict(sorted(checks.items())),
        "input_tokens": _distribution(lengths),
        "trainable_tokens": _distribution(trainable_lengths),
        "input_tokens_by_kind": {
            kind: _distribution(values) for kind, values in sorted(by_kind.items())
        },
        "longest_rows": sorted(
            longest,
            key=lambda item: int(item["input_tokens"]),
            reverse=True,
        )[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

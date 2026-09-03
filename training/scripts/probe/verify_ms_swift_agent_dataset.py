"""Verify IFV Qwen/ms-swift rows with the real processor and train template.

This is intentionally a runtime probe rather than a JSON-only validator. It
checks that supervised thought tokens survive template encoding, tool calls and
observations are present in the encoded conversation, and multimodal rows keep
their images.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


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
    args = parser.parse_args()

    from swift import get_processor, get_template

    processor = get_processor(args.model)
    template = get_template(processor, loss_scale="default+ignore_empty_think")
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

    for kind, path in datasets:
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
                    if role in {"tool_call", "tool_response"}:
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
        "schema_version": "ifv-ms-swift-agent-processor-verification-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(Path(args.model).expanduser().resolve()),
        "processor_class": type(processor).__name__,
        "template_class": type(template).__name__,
        "max_context": args.max_context,
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

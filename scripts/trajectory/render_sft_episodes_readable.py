"""Render one-row-per-episode SFT JSONL into readable audit files.

The canonical JSONL is copied byte-for-byte. The Markdown transcript supports
both current Qwen/ms-swift roles and historical exporter rows.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping


def _json(value: Any, *, limit: int | None = None) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    if limit is not None and len(text) > limit:
        return text[: max(0, limit - 20)] + "\n... [truncated]\n"
    return text


def _safe_name(value: str, *, limit: int = 100) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text[:limit].rstrip("._-") or "episode"


def _extract_json_object(content: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _extract_thought(content: str) -> str:
    match = re.search(r"<think>\s*(.*?)\s*</think>", content, re.I | re.S)
    return match.group(1).strip() if match else ""


def _tool_call_from_content(content: str) -> tuple[str, dict[str, Any]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, Mapping) and str(payload.get("name", "")).strip():
        name = str(payload["name"]).strip()
        raw_arguments = payload.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = raw_arguments
        else:
            arguments = raw_arguments
        return name, arguments if isinstance(arguments, dict) else {"value": arguments}

    match = re.search(
        r"<function=([^>\s]+)>(.*?)</function>",
        content,
        re.I | re.S,
    )
    if match:
        arguments: dict[str, Any] = {}
        for parameter in re.finditer(
            r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
            match.group(2),
            re.I | re.S,
        ):
            raw_value = parameter.group(2).strip()
            try:
                value = json.loads(raw_value)
            except json.JSONDecodeError:
                value = raw_value
            arguments[parameter.group(1).strip()] = value
        return match.group(1).strip(), arguments
    return "", {}


def _render_initial_user(content: str) -> list[str]:
    payload = _extract_json_object(content)
    if payload is None:
        return ["### Episode input", "", content.strip()]
    output = [
        "### Episode input",
        "",
        f"- Case ID: `{payload.get('case_id', '')}`",
        f"- Image SHA-256: `{payload.get('image_sha256', '')}`",
        f"- Input mode: `{payload.get('input_mode', '')}`",
    ]
    observations = payload.get("initial_observations")
    if isinstance(observations, list):
        output.extend(["", "### Initial observations", ""])
        for item in observations:
            if not isinstance(item, Mapping):
                continue
            result = item.get("result")
            result = result if isinstance(result, Mapping) else {}
            tool = item.get("tool", "unknown")
            output.append(f"- `{tool}` status: `{result.get('status', 'unknown')}`")
            for key in ("scene_description", "full_text", "text_regions", "entities"):
                value = result.get(key)
                if value not in (None, "", [], {}):
                    output.extend(
                        [f"  - {key}", "", f"    ```json\n{_json(value)}\n    ```"]
                    )
    return output


def _render_stage_control(content: str) -> list[str]:
    payload = _extract_json_object(content)
    if payload is None:
        return ["### Runtime handoff", "", content.strip()]
    output = [
        "### Runtime handoff",
        "",
        f"- Stage: `{payload.get('stage', '')}`",
        f"- Output mode: `{payload.get('output_mode', '')}`",
    ]
    tools = payload.get("authorized_tool_names")
    if isinstance(tools, list):
        output.append("- Allowed tools: " + ", ".join(f"`{item}`" for item in tools))
    input_payload = payload.get("input_payload")
    if isinstance(input_payload, str):
        try:
            input_payload = json.loads(input_payload)
        except json.JSONDecodeError:
            pass
    if isinstance(input_payload, Mapping):
        for key in (
            "phase",
            "action_count",
            "stop_reason",
            "proposed_verdict",
            "target_facts",
            "open_gaps",
            "recent_observations",
            "recent_state_deltas",
        ):
            value = input_payload.get(key)
            if value not in (None, "", [], {}):
                output.extend(
                    ["", f"**{key}**", "", f"```json\n{_json(value, limit=6000)}\n```"]
                )
    return output


def _render_tool_response(content: str, tool_name: str = "") -> list[str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return [f"### Tool result: `{tool_name or 'unknown'}`", "", content.strip()]

    result: Any = payload
    output = [f"### Tool result: `{tool_name or 'unknown'}`", ""]
    if isinstance(payload, Mapping) and "result" in payload:
        result = payload.get("result")
        output[0] = f"### Tool result: `{tool_name or payload.get('tool', 'unknown')}`"

    output.extend(["**Observed result**", ""])
    if isinstance(result, Mapping):
        for key, value in result.items():
            if value not in (None, "", [], {}):
                output.extend(
                    [f"**{key}**", "", f"```json\n{_json(value, limit=10000)}\n```"]
                )
    elif result not in (None, ""):
        output.append(str(result))
    return output


def _render_message(index: int, message: Mapping[str, Any]) -> list[str]:
    role = str(message.get("role", "unknown"))
    content = str(message.get("content", ""))
    output = [f"## Message {index} — `{role}`", ""]

    if role == "assistant":
        thought = _extract_thought(content)
        tool_name, arguments = _tool_call_from_content(content)
        if thought:
            output.extend(["**Thought**", "", thought, ""])
        if tool_name:
            output.extend([f"**Tool call:** `{tool_name}`", ""])
            if arguments:
                output.extend(
                    ["**Arguments**", "", f"```json\n{_json(arguments, limit=8000)}\n```", ""]
                )
        if not thought and not tool_name:
            output.extend(
                ["```json", _json(_extract_json_object(content) or content, limit=12000), "```", ""]
            )
        return output

    if role == "tool_call":
        tool_name, arguments = _tool_call_from_content(content)
        output.extend([f"**Tool call:** `{tool_name or 'unknown'}`", ""])
        if arguments:
            output.extend(
                ["**Arguments**", "", f"```json\n{_json(arguments, limit=8000)}\n```", ""]
            )
        return output

    if role in {"tool", "tool_response"}:
        return output + _render_tool_response(content)

    if role == "user":
        if content.lstrip().startswith("<image>") or "Image factual verification episode." in content:
            return output + _render_initial_user(content)
        if '"authorized_tool_names"' in content or '"stage"' in content:
            return output + _render_stage_control(content)
        return output + [content.strip()]

    return output + [content.strip()]


def render_episode(row: Mapping[str, Any], destination: Path, index: int) -> None:
    episode_id = str(row.get("episode_id", "")).strip()
    case_id = str(row.get("case_id", "")).strip()
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"episode {episode_id!r} has no messages list")

    episode_dir = destination / f"{index:04d}-{_safe_name(case_id)}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "trajectory.json").write_text(
        _json(row) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Episode {index:04d}",
        "",
        f"- Case ID: `{case_id}`",
        f"- Episode ID: `{episode_id}`",
        f"- Source run: `{row.get('source_run_id', '')}`",
        f"- Trajectory version: `{row.get('trajectory_version', '')}`",
        f"- Message count: `{row.get('message_count', len(messages))}`",
        f"- Tool-call count: `{row.get('tool_call_count', '')}`",
        f"- Token estimate: `{row.get('token_count_estimate', '')}`",
        "",
        "This Markdown file is a human-audit view. The adjacent "
        "`trajectory.json` is the complete exported row; neither file replaces "
        "the canonical training JSONL.",
        "",
    ]
    for message_index, message in enumerate(messages, 1):
        if isinstance(message, Mapping):
            lines.extend(_render_message(message_index, message))
    (episode_dir / "episode.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"trajectory JSONL does not exist: {input_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_dir / "trajectory_sft.jsonl")

    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("trajectory JSONL row is not an object")
    for index, row in enumerate(rows, 1):
        render_episode(row, output_dir / "episodes", index)

    copied: list[str] = []
    for name in (
        "manifest.json",
        "selection-manifest.jsonl",
        "source-30-selection-manifest.jsonl",
    ):
        source = input_path.parent / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
            copied.append(name)
    readme = [
        "# Readable Agent SFT Episodes",
        "",
        f"- Source: `{input_path}`",
        f"- Episode count: `{len(rows)}`",
        "",
        "`trajectory_sft.jsonl` is the canonical export; one JSON line is one complete episode.",
        "`episodes/*/episode.md` is only a human-readable audit view.",
        "The renderer preserves model thought, tool arguments, and tool observations verbatim.",
        "It supports current `tool_call`/`tool_response` rows and historical rows.",
        "",
        "Copied metadata: " + (", ".join(f"`{name}`" for name in copied) or "none"),
    ]
    (output_dir / "README.md").write_text(
        "\n".join(readme) + "\n",
        encoding="utf-8",
    )
    print(f"rendered {len(rows)} episodes to {output_dir}")


if __name__ == "__main__":
    main()

"""Render one-row-per-episode SFT JSONL into readable per-episode files.

The canonical JSONL remains unchanged. This creates a human-audit view with
one directory per episode, a compact Markdown transcript, and the complete
episode JSON data beside it.
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
    return (text[:limit].rstrip("._-") or "episode")


def _extract_json_after_text(content: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


def _extract_thought(content: str) -> str:
    match = re.search(r"<think>\s*(.*?)\s*</think>", content, re.S | re.I)
    if match:
        return match.group(1).strip()
    return ""


def _extract_tool_call(content: str) -> str:
    patterns = (
        r"<function>\s*([^<\s]+)\s*</function>",
        r"<function=([^>\s]+)>",
        r'"name"\s*:\s*"([^"]+)"',
    )
    for pattern in patterns:
        match = re.search(pattern, content, re.I)
        if match:
            return match.group(1).strip()
    return ""


def _extract_tool_call_arguments(content: str) -> dict[str, Any]:
    """Parse the Qwen tool-call parameter blocks used by the exporter."""

    arguments: dict[str, Any] = {}
    pattern = re.compile(
        r"<parameter=(?P<name>[^>]+)>\s*(?P<value>.*?)\s*</parameter>",
        re.S | re.I,
    )
    for match in pattern.finditer(content):
        name = match.group("name").strip()
        value = match.group("value").strip()
        try:
            parsed: Any = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        arguments[name] = parsed
    return arguments


def _render_initial_user(content: str) -> list[str]:
    lines = content.splitlines()
    payload = _extract_json_after_text(content)
    if payload is None:
        return [content.strip()]
    output = [
        "### Episode input",
        "",
        f"- Case ID: `{payload.get('case_id', '')}`",
        f"- Image SHA-256: `{payload.get('image_sha256', '')}`",
        f"- Input mode: `{payload.get('input_mode', '')}`",
        "",
        "Initial observations:",
    ]
    observations = payload.get("initial_observations")
    if isinstance(observations, list):
        for item in observations:
            if not isinstance(item, Mapping):
                continue
            result = item.get("result")
            result = result if isinstance(result, Mapping) else {}
            tool = item.get("tool", "unknown")
            status = result.get("status", "unknown")
            output.append(f"- `{tool}` — status `{status}`")
            for key in ("scene_description", "full_text", "goal", "expected_property"):
                value = result.get(key)
                if value:
                    output.append(f"  - {key}: {value}")
    return output


def _render_stage_control(content: str) -> list[str]:
    payload = _extract_json_after_text(content)
    if payload is None:
        return [content.strip()]
    output = [
        "### Runtime handoff",
        "",
        f"- Stage: `{payload.get('stage', '')}`",
        f"- Output mode: `{payload.get('output_mode', '')}`",
    ]
    tools = payload.get("authorized_tool_names")
    if isinstance(tools, list):
        output.append(f"- Allowed tools: {', '.join(f'`{item}`' for item in tools)}")
    input_payload = payload.get("input_payload")
    if isinstance(input_payload, str):
        try:
            parsed = json.loads(input_payload)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
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
                value = parsed.get(key)
                if value not in (None, "", [], {}):
                    output.extend(["", f"**{key}**", "", f"```json\n{_json(value, limit=6000)}\n```"])
    return output


def _render_tool(content: str, tool_name: str) -> list[str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return [f"### Tool result: `{tool_name or 'unknown'}`", "", content.strip()]
    result = payload.get("result")
    update = payload.get("investigation_state_update")
    state_update = update.get("state_update") if isinstance(update, Mapping) else None
    output = [
        f"### Tool result: `{tool_name or payload.get('tool', 'unknown')}`",
        "",
        f"- Accepted: `{(state_update or {}).get('accepted', payload.get('accepted', ''))}`",
        f"- Function call ID: `{payload.get('function_call_id', '')}`",
    ]
    if isinstance(state_update, Mapping):
        for key in (
            "phase",
            "action_count",
            "completed_tools",
            "next_available_tools",
            "created_visual_fact_ids",
            "created_discovery_ids",
            "created_evidence_ids",
            "created_finding_ids",
            "rejected_reason",
        ):
            value = state_update.get(key)
            if value not in (None, "", [], {}):
                output.append(f"- {key}: `{_json(value) if isinstance(value, (list, dict)) else value}`")
    if isinstance(result, Mapping):
        output.extend(["", "**Observed result**", ""])
        for key in (
            "status",
            "scene_description",
            "full_text",
            "text_regions",
            "entities",
            "candidate_count",
            "items",
            "evidence",
            "findings",
            "error",
            "failure_kind",
            "message",
        ):
            value = result.get(key)
            if value not in (None, "", [], {}):
                output.extend([f"**{key}**", "", f"```json\n{_json(value, limit=10000)}\n```"])
    return output


def _render_message(index: int, message: Mapping[str, Any]) -> list[str]:
    role = str(message.get("role", "unknown"))
    content = str(message.get("content", ""))
    output = [f"## Turn message {index} — `{role}`", ""]
    if role == "assistant":
        thought = _extract_thought(content)
        tool_name = _extract_tool_call(content)
        if thought:
            output.extend(["**Thought**", "", thought, ""])
        if tool_name:
            output.extend([f"**Tool call:** `{tool_name}`", ""])
            arguments = _extract_tool_call_arguments(content)
            if arguments:
                output.extend(
                    [
                        "**Arguments**",
                        "",
                        f"```json\n{_json(arguments, limit=8000)}\n```",
                        "",
                    ]
                )
        if not thought and not tool_name:
            output.extend(["```json", _json(_extract_json_after_text(content) or content, limit=12000), "```", ""])
        return output
    if role == "tool":
        return output + _render_tool(content, "")
    if role == "user":
        if content.lstrip().startswith("Image factual verification episode."):
            return output + _render_initial_user(content)
        if "authorized_tool_names" in content or '"stage"' in content:
            return output + _render_stage_control(content)
        return output + _render_initial_user(content)
    return output + [content.strip()]


def render_episode(row: Mapping[str, Any], destination: Path, index: int) -> None:
    episode_id = str(row.get("episode_id", "")).strip()
    case_id = str(row.get("case_id", "")).strip()
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"episode {episode_id!r} has no messages list")
    episode_dir = destination / f"{index:04d}-{_safe_name(case_id)}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    with (episode_dir / "trajectory.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(_json(row) + "\n")
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
        "The adjacent `trajectory.json` contains the same complete exported row. "
        "This Markdown file is a readable audit view and is not a training input.",
        "",
    ]
    for message_index, message in enumerate(messages, 1):
        if isinstance(message, Mapping):
            lines.extend(_render_message(message_index, message))
    with (episode_dir / "episode.md").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


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

    rows: list[dict[str, Any]] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("trajectory JSONL row is not an object")
            rows.append(value)
    for index, row in enumerate(rows, 1):
        render_episode(row, output_dir / "episodes", index)

    source_files = [
        input_path.parent / "manifest.json",
        input_path.parent / "selection-manifest.jsonl",
        input_path.parent / "source-30-selection-manifest.jsonl",
    ]
    copied = []
    for source in source_files:
        if source.is_file():
            shutil.copy2(source, output_dir / source.name)
            copied.append(source.name)
    readme = [
        "# Readable Agent SFT Episodes",
        "",
        f"- Source: `{input_path}`",
        f"- Episode count: `{len(rows)}`",
        "",
        "## Which file is the trajectory?",
        "",
        "`trajectory_sft.jsonl` is the canonical export: one JSON line is one complete episode.",
        "This directory adds a readable view only:",
        "",
        "- `trajectory_sft.jsonl`: a byte-for-byte copy of the canonical source JSONL;",
        "- `episodes/*/trajectory.json`: the complete one-episode JSON data in readable form;",
        "- `episodes/*/episode.md`: compact human-readable transcript;",
        "- copied `manifest*` files: indexes and metadata, not trajectories.",
        "",
        "The Markdown view omits repeated runtime schemas and large state deltas for readability.",
        "It must not be used as a replacement for the canonical training JSONL.",
        "",
        "Copied metadata: " + ", ".join(f"`{name}`" for name in copied),
    ]
    with (output_dir / "README.md").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("\n".join(readme) + "\n")
    print(f"rendered {len(rows)} episodes to {output_dir}")


if __name__ == "__main__":
    main()

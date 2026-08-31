"""Render exported SFT episodes into a Chinese human-audit view.

The canonical trajectory JSONL is copied without modification.  Only the
readable Markdown layer is localized: labels, headings, round descriptions,
tool/state/parameter/observation labels are Chinese, while model-generated
thoughts, structured outputs, tool arguments, and tool observations remain
verbatim.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping


FIELD_LABELS = {
    "case_id": "案例 ID",
    "episode_id": "轨迹 ID",
    "source_run_id": "来源运行",
    "trajectory_version": "轨迹版本",
    "message_count": "消息数",
    "tool_call_count": "工具调用数",
    "token_count_estimate": "Token 估计值",
    "input_mode": "输入模式",
    "initial_observations": "初始观察",
    "status": "状态",
    "scene_description": "场景描述",
    "full_text": "完整文字",
    "goal": "目标",
    "expected_property": "期望核对属性",
    "text_regions": "文字区域",
    "entities": "实体",
    "phase": "阶段",
    "output_mode": "输出模式",
    "authorized_tool_names": "允许使用的工具",
    "action_count": "动作计数",
    "stop_reason": "停止原因",
    "proposed_verdict": "拟议判定",
    "target_facts": "目标事实",
    "open_gaps": "未闭合问题",
    "recent_observations": "最近观察",
    "recent_state_deltas": "最近状态变化",
    "accepted": "是否接受",
    "function_call_id": "函数调用 ID",
    "completed_tools": "已完成工具",
    "next_available_tools": "下一步可用工具",
    "created_visual_fact_ids": "新建视觉事实 ID",
    "created_discovery_ids": "新建检索发现 ID",
    "created_evidence_ids": "新建证据 ID",
    "created_finding_ids": "新建判断发现 ID",
    "rejected_reason": "拒绝原因",
    "result": "结果",
    "candidate_count": "候选数量",
    "items": "条目",
    "evidence": "证据",
    "findings": "发现",
    "error": "错误",
    "failure_kind": "失败类型",
    "message": "消息",
    "stage": "阶段",
    "input_payload": "阶段输入",
}

ROLE_LABELS = {
    "system": "系统",
    "user": "用户 / 运行时",
    "assistant": "模型",
    "tool": "工具",
}


def _json(value: Any, *, limit: int | None = None) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    if limit is not None and len(text) > limit:
        return text[: max(0, limit - 20)] + "\n... [已截断，仅用于阅读]\n"
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
    return match.group(1).strip() if match else ""


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


def _label(key: str) -> str:
    return FIELD_LABELS.get(key, key)


def _value_block(value: Any, *, limit: int = 10000) -> list[str]:
    if isinstance(value, str) and "\n" not in value and len(value) <= 500:
        return [f"`{value}`"]
    return [f"```json\n{_json(value, limit=limit)}\n```"]


def _append_field(output: list[str], key: str, value: Any, *, limit: int = 10000) -> None:
    output.extend(["", f"**{_label(key)}** (`{key}`)", ""])
    output.extend(_value_block(value, limit=limit))


def _render_initial_user(content: str) -> list[str]:
    payload = _extract_json_after_text(content)
    if payload is None:
        return ["### 原始运行消息", "", content.strip()]

    output = [
        "### 案例输入",
        "",
        f"- {_label('case_id')}：`{payload.get('case_id', '')}`",
        f"- Image SHA-256：`{payload.get('image_sha256', '')}`",
        f"- {_label('input_mode')}：`{payload.get('input_mode', '')}`",
        "",
        "#### 初始观察",
    ]
    observations = payload.get("initial_observations")
    if isinstance(observations, list):
        for item in observations:
            if not isinstance(item, Mapping):
                continue
            result = item.get("result")
            result = result if isinstance(result, Mapping) else {}
            tool = item.get("tool", "unknown")
            output.extend(
                [
                    "",
                    f"- 工具：`{tool}`；状态：`{result.get('status', 'unknown')}`",
                ]
            )
            for key in ("scene_description", "full_text", "goal", "expected_property"):
                value = result.get(key)
                if value:
                    output.extend([f"  - {_label(key)} (`{key}`)：{value}"])
    return output


def _render_stage_control(content: str) -> list[str]:
    payload = _extract_json_after_text(content)
    if payload is None:
        return ["### 原始运行交接消息", "", content.strip()]

    output = [
        "### 运行时交接",
        "",
        f"- {_label('stage')}：`{payload.get('stage', '')}`",
        f"- {_label('output_mode')}：`{payload.get('output_mode', '')}`",
    ]
    tools = payload.get("authorized_tool_names")
    if isinstance(tools, list):
        output.append(
            f"- {_label('authorized_tool_names')}："
            + ", ".join(f"`{item}`" for item in tools)
        )
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
                    _append_field(output, key, value, limit=6000)
    return output


def _render_tool(content: str, tool_name: str) -> list[str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return [
            f"### 工具结果：`{tool_name or 'unknown'}`",
            "",
            "#### 原始观察文本",
            "",
            content.strip(),
        ]

    result = payload.get("result")
    update = payload.get("investigation_state_update")
    state_update = update.get("state_update") if isinstance(update, Mapping) else None
    state_update = state_update if isinstance(state_update, Mapping) else {}
    actual_tool = tool_name or payload.get("tool", "unknown")
    output = [
        f"### 工具结果：`{actual_tool}`",
        "",
        f"- {_label('accepted')}：`{state_update.get('accepted', payload.get('accepted', ''))}`",
        f"- {_label('function_call_id')}：`{payload.get('function_call_id', '')}`",
    ]
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
            _append_field(output, key, value, limit=6000)

    if isinstance(result, Mapping):
        output.extend(["", "#### 工具观察", ""])
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
                _append_field(output, key, value)
    return output


def _render_message(index: int, message: Mapping[str, Any]) -> list[str]:
    role = str(message.get("role", "unknown"))
    content = str(message.get("content", ""))
    output = [f"## 第 {index} 轮消息（角色：{ROLE_LABELS.get(role, role)}）", ""]

    if role == "assistant":
        thought = _extract_thought(content)
        tool_name = _extract_tool_call(content)
        if thought:
            output.extend(
                [
                    "#### 模型原生 thought（原文，未翻译）",
                    "",
                    thought,
                    "",
                ]
            )
        if tool_name:
            output.extend([f"#### 工具调用：`{tool_name}`", ""])
            arguments = _extract_tool_call_arguments(content)
            if arguments:
                output.extend(
                    [
                        "#### 调用参数（原文结构，未改写）",
                        "",
                        f"```json\n{_json(arguments, limit=8000)}\n```",
                        "",
                    ]
                )
        if not thought and not tool_name:
            output.extend(
                [
                    "#### 模型结构化输出（原文，未翻译）",
                    "",
                    f"```json\n{_json(_extract_json_after_text(content) or content, limit=12000)}\n```",
                    "",
                ]
            )
        return output

    if role == "tool":
        return output + _render_tool(content, "")
    if role == "user":
        if content.lstrip().startswith("Image factual verification episode."):
            return output + _render_initial_user(content)
        if "authorized_tool_names" in content or '"stage"' in content:
            return output + _render_stage_control(content)
        return output + ["### 原始运行消息", "", content.strip()]
    return output + ["### 原始消息内容", "", content.strip()]


def render_episode(row: Mapping[str, Any], destination: Path, index: int) -> None:
    episode_id = str(row.get("episode_id", "")).strip()
    case_id = str(row.get("case_id", "")).strip()
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"episode {episode_id!r} has no messages list")

    episode_dir = destination / f"{index:04d}-{_safe_name(case_id)}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    with (episode_dir / "trajectory.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(_json(row) + "\n")

    lines = [
        f"# 轨迹 {index:04d}",
        "",
        f"- {_label('case_id')}：`{case_id}`",
        f"- {_label('episode_id')}：`{episode_id}`",
        f"- {_label('source_run_id')}：`{row.get('source_run_id', '')}`",
        f"- {_label('trajectory_version')}：`{row.get('trajectory_version', '')}`",
        f"- {_label('message_count')}：`{row.get('message_count', len(messages))}`",
        f"- {_label('tool_call_count')}：`{row.get('tool_call_count', '')}`",
        f"- {_label('token_count_estimate')}：`{row.get('token_count_estimate', '')}`",
        "",
        "本目录是中文人工阅读视图，不是新的训练输入。",
        "同目录下的 `trajectory.json` 保留该条完整导出数据；模型原生 thought、结构化输出、工具参数和工具观察中的原文均未翻译。",
        "",
    ]
    for message_index, message in enumerate(messages, 1):
        if isinstance(message, Mapping):
            lines.extend(_render_message(message_index, message))

    with (episode_dir / "episode.md").open("w", encoding="utf-8", newline="\n") as handle:
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

    # The canonical source is copied byte-for-byte; no source trajectory is edited.
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

    copied: list[str] = []
    for name in ("manifest.json", "selection-manifest.jsonl", "source-30-selection-manifest.jsonl"):
        source = input_path.parent / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
            copied.append(name)

    readme = [
        "# Agent 轨迹中文阅读目录",
        "",
        f"- 原始来源：`{input_path}`",
        f"- 轨迹数量：`{len(rows)}`",
        "",
        "## 文件说明",
        "",
        "- `trajectory_sft.jsonl`：原始 canonical 导出的字节级副本，一行一条完整轨迹；",
        "- `episodes/*/trajectory.json`：对应 episode 的完整 JSON 数据；",
        "- `episodes/*/episode.md`：中文人工阅读稿；",
        "- `manifest*`：索引和元数据，不是轨迹正文。",
        "",
        "## 原文保留规则",
        "",
        "中文只用于标题、轮次、角色、工具、状态、参数和观察字段的阅读标注。",
        "模型原生 thought、模型结构化输出、工具调用参数、工具返回内容和其中的英文文本均按原文保留。",
        "因此 `episode.md` 是阅读辅助视图，不应当被当作翻译后的原始轨迹或训练数据。",
        "",
        "已复制的元数据：" + ("、".join(f"`{name}`" for name in copied) if copied else "无"),
    ]
    with (output_dir / "README.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(readme) + "\n")

    print(f"已生成 {len(rows)} 条中文阅读轨迹：{output_dir}")


if __name__ == "__main__":
    main()

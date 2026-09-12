"""把一行一个 episode 的 SFT JSONL 渲染成人类可读的中文审阅目录。

只翻译阅读视图的标题和字段名；canonical JSONL、模型 thought、工具参数和工具观察
均按原文保留。
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
    "token_count_estimate": "Token 估算",
    "input_mode": "输入模式",
    "authorized_tool_names": "允许使用的工具",
    "accepted": "是否接受",
    "completed_tools": "已完成工具",
    "next_available_tools": "下一步可用工具",
    "created_visual_fact_ids": "新建视觉事实 ID",
    "created_discovery_ids": "新建检索发现 ID",
    "created_evidence_ids": "新建证据 ID",
    "created_finding_ids": "新建判断发现 ID",
    "rejected_reason": "拒绝原因",
}


def _json(value: Any, *, limit: int | None = None) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    if limit is not None and len(text) > limit:
        return text[: max(0, limit - 24)] + "\n... [已截断，仅供阅读]\n"
    return text


def _safe_name(value: str, *, limit: int = 100) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text[:limit].rstrip("._-") or "episode"


def _label(key: str) -> str:
    return FIELD_LABELS.get(key, key)


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

    match = re.search(r"<function=([^>\s]+)>(.*?)</function>", content, re.I | re.S)
    if not match:
        return "", {}
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


def _field(output: list[str], key: str, value: Any, *, limit: int = 10000) -> None:
    output.extend(["", f"**{_label(key)}** (`{key}`)", ""])
    if isinstance(value, str) and "\n" not in value and len(value) <= 500:
        output.append(f"`{value}`")
    else:
        output.append(f"```json\n{_json(value, limit=limit)}\n```")


def _render_initial_user(content: str) -> list[str]:
    payload = _extract_json_object(content)
    if payload is None:
        return ["### 案例输入", "", content.strip()]
    output = [
        "### 案例输入",
        "",
        f"- {_label('case_id')}：`{payload.get('case_id', '')}`",
        f"- 图片 SHA-256：`{payload.get('image_sha256', '')}`",
        f"- {_label('input_mode')}：`{payload.get('input_mode', '')}`",
    ]
    observations = payload.get("initial_observations")
    if isinstance(observations, list):
        output.extend(["", "### 初始观察", ""])
        for item in observations:
            if not isinstance(item, Mapping):
                continue
            result = item.get("result")
            result = result if isinstance(result, Mapping) else {}
            tool = item.get("tool", "unknown")
            output.append(f"- 工具：`{tool}`；状态：`{result.get('status', 'unknown')}`")
            for key in ("scene_description", "full_text", "text_regions", "entities"):
                value = result.get(key)
                if value not in (None, "", [], {}):
                    _field(output, key, value, limit=6000)
    return output


def _render_stage_control(content: str) -> list[str]:
    payload = _extract_json_object(content)
    if payload is None:
        return ["### 运行交接", "", content.strip()]
    output = [
        "### 运行交接",
        "",
        f"- {_label('phase')}：`{payload.get('stage', '')}`",
        f"- 输出模式：`{payload.get('output_mode', '')}`",
    ]
    tools = payload.get("authorized_tool_names")
    if isinstance(tools, list):
        output.append(
            f"- {_label('authorized_tool_names')}："
            + "、".join(f"`{item}`" for item in tools)
        )
    input_payload = payload.get("input_payload")
    if isinstance(input_payload, str):
        try:
            input_payload = json.loads(input_payload)
        except json.JSONDecodeError:
            pass
    if isinstance(input_payload, Mapping) and input_payload:
        _field(output, "input_payload", input_payload, limit=6000)
    return output


def _render_tool_response(content: str) -> list[str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return ["### 工具结果", "", content.strip()]

    result: Any = payload
    output = ["### 工具结果", ""]
    if isinstance(payload, Mapping) and "result" in payload:
        result = payload.get("result")
    output.extend(["", "#### 工具观察", ""])
    if isinstance(result, Mapping):
        for key, value in result.items():
            if value not in (None, "", [], {}):
                _field(output, key, value, limit=10000)
    elif result not in (None, ""):
        output.append(str(result))
    return output


def _render_message(index: int, message: Mapping[str, Any]) -> list[str]:
    role = str(message.get("role", "unknown"))
    content = str(message.get("content", ""))
    role_label = {
        "system": "系统",
        "user": "用户/运行时",
        "assistant": "模型",
        "tool_call": "工具调用",
        "tool_response": "工具结果",
        "tool": "工具结果（旧格式）",
    }.get(role, role)
    output = [f"## 第 {index} 条消息（{role_label}：`{role}`）", ""]

    if role == "assistant":
        thought = _extract_thought(content)
        tool_name, arguments = _tool_call_from_content(content)
        if thought:
            output.extend(["#### 模型原生 thought（原文保留）", "", thought, ""])
        if tool_name:
            output.extend([f"#### 工具调用：`{tool_name}`", ""])
            if arguments:
                _field(output, "arguments", arguments, limit=8000)
        if not thought and not tool_name:
            output.extend(
                [
                    "#### 模型结构化输出（原文保留）",
                    "",
                    f"```json\n{_json(_extract_json_object(content) or content, limit=12000)}\n```",
                    "",
                ]
            )
        return output

    if role == "tool_call":
        tool_name, arguments = _tool_call_from_content(content)
        output.extend([f"#### 工具调用：`{tool_name or 'unknown'}`", ""])
        if arguments:
            _field(output, "arguments", arguments, limit=8000)
        return output

    if role in {"tool", "tool_response"}:
        return output + _render_tool_response(content)

    if role == "user":
        if content.lstrip().startswith("<image>") or "Image factual verification episode." in content:
            return output + _render_initial_user(content)
        if '"authorized_tool_names"' in content or '"stage"' in content:
            return output + _render_stage_control(content)
        return output + ["### 原始运行消息", "", content.strip()]

    return output + ["### 原始内容", "", content.strip()]


def render_episode(row: Mapping[str, Any], destination: Path, index: int) -> None:
    episode_id = str(row.get("episode_id", "")).strip()
    case_id = str(row.get("case_id", "")).strip()
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"episode {episode_id!r} 没有 messages")

    episode_dir = destination / f"{index:04d}-{_safe_name(case_id)}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "trajectory.json").write_text(
        _json(row) + "\n",
        encoding="utf-8",
    )
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
        "本 Markdown 仅用于人工审阅；同目录的 `trajectory.json` 是完整导出行。"
        "canonical `trajectory_sft.jsonl` 未被修改，模型 thought、工具参数和观察原文均保留。",
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
        raise SystemExit(f"trajectory JSONL 不存在：{input_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"输出目录必须不存在或为空：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_dir / "trajectory_sft.jsonl")

    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("trajectory JSONL 行必须是对象")
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
        "# Agent 轨迹中文阅读目录",
        "",
        f"- 原始来源：`{input_path}`",
        f"- 轨迹数量：`{len(rows)}`",
        "",
        "`trajectory_sft.jsonl` 是 canonical 训练导出，一行对应一条完整轨迹。",
        "`episodes/*/episode.md` 只用于人工阅读；`trajectory.json` 保留完整单条数据。",
        "中文只用于标题和阅读标签；模型 thought、结构化输出、工具参数与工具观察均保留原文。",
        "",
        "复制的元数据：" + (", ".join(f"`{name}`" for name in copied) or "无"),
    ]
    (output_dir / "README.md").write_text(
        "\n".join(readme) + "\n",
        encoding="utf-8",
    )
    print(f"已生成 {len(rows)} 条中文阅读轨迹：{output_dir}")


if __name__ == "__main__":
    main()

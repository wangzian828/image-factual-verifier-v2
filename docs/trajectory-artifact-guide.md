# 轨迹产物说明

## 1. 哪个文件是真正的轨迹

| 文件 | 内容 | 是否用于训练 |
| --- | --- | --- |
| `traces/<episode>.json` | runtime 保存的 canonical trace，含完整状态和调用记录 | 不是直接训练输入 |
| `trajectory_sft.jsonl` | 一行一条完整 policy episode | 是，转换后用于 reasoning SFT |
| `action_only.jsonl` | 有可执行动作但没有可读 thought 的完整 episode | 不进入 reasoning SFT |
| `perception_trajectories.jsonl` | 独立图片观察样本 | 转换后用于 perception SFT |
| `manifest.json` | 数量、版本、来源和哈希 | 否 |
| `index.jsonl` / selection manifest | 行号与来源索引 | 否 |

最适合人工查看的是：

```text
readable-episodes/episodes/<编号>-<case>/episode.md
```

它只是阅读视图。完整单条 JSON 在旁边的 `trajectory.json`，canonical 训练来源仍是
`trajectory_sft.jsonl`。

## 2. policy 轨迹消息

```text
system
user
assistant: <think>...</think>
tool_call: {"name":"...","arguments":"{...}"}
tool_response: 完整公开工具结果
...
assistant: <think>...</think><answer>{...}</answer>
```

每个消息只有 `role` 和 `content`。图片通过顶层 `images` 提供，工具 schema 通过顶层
`tools` 提供。不要根据文件名猜测数据类型，以行内容和 manifest 为准。

工具结果不会只保留 `{"status":"success"}` 这种 transport 状态；导出器会保留模型可见
的完整公开结果。provider wire、内部 state、缓存和 private gold 不进入训练消息。
图片引用可以是本地路径或 `data:image/...;base64,...`；审计器两种都接受。

## 3. 中文阅读目录

```powershell
python scripts/trajectory/render_sft_episodes_zh_readable.py `
  --input <package>\trajectory_sft.jsonl `
  --output-dir <package>\readable-episodes-zh
```

生成的中文目录只翻译标题、轮次和字段标签。模型原生 thought、结构化输出、工具参数
及工具观察原文保留，不把阅读视图当作翻译后的训练数据。

## 4. 训练前检查

```powershell
cd training
python -m ifv_training audit --strict --input <ms-swift-policy>
python scripts/probe/verify_ms_swift_agent_dataset.py `
  --model <Qwen checkpoint> `
  --policy-dir <ms-swift-policy> `
  --output <processor-verification.json>
```

真实 processor 验证通过后，才可把 `ms-swift-policy` 交给训练。

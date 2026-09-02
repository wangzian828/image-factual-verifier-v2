# 轨迹产物说明

一条完整 Agent 轨迹通常同时有三层文件：

| 文件 | 作用 | 是否是轨迹 |
| --- | --- | --- |
| `traces/<episode>.json` | runtime 保存的 canonical trace，含完整状态和调用记录 | 是 |
| `trajectory_sft.jsonl` | Qwen SFT 导出；每一行是一个完整 episode | 是，训练用 |
| `action_only.jsonl` | 没有可读 thought、但动作可执行的独立导出 | 是，单独使用 |
| `perception_trajectories.jsonl` | 从 canonical trace 导出的独立视觉观察样本 | 是，单独训练 |
| `manifest.json`、`selection-manifest.jsonl` | 选择结果、哈希、数量、来源和统计 | 否 |

`trajectory_sft.jsonl` 为了可训练，会把消息、工具调用、观察和紧凑状态上下文放在同一
个 JSON 行里，因此肉眼直接打开很难读。这不代表它是多条轨迹拼在一起：一行就是一条
完整 episode。

其中，`text_search`、`text_image_search` 和 `reverse_image_search` 的候选会作为
未验证 Discovery 保留在轨迹中；只有后续检查形成的有效观察或正文片段才属于
Evidence。SFT judge 使用同一条轨迹的有序动作和工具观察做审计，不把搜索候选
直接升级为证据。

## 推荐阅读方式

```powershell
python scripts/trajectory/render_sft_episodes_readable.py `
  --input <package>\trajectory_sft.jsonl `
  --output-dir <package>\readable-episodes
```

生成结果：

- `readable-episodes/episodes/0001-*/episode.md`：人类可读的单条 transcript；
- `readable-episodes/episodes/0001-*/trajectory.json`：该条完整 JSON 数据的可读重排版；
- `readable-episodes/README.md`：文件关系和使用边界。

Markdown 版会隐藏重复的动态 schema 和大块 state delta，只用于审阅；训练和程序处理仍
使用原始 `trajectory_sft.jsonl`。源 JSONL 不会被这个脚本修改。

`action_only` 不会混入 `trajectory_sft.jsonl`，但不代表整条 case 被丢弃：
同一条 canonical trace 的 perception 结果仍可进入独立 perception SFT。阅读
package 时应分别查看 `accepted-dataset/action_only.jsonl`、
`accepted-dataset/perception.*.jsonl` 和 `ms-swift-perception/`。

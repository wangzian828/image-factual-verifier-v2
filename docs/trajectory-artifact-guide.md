# 轨迹产物说明

一条完整 Agent 轨迹通常同时有三层文件：

| 文件 | 作用 | 是否是轨迹 |
| --- | --- | --- |
| `traces/<episode>.json` | runtime 保存的 canonical trace，含完整状态和调用记录 | 是 |
| `trajectory_sft.jsonl` | Qwen SFT 导出；每一行是一个完整 episode | 是，训练用 |
| `manifest.json`、`selection-manifest.jsonl` | 选择结果、哈希、数量、来源和统计 | 否 |

`trajectory_sft.jsonl` 为了可训练，会把消息、工具调用、观察和紧凑状态上下文放在同一
个 JSON 行里，因此肉眼直接打开很难读。这不代表它是多条轨迹拼在一起：一行就是一条
完整 episode。

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

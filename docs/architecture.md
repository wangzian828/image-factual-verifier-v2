# 当前架构：`unified-react-v1`

## 主流程

```text
原图 + 固定任务
  → 紧凑上下文的 ReAct 请求
  → thought + 一个工具调用
  → 工具结果
  → reducer 写入 state delta
  → 下一轮 ReAct
  → finish 或预算结束
  → Judgment 输出 real/fake 与 fact-check report
```

视觉工具没有固定顺序，也不是隐藏的独立阶段。每轮请求都会重新附加受控压缩
原图；文本上下文只携带有限的视觉记忆、候选、证据、失败和近期动作。

## 职责边界

| 部件 | 负责 | 不负责 |
| --- | --- | --- |
| ReAct policy | thought、工具选择、公开参数 | 修改 state、生成 ID、伪造 Evidence |
| runtime adapter | 隐藏内部参数、注入图片和运行时上下文 | 改写成熟工具语义 |
| 成熟工具 | 感知、OCR、搜索、网页提取、图像比较、视觉复查 | 最终二分类 |
| reducer | 校验动作、去重、预算、失败分类、写入 state delta | 用规则替模型猜标签 |
| Judgment | 综合已记录上下文并写报告 | 新增工具调用或虚构来源 |

## 状态与证据

当前 state 使用 `objective`、`visual_memory`、`discoveries`、`evidence`、
`failures`、查询/URL 历史和预算字段。搜索结果、标题、摘要和反向搜图结果先
作为 Discovery；只有成功视觉观察、有效比较或已检查页面的具体片段才进入
Evidence。

当前主流程不创建 `target_facts`、`search_hypotheses`、`tasks` 或 Claim
ownership。旧图状态只用于 legacy trace 回放，不会被 active runtime 调用。

## 图片 API

- `direct_multimodal`：ReAct 和 Judgment 请求都带一份临时压缩原图。
- `separate_vlm`：图片只交给视觉工具，policy 使用结构化观察。

两种模式共用工具契约、reducer 和 trace 记录。压缩图片不进入累计文本历史，
trace 只保存哈希、尺寸和外部化媒体引用。

## 产物

- canonical trace：完整请求、thought、工具结果、状态增量和最终报告；
- SFT：一条完整 episode 一条 Qwen 对话，去掉重复 workspace；
- RL/reward：读取完整轨迹、视觉记忆、调查证据和动作历史，不重新创建 Claim 图。

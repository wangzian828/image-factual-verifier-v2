# Agent 上下文与 SFT 轨迹压缩

更新：2026-08-26。

> 历史记录：本页数据来自旧的多阶段 Claim/Task runtime。当前
> `unified-react-v1` 已改为单一 ReAct 状态；现行边界见
> [Agent 结构](agent-structure.md) 和
> [Agent Prompt 与运行时说明](agent-prompt-and-runtime-guide-zh.md)。

## 结论

两层都有重复，但严重程度不同：

- 运行时：部分阶段把阶段专属 `stage_input` 和完整 `workspace` 同时发送，属于单次请求内重复。
- SFT 导出：把每轮累计的完整阶段输入再次串进整条 episode，才是超出 128K 的主因。

原始 canonical trace、完整 handoff 和工具产物全部保留；压缩只发生在模型训练视图。

## 真实轨迹核验

对 `main-06085` 的 context ledger：

| 请求 | 阶段 | Gemini provider input tokens |
| --- | --- | ---: |
| 1 | Image Account Planning | 3,475 |
| 2 | ReAct | 7,508 |
| 3 | ReAct | 9,888 |
| 4 | Discrepancy Decision | 10,577 |
| 5 | Route-local Replan | 19,987 |

因此，运行时单次请求没有增长到完整 SFT 的数十万 token；完整 episode 过长主要不是
Gemini thought，也不是 provider 在同一 interaction 中无限累积历史。

第 5 个请求仍暴露了运行时问题：阶段专属输入约 47,970 字符，同时叠加约 31,029 字符的
完整 workspace，二者有 47 个相同状态 ID。Route-local Replan 还无必要地重传了约 390KB
原图。修复后，这些 stage-owned 阶段不再附加完整 workspace，Route-local Replan 不再重传原图。

## SFT 的主膨胀点

同一条完整 episode 有 79 条 messages、24 个带工具结果的 ReAct 边界。旧导出把下一阶段完整
packet 放进每个 `role=tool` 消息；阶段 packet 合计约 2.50MB。工具真实结果、调用参数和
reducer state delta 合计只约 126KB。

这意味着问题不是“轨迹本身太长”，而是把同一份累计 state 反复序列化进历史。

## v3 导出格式

首个 user 消息保留初始观察和完整的首个 stage packet。之后：

```text
assistant action
  -> tool observation + investigation_state_update + next_stage_control
  -> assistant next action
```

非工具阶段之间使用一次 `user` 的 `stage_control`。它只包含：

- 当前 stage；
- 输出模式（原生工具调用或结构化 JSON）；
- 当前 stage instruction；
- ReAct 当前允许的工具名。

历史中已经有前序 assistant action、真实 tool observation 和 reducer delta，因此不再重复
`workspace`、累计 `input_payload` 或完整 response schema。

对该真实轨迹，按 v3 格式模拟后从 3,359,479 UTF-8 字节降到约 307,284 字节，减少 90.9%。
最终是否准入训练仍必须由真实 Qwen/ms-swift processor 重新编码审计，不能只依赖字节估算。

## 运行时原则

每个阶段从 canonical state 生成一个有边界的投影；投影只服务当前阶段。完整 state 留在 archive，
模型需要旧材料时通过已记录的 observation、Evidence ID 或受控 archive recall 获得，不通过重复
粘贴全量 workspace 获得。

后续可额外从同一 canonical trace 派生 `prefix → next_action` 样本，但这不是替代 v3 完整 episode，
而是用于极长轨迹的补充训练视图。

## 2026-09-02 当前 ReAct 观察传递

当前运行时采用“上一轮结果显式前递一次”的方式：

```text
tool call
  -> canonical function_result（下一轮完整看到）
  -> bounded state index（只保留索引、近期观察和预算）
```

原始工具结果写入 archive，模型上下文不携带二进制图片、base64、完整 HTML 或旧的
Claim/Task 控制状态。原图按当前请求需要重新附加，但不会写入历史文本反复累积。

`visit` 的网页证据由 summary model 从候选段落中选择；被选段落原样保留，并在指代不完整时
加入必要的前置完整段落。上下文窗口只删除完整字段或完整记录，不切半段来源文本。

因此要区分三件事：

1. provider 是否成功返回；
2. 模型是否实际收到本轮工具观察；
3. 观察是否足以支持事实判断。

`status=success` 只说明第 1 点；空搜索结果、空页面提取和无效参考图不会自动变成
Evidence，也不会替代模型的最终二分类判断。

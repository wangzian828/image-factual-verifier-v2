# Agent 结构

当前生产入口是 `unified-react-v1`。IFV 是“策略模型 + 工具 + runtime state”的闭环，
不是多个固定顺序、各自独立规划的 Agent。

```text
                 ┌────────────────────────────┐
图片、case 字段 ─>│ unified-react-v1 policy     │
                 │ 压缩原图 + thought -> 工具  │
                 └─────────────┬──────────────┘
                               │
                         runtime adapter
                               │
       ┌───────────────────────┼───────────────────────┐
       │                       │                       │
  视觉工具                 检索工具                控制工具
 scene/OCR/compare      search/visit/read       stop_route
       │                       │                       │
       └──────────────> reducer/state delta <──────────┘
                               │
                  Reflection / Decision / Judgment
                               │
                         canonical trace
```

模型每轮只能调用一个 native tool。视觉感知不是隐藏的固定前置步骤，而是 ReAct 可以选择的
两个 bootstrap 工具；runtime 只规定在两者完成前不开放外部调查工具。

首次调查动作携带 `investigation_intent`。它提供一个图像锚定的正向 `target_fact` 和本次
动作的主路线；runtime 再创建 canonical route/task。后续换 query、换网页、换视觉方向，都
直接表现为下一轮 ReAct 的新动作。`route_local_replan` 如果被暴露，也是这个循环中的普通
控制工具，不是独立 Replan 阶段。

Reflection 只做低频全局检查；Discrepancy Decision 只解释已经记录的证据；Judgment 只处理
runtime 编译的最终 basis，并生成读者可读的 fact-check report。

图片输入边界：在 `direct_multimodal` 模式下，统一 ReAct、Reflection、
Discrepancy Decision 和 Judgment 的每次请求都会临时附加一份受控压缩原图。图片不追加到
累计文本历史，也不写入轨迹或 `policy_input` 的 base64；运行时只记录图片哈希、尺寸、压缩
大小和 artifact 引用。结构化视觉 workspace 只保存有限数量的实体属性、关系、场景细节和
不确定性，作为原图的紧凑索引。

`perceive_scene` 不再只返回一句场景摘要：它最多返回 16 个可见实体、实体属性、最多 16
条可见关系、场景细节和像素层面的不确定性。Bootstrap 将这些内容物化为绑定实体和
`VisualFact`，供后续 ReAct 和收尾阶段使用。

字段边界：

- `target_facts`：图片传达、需要核查的正向现实事实；
- `search_hypotheses` / `tasks`：runtime 管理的路线和任务；
- `discoveries`：检索线索；
- `evidence`：成功工具结果中可追溯的事实材料；
- `failures`：访问失败或工具失败记录；
- `verdict_basis`：最终结论实际引用的对象集合。

## 一条 episode 的真实顺序

```text
空 workspace
  -> ReAct 选择 perceive_scene 或 ocr_with_position
  -> ReAct 选择另一个视觉工具
  -> ReAct 首次调查动作 + investigation_intent
  -> ReAct 在 route/task 之间持续选择工具
  -> 低频 Reflection / Discrepancy Decision
  -> runtime 编译 verdict basis
  -> Judgment + fact_check_report
  -> canonical trace
```

每次工具返回后，Reducer 写入 state delta；模型不直接写 state。成功观察、Discovery、
Evidence、Finding、失败和路线状态都由 runtime 记录，模型只产生 thought、工具调用和
检查点 JSON。

## Prompt 与轨迹文件

运行时主 prompt 使用英文，源文件是 `src/orchestrator/unified_prompts.py`。精确备份由
`scripts/docs/export_active_agent_prompts.py` 生成到 `docs/active-agent-system-prompts.md`；
中文文件只是阅读对照。

完整 canonical trace 是 `traces/*.json`。Qwen SFT 导出中的
`trajectory_sft.jsonl` 是一行一个完整 episode；`manifest.json` 和
`selection-manifest.jsonl` 只是元数据/索引，不是轨迹。人类阅读版可用
`scripts/trajectory/render_sft_episodes_readable.py` 生成。

# Agent Prompt 与运行时说明（中文）

当前生产版本是 `unified-react-v1`。代码中的 prompt 和 schema 是唯一准则；本文件只解释
它们的配合方式。

1. **一个大的 ReAct**：每轮 `thought -> 一个 native tool -> observation/state delta`。
2. **视觉工具也是工具**：模型先选择 `perceive_scene` 或 `ocr_with_position`，二者完成后
   才开放外部调查工具，顺序不写死。
3. **首次调查携带 intent**：`investigation_intent.target_fact` 写正向现实事实，引用
   `anchor_fact_ids`；`route` 写本次工具要查的信息。runtime 创建 task 和 route。
4. **换方向直接换动作**：新 query、新页面、新视觉检查都直接成为下一轮 ReAct action，没
   有独立 Replan 阶段。
5. **低频检查**：Reflection 看全局策略；Discrepancy Decision 看已有证据语义；Judgment
   看 runtime 编译的 basis。

SFT 导出保留完整 episode，但删除重复的累计 workspace，只保留每轮需要的上下文和 state
delta。Qwen 目标格式是 `<think>...</think>` 加 Qwen 原生 `<tool_call>`；没有 provider
实际 thought 文本的轨迹进入 action-only/RL 桶，不伪造 thought。

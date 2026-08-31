# Agent Prompt 与运行时说明（中文对照）

当前生产版本是 `unified-react-v1`。运行时实际发送的主 Agent prompt 是英文，唯一源文件是
`src/orchestrator/unified_prompts.py`；本文件只提供中文说明，不是运行时输入。精确英文备份见
[`active-agent-system-prompts.md`](active-agent-system-prompts.md)。

1. **一个大的 ReAct**：每轮 `thought -> 一个 native tool -> observation/state delta`。
2. **视觉工具也是工具**：模型先选择 `perceive_scene` 或 `ocr_with_position`，二者完成后
   才开放外部调查工具，顺序不写死。
3. **首次调查携带 intent**：`investigation_intent.target_fact` 写正向现实事实，引用
   `anchor_fact_ids`；provider 写一个主 `route`，可选写 `alternate_route_focuses`。runtime
   将其扩展为 2–3 条候选 route/task，本轮只执行主路线。
4. **换方向直接换动作**：新 query、新页面、新视觉检查都直接成为下一轮 ReAct action，没
   有独立 Replan 阶段。
5. **路线不能过早结束**：只有当前 route 的候选和可执行 material step 都耗尽，才能调用
   `stop_route`；其他未完成 route 仍可继续调查。
6. **低频检查**：Reflection 看全局策略和原图；Discrepancy Decision 看已有证据语义和需要
   复核的可见属性；Judgment 看 runtime 编译的 basis，并用可用的图像材料完成最后二分类。

SFT 导出保留完整 episode，但删除重复的累计 workspace，只保留每轮需要的上下文和 state
delta。Qwen 目标格式是 `<think>...</think>` 加 Qwen 原生 `<tool_call>`；没有 provider
实际 thought 文本的轨迹进入 action-only/RL 桶，不伪造 thought。

在 `direct_multimodal` 模式下，统一 ReAct、Reflection、Discrepancy Decision 和 Judgment
的每次请求都会临时附加一份压缩原图（默认最长边 1280、JPEG quality 88）。图片不会追加到
累计文本历史，也不会把 base64 写进轨迹；context ledger 只保存图片引用、哈希、尺寸和大小。
`perceive_scene` 的结构化结果是原图的紧凑索引，最多保留 16 个实体及属性、16 条可见关系、
场景细节和像素不确定性，不能替代原图。

# Agent Prompt and Runtime Guide

当前唯一生产策略：`unified-react-v1`。实际 system prompt 在
`src/orchestrator/unified_prompts.py`，本文说明运行边界。

## 1. 主 ReAct

每轮输出：

```text
thought -> exactly one native tool call -> tool observation/state delta
```

模型可以选择 scene 与 OCR 的先后顺序，但二者完成前不能调查外部来源。首次非 bootstrap
动作必须带 `investigation_intent`，其中的 `target_fact` 是正向、原子、图像锚定的现实
命题，`route` 是本次工具实际要获取的信息。runtime 随后创建 task/route/ID。

后续换 query、换候选页或换视觉检查，直接调用下一工具；不再有独立 Query Replan、Route
Replan 或 Planning 请求。

## 2. 低频检查点

- `unified_reflection`：只总结全局缺口和策略，不选具体工具。
- `unified_discrepancy_decision`：只依据已记录 Evidence、Finding 和视觉锚点更新语义状态，
  不创建新路线。
- `unified_judgment`：只复现 runtime 编译的 verdict/basis，或在未闭合时做受限二元判断。

## 3. 上下文和输出上限

runtime 每轮只发送紧凑 workspace、最近观察、活跃路线、开放缺口和 state delta；完整累计
workspace 只保存在 archive，不重复塞进每个训练 turn。

默认输出上限：ReAct、Reflection、Decision、Judgment 均为 8192；实际 provider 可通过现有
环境变量覆盖。输出上限是 completion 上限，不等于单次工具超时。

## 4. API 边界

`direct_multimodal` 把图片交给主策略请求；`separate_vlm` 把图片交给视觉工具/VLM，主策略
只接收结构化观察。工具内部 prompt、上传、压缩、超时和重试逻辑保持在工具实现中。

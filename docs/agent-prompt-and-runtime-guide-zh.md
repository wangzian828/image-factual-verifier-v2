# Agent Prompt 与运行时说明（中文阅读版）

当前生产入口是 `unified-react-v1`。运行时英文 prompt 定义在
`src/orchestrator/unified_prompts.py`，精确导出见
[当前英文 system prompt](active-agent-system-prompts.md)。

## 一个 raw-history ReAct loop

```text
原图 + 固定任务 + provider 累计历史
  → thought
  → 恰好一个公开 native tool 调用
  → 原始 function result
  → 同一 session 的下一次 interaction
```

当前没有强制视觉 bootstrap，也没有独立的 Planning、路线图、Replan、Reflection
或 Discrepancy Decision。模型通过下一轮选择其他工具来调整方向；当继续行动不太可能
实质改变有界判断时，可以调用 `finish_investigation`。

runtime 只暴露公开参数、注入执行字段、校验来源策略和预算，并记录工具结果。它不
把结果压缩成 visual memory，也不分类成 Discovery、Evidence 或 failure。成功、空结果、
malformed 和错误结果都原样保留在 raw history 中。

## 会话和图片

完整调查与 Judgment 共用一个 `InteractionSession`。`direct_multimodal` 模式下，根
请求附加一次受控原图；下一轮只提交刚完成的 `function_result`，并通过
`previous_interaction_id` 继续会话。更早的观察由 provider history 保留，不重新拼接
成 workspace。

`separate_vlm` 模式下，视觉工具接收图片，policy 读取其原始结果。候选图、裁剪图和
聚焦视图只在产生它们的 observation 边界附加。canonical text 不保存 base64，媒体由
artifact 引用和 SHA-256 定位。

机械 context 只包含目标、原图可用性和总动作预算；持久化控制状态只包含 case ID、
图片 SHA-256、动作数、停止原因和完成说明。

## 解释边界

搜索标题、摘要、URL 和反向搜图候选只是线索，直到实际返回的网页内容或比较结果回答
当前问题。工具错误和访问失败只是限制；一次成功的定向无匹配只说明该 query 没找到
匹配，不能单独证明 `fake`。policy 和 Judgment 必须在每个原始结果的实际范围内解释。

`visit` 获取页面后只选择有界正文片段，同时在 trace artifact 中保存精确 source span；
完整原始网页不进入 Agent 文本。视觉工具在 provider schema 和本地规范化层分别执行
兼容性与长度边界。

## Judgment 与 SFT

finish、预算耗尽或有界协议停止后，Judgment 在同一 session 中继续。它接收机械
observation locator，只能引用成功的原始 observation ID，不能再调用工具或发明观察。

SFT exporter 按时间顺序导出一整条 Qwen episode：thought、native tool call、原始
tool response 和最终 answer。它恢复每个请求边界实际看到的图片，写入可迁移 data URI，
并按 SHA-256 去重。缺少真实 ReAct thought 的轨迹只标记为 action-only/RL 材料，绝不
伪造 thought。

# SFT 训练与数据构造

本文只描述当前的 reasoning policy SFT 和独立 perception SFT。

## 1. policy SFT 一行是什么

一行就是一条完整 Agent episode，不按阶段拆成互不相干的样本：

```text
system
user: 图片 + 事实核查任务
assistant: <think>...</think>
tool_call: {"name":"工具名","arguments":"{...}"}
tool_response: 完整公开工具结果
tool_call: {"name":"另一个工具","arguments":"{...}"}
tool_response: 完整公开工具结果
assistant: <think>...</think>
tool_call: ...
tool_response: ...
assistant: <think>...</think><answer>{最终报告}</answer>
```

policy 学的是“给定已有上下文，下一步如何思考、是否调用哪个工具、最后如何写报告”。
工具结果是条件上下文，不是模型要复述的目标；assistant 的 thought、工具调用和最终
报告才是监督目标。

## 2. 导出字段

policy 行的训练输入只保留：

```json
{
  "tools": "[...]",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image>..."},
    {"role": "assistant", "content": "<think>...</think>"},
    {"role": "tool_call", "content": "{\"name\":\"...\",\"arguments\":\"{...}\"}"},
    {"role": "tool_response", "content": "..."},
    {"role": "assistant", "content": "<think>...</think><answer>...</answer>"}
  ],
  "images": ["data:image/jpeg;base64,..."]
}
```

每个消息只有 `role` 和 `content`。不写 `loss`、`channel` 或
`chat_template_kwargs`，由目标 Qwen checkpoint 的 ms-swift template 自动生成 labels。
真实 ms-swift 样本允许在一个工具结果后紧接另一个 `tool_call`；导出器保留这种顺序，
但每个工具调用都必须有配对的工具结果。

每个 ReAct 动作的 `<think>` 必须来自 provider 原生返回。导出器不会凭空补
thought；缺少 ReAct thought 的轨迹不进入 reasoning SFT。最终 Judgment 或其他
结构化输出阶段没有 thought，不会单独导致一条完整 ReAct 轨迹被拒绝。

## 3. 上下文与图片

- 初始 user 消息提供图片占位符和任务。
- 每轮工具结果在生成后进入下一轮一次。
- 不把累计 workspace、旧 request snapshot 或同一工具结果重复嵌入每轮。
- 正式发布包把图片写在顶层 `images`，内容为可迁移 data URI；base64 不进入
  `messages[].content`。
- 初始原图对应初始 user 的 `<image>`；搜索候选图、裁剪图和聚焦复查图对应产生
  它们的 `tool_response` 后追加的 `<image>`。
- 图片按 SHA-256 去重，marker 总数必须等于 `images` 总数。
- 主 Agent 使用受控原图；视觉工具按需单独接收原图、裁剪图或参考图。

这样既保留调查所需上下文，也避免完整状态在 episode 中指数式重复。

## 4. policy 与 perception 分开

### policy

输入是完整 ReAct episode，目标包含 thought、工具动作和最终 `<answer>`。它训练调查
策略，不训练隐藏 runtime 字段或 evaluator private gold。

### perception

输入是：

```text
system
user: <image> + “只报告图片中可见内容”
assistant: PerceptionReport JSON
```

perception 不伪造 `<think>`，也不混入 policy 的工具调用。它是独立图片观察能力，
通过 `ms-swift-perception/` 训练。

## 5. 数据筛选

原始 rollout 先经过工程审计和 SFT judge，再进入 frozen accepted release。筛选记录
保留原始 trace，不覆盖或删除失败轨迹。

进入 reasoning policy SFT 的基本条件：

- episode 结构完整；
- 工具调用和工具结果顺序正确；
- provider thought 可读且位于真实 assistant turn；
- 没有 evaluator private 字段；
- 图像路径存在；
- 目标 processor 编码成功并产生非空 assistant labels；
- 没有超过当前上下文上限。

质量桶由现有 SFT judge 负责；`high`、`usable`、`rejected` 和
`engineering_error` 不由转换器重新调用 judge。action-only 和 perception 是独立
产物，不能静默混进 reasoning policy 文件。

## 6. 训练设置

训练入口仍使用 ms-swift 的 `swift sft`。本项目转换器只负责生成标准输入，不实现
第二套 trainer。

- 监督目标由 Qwen chat template 生成；
- assistant 的 thought、tool call 和最终 answer 参与 loss；
- system、user、tool response 作为条件上下文；
- 真实 processor 决定 special tokens、图片 token、labels 和最终长度；
- reasoning SFT 保持 `IFV_ADD_NON_THINKING_PREFIX` 关闭；
- action-only 不进入 reasoning SFT。

训练前必须运行真实 processor 验证脚本，不能仅依据 JSON 校验通过就开训。

## 7. 检查命令

```powershell
cd training
python -m pytest -q
python scripts/probe/verify_ms_swift_agent_dataset.py `
  --model <Qwen checkpoint> `
  --policy-dir <ms-swift-policy> `
  --perception-dir <ms-swift-perception> `
  --output <processor-verification.json>
```

验证脚本会按 Qwen 模板实际渲染的 `<function=...>` /
`<parameter=...>` 检查工具调用，检查 thought 是否进入 labels、tool response 是否
进入编码输入、图片是否被 processor 接收，以及最长样本是否超过上下文上限。

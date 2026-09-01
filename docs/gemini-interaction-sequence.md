# Gemini 交互顺序

当前生产路径是 `unified-react-v1`。本文件只描述当前主流程；旧的
Claim/route/task 链路仅作为历史资料保留。

## 单条 episode

```text
创建空 workspace
  -> Gemini ReAct：选择一个当前可用工具
  -> function result
  -> reducer 写入观察和 state delta
  -> Gemini ReAct：读取累计会话历史、当前原图和最新紧凑状态
  -> ...重复 ReAct action...
  -> finish_investigation 或达到预算
  -> runtime 编译 basis
  -> 独立 Judgment
```

每个 ReAct action 只允许一个 native function。ReAct 的所有 action 共用一个
`InteractionSession`：

1. 第一个请求上传压缩后的原图和初始上下文。
2. 工具完成后，下一请求显式提交这一轮的完整 `function_result`。
3. 下一请求使用上一响应的 `previous_interaction_id`，由 Gemini 保留更早的会话历史。
4. 原图不会在每一轮重复上传；它已经存在于 provider 会话根请求中。
5. 参考图或聚焦视觉结果作为当前轮新增的多模态 `user_input` 追加，并与当前文本合并为一个
   `user_input` 步骤。

每轮工具调用还携带模型自行维护的 `investigation_progress`：

- `investigating`：仍有重要事实问题没有闭合；
- `decision_capable_support`：模型判断当前材料已经直接支持图片表达的事实；
- `decision_capable_refute`：模型判断当前材料已经直接反驳图片表达的事实。

这只是模型的调查状态。runtime 负责校验、保存和传回，不根据工具名称或工具结果
自动推断，也不根据它动态增删工具。模型只有在声明后两个方向性状态之一时才能
主动调用 `finish_investigation`；达到 24 次动作上限时仍沿用现有流程进入 Judgment。

因此，工具结果的生命周期是：

```text
第 N 轮：function_call -> 本地执行工具 -> 记录完整 tool_result
第 N+1 轮：function_result(call_id=N) + 当前上下文
第 N+2 轮：只提交第 N+1 轮的新 function_result，不手工重复第 N 轮
```

这与 WebWatcher 的累计消息历史原则一致：历史由会话链保留，当前轮只增加新的工具观察。
网页原文不直接进入 Agent；`visit` 先由 Jina 读取，再由 summary model 提取可用证据和摘要。

## 轨迹、审计和训练

- canonical trace 保存每轮真实的 `thought`、函数调用、工具结果、父交互 ID、请求快照和 reducer
  delta。
- strict audit 要求 unified ReAct 的 native interaction 父链连续，并检查完成工具动作后的
  `function_result` 确实出现在下一请求。
- SFT exporter 按时间顺序拼成一条完整 episode：
  `thought -> tool_call -> tool result -> next thought`。
- SFT 不重复展开 provider 已保存的历史，也不把 `policy_input` 的历史快照当成额外监督样本。
- `trajectory_sft.jsonl` 保留完整 episode；`report_history` 和可读视图只投影一次事件，不重复
  展开累计 workspace。
- hidden reasoning 不写入训练数据；只有 API 实际返回的可读 thought 才进入 reasoning SFT。

工具失败写入当前 action 和 runtime failure 列表；SSL、验证码、页面不可访问等外部问题不等同
于工程失败，malformed tool contract 才进入工程错误路径。

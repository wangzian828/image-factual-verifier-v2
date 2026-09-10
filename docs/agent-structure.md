# IFV Agent 结构与训练接口

本文描述当前生产入口 `unified-react-v1`。旧 v4 graph、reducer 和 replay 兼容路径
不属于当前运行协议。

## 1. 运行结构

```text
图片 + 固定事实核查目标
        ↓
同一个 provider InteractionSession
        ↓
thought → 一个 native tool call → 原始 tool result
        ↖──────────────────────────────┘
        ↓ finish / budget / protocol boundary
Judgment → real/fake + fact-check report + observation IDs
```

每轮的完整工具结果在 provider history 和 canonical trace 中各保留一份。下一轮仅
提交刚完成的 `function_result` 并继续父 interaction，不把历史重新拼成压缩 workspace。
runtime 只维护动作计数、停止原因和完成说明。

## 2. 单轮规则

1. 模型读取固定任务、原图（模式允许时）和累计 raw history。
2. 模型输出可读 thought，并选择恰好一个公开 native tool。
3. runtime 校验公开参数、来源策略和预算，执行成熟工具。
4. 成功、空结果、外部不可用、错误和 malformed 结果都原样入 trace。
5. 模型在下一轮自行解释结果并换路线，或调用 `finish_investigation`。

Planning、Query Replan、Route Replan、Reflection 和 Discrepancy Decision 都不是
独立阶段。工具失败不会被静默丢弃，也不会被转换成事实证据；只有不可恢复的 case、
worker 或持久化故障会让运行报错退出。

## 3. 当前工具

- 视觉/OCR：`perceive_scene`、`ocr_with_position`、
  `focused_visual_inspection`、`crop_and_inspect`、`check_consistency`、
  `analyze_visual_anomalies`、`count_objects`；
- 检索/访问：`text_search`、`text_image_search`、`reverse_image_search`、`visit`；
- 比较/控制：`compare_with_reference`、`current_time`、`finish_investigation`。

搜索候选只是待核查线索。网页正文、视觉观察和有效比较也必须按工具实际返回的
精确范围解释；画质、疑似生成痕迹、访问失败或没有搜到结果不能自动决定标签。

## 4. 图片和上下文

- `direct_multimodal` 根请求上传一次受控原图，后续请求复用 provider session；
- `separate_vlm` 由视觉工具接收图片，policy 读取其 raw result；
- 视觉 API 输入统一受尺寸和编码边界控制，文本历史不嵌入 base64；
- 候选图、裁剪图和聚焦视图只附在产生它们的 observation 边界；
- evaluator private gold、内部路径、缓存和 provider wire 私有字段不进入模型内容。

## 5. Judgment

Judgment 复用调查 session，读取完整 raw history 和一个只含 locator 的机械上下文。
它输出二分类、读者可读报告以及实际采用的成功 observation IDs。Judgment 不调用
新工具、不新增观察，也不能把证据缺口或视觉瑕疵自动当成 `fake`。

## 6. SFT 消息格式

policy SFT 一行对应一个完整 episode：

```text
system
user: <image> + task
assistant: <think>...</think>
tool_call: {"name":"text_search","arguments":"{...}"}
tool_response: 原始公开工具结果
assistant: <think>...</think>
tool_call: ...
tool_response: ...
assistant: <answer>{...}</answer>
```

顶层只使用 `tools`、`messages` 和 `images`。正式可迁移包的图片写成
`data:image/...;base64,...`，并按 SHA-256 在 episode 内去重；`<image>` marker
数量必须等于 `images` 长度。每条 message 只含 `role` 和 `content`，真实 Qwen/
ms-swift processor 负责模板和 labels。

导出器不编造 thought。缺少 ReAct thought 的轨迹进入 action-only/RL 用途；无
thought 的结构化 Judgment 不会淘汰前面已有完整 thought 的 episode。发布前必须做
strict trace audit 和目标 checkpoint 的真实 processor 验证。

# IFV Agent 结构与训练接口

本文描述当前生产树 `unified-react-v1`。旧 v4 阶段图和旧兼容路径不属于当前运行入口。

## 1. 运行结构

```text
图片 + 事实核查任务
        │
        ▼
统一 ReAct 循环
  每轮：模型 thought → 一个工具动作
        ▲              │
        │              ▼
        │       工具结果 + 本轮状态更新
        │
        └── 运行时保留有限状态、预算和失败记录
        │
        ▼
finish_investigation
        │
        ▼
最终 Judgment：二分类 verdict + 完整事实核查报告
```

Planning、Query Replan、Route Replan、Reflection 不再作为主流程中的独立
“大阶段”。必要的路线调整由 ReAct 模型在下一轮 thought 中完成；运行时只负责工具
契约、状态落盘、预算和错误边界。

## 2. ReAct 每轮怎么走

1. 模型看到图片、任务、之前一轮的工具结果和紧凑状态。
2. 模型输出原生 `<think>...</think>`，然后选择一个工具。
3. runtime 执行工具，并把完整的公开工具结果作为下一轮输入。
4. runtime 同时保存结构化状态，例如视觉记忆、检索发现、访问失败、已尝试动作和预算。
5. 模型继续调查，或调用 `finish_investigation` 进入最终 Judgment。

下一轮复用上一轮工具结果一次，不把同一份结果无限复制到后续请求。原图由图像输入
链路单独管理，不写入文本历史，不把 base64 反复追加到轨迹。

## 3. 当前工具

- `perceive_scene`：整体视觉观察、实体、关系、文字和不确定性。
- `ocr_with_position`：带位置的文字识别。
- `focused_visual_inspection`：针对当前问题的局部或关系复查。
- `crop_and_inspect`：按区域复查图片。
- `check_consistency`：检查图像内部的视觉一致性。
- `analyze_visual_anomalies`：提供视觉异常线索；异常线索不能单独证明事实为假。
- `text_search`：文本检索网页候选。
- `text_image_search`：文本检索图片和含图页面候选。
- `reverse_image_search`：当前图片的反向搜图候选。
- `visit`：访问候选网页并提取与当前调查相关的内容。
- `compare_with_reference`：将当前图与有效参考图比较。
- `finish_investigation`：结束 ReAct 调查，不直接决定 `real/fake`。

搜索候选只是线索。只有后续工具结果中真正与当前图片事实相关的内容，才可作为
调查依据。网页访问失败、SSL、验证码和参考图不可用要记录为外部不可用，不伪装成
正常证据；空结果和 malformed tool result 同样只是当前动作失败观察。它们保留在
failure ledger 并进入下一轮，Agent 应换路线继续；只有 worker 或不可恢复运行时状态
故障才会终止当前 case。

## 4. 图像与上下文

- 主 Agent 的每次视觉请求都使用受控原图；同一 provider session 复用原图，不重复上传。
- 视觉工具需要时单独接收原图、裁剪图或候选参考图。
- 进入视觉 API 的图像最长边为 1024，JPEG 质量保持在较高水平；文本历史不保存 base64。
- 工具返回的公开结果会进入下一轮 provider 会话，并进入 canonical trace；后续轮次
  通过同一个 Interaction 历史继续可见，不在每轮文本包里重复复制。
- `text_image_search`、`reverse_image_search` 和 `crop_and_search` 新得到的候选图最多
  取 3 张，作为紧邻工具结果的下一轮多模态输入；裁剪/聚焦检查产生的新视图同样
  紧邻对应工具结果。
- runtime 内部字段、provider wire 数据、缓存和 evaluator 私有 gold 不进入模型可见内容。

## 5. 最终输出

Judgment 接收调查过程和最终报告上下文，输出：

- 二分类 `verdict`；
- 简短但完整的事实核查报告；
- 支持判断的调查理由；
- 不能把“疑似 AI 痕迹”、画质问题、OCR 乱码或页面不可访问本身当作事实错误。

最终 Judgment 是收尾步骤，不替代 ReAct 调查，也不要求所有 case 都找到决定性外部证据。
轨迹质量由后续统一的 SFT judge 和测试集 private-gold judge 分别评估。

## 6. SFT 消息格式

当前 policy SFT 一行对应一条完整 episode：

```text
system
user: <image> + 任务
assistant: <think>...</think>
tool_call: {"name":"text_search","arguments":"{\"query\":\"...\"}"}
tool_response: 完整公开工具结果
tool_call: {"name":"visit","arguments":"{\"url\":\"...\"}"}
tool_response: 完整公开工具结果
assistant: <think>...</think>
tool_call: ...
tool_response: ...
assistant: <think>...</think><answer>{...}</answer>
```

顶层字段是：

```json
{
  "tools": "[...]",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image>..."},
    {"role": "assistant", "content": "<think>...</think>"},
    {"role": "tool_call", "content": "{\"name\":\"text_search\",\"arguments\":\"{...}\"}"},
    {"role": "tool_response", "content": "..."},
    {"role": "assistant", "content": "<think>...</think><answer>...</answer>"}
  ],
  "images": ["data:image/jpeg;base64,..."]
}
```

正式可迁移发布包使用 `data:image/...;base64,...`。导出器从 runtime context
artifact 恢复“每次 policy 请求实际看到的图片”，将初始原图放在初始 user
`<image>`，将工具后新出现的候选图、裁剪图或聚焦视图放在对应 `tool_response`
后，并按 SHA-256 全 episode 去重。`<image>` marker 数必须与顶层 `images`
长度严格相等。

每个消息只含 `role` 和 `content`。不在消息上写 `loss`、`channel` 或
`chat_template_kwargs`；由真实 Qwen/ms-swift processor 根据模板生成 labels。
同一轮 assistant 动作批次可以在一个 `tool_response` 后继续出现下一个
`tool_call`；每个 `tool_call` 都必须有对应的 `tool_response`。

policy SFT 要求每个 ReAct 动作有 provider 原生 thought；Judgment 等结构化阶段
没有 thought 不会误伤整条轨迹。导出器不编造 thought。缺少 ReAct thought 的轨迹
进入独立 action-only 或其他非 reasoning 用途。

## 7. 两条训练输入

- policy：学习在上下文中思考、选择工具和输出最终报告。
- perception：独立的图片观察任务，学习可见内容 JSON；不伪造 thought。

两者都使用 ms-swift 的 `messages` + `images` 输入。训练前必须用目标 checkpoint
的真实 processor 编码验证，不能只凭 JSON 结构判断可训练。

# Agent Prompt 与运行时说明（中文阅读版）

当前生产入口是 `unified-react-v1`。实际发送给模型的主 prompt 使用英文，源文件是
`src/orchestrator/unified_prompts.py`；本文件只供人工阅读，不会作为运行时输入。

## 1. 一个连续的 ReAct

每轮都按下面的形式运行：

```text
原图 + 固定任务 + 紧凑记忆
  → thought
  → 一个公开 native tool 调用
  → 工具观察结果 + reducer 状态增量
```

`perceive_scene` 和 `ocr_with_position` 都是普通工具。模型自行决定先后顺序，
后续也可以再次调用视觉工具。当前主流程没有强制 bootstrap 顺序，也不再让模型
先生成 Planning、Query Replan 或 Route/Task 图。

换 query、换网页、检查反向搜图候选、提出新的视觉问题和结束调查，全部是同一
ReAct loop 中的动作。ID、图片路径、内部工具参数、去重、重试、预算、失败记录
和状态更新由 runtime 管理。

三个搜索工具职责不同：

- `text_search`：根据事实 query 查找网页候选；
- `text_image_search`：根据事实 query 查找图片或含图页面，适合已有具体的
  人物、事件、地点、物体或图片文字，需要定位相关图片时使用；
- `reverse_image_search`：上传当前图片，寻找 Lens 或语义图像对应候选。

三者返回的都只是未验证 Discovery。`text_image_search` 找到候选后，仍需用
`visit` 或 `compare_with_reference` 继续核查，不能直接当作 Evidence。

## 2. 上下文和图片

每个 Gemini Interaction 的根请求会临时附加一份受控原图，后续请求通过同一个
provider session 复用这张图，不重复上传。图片不会追加到文本历史，base64 也
不会写入 trace。下一轮会接收：

- 固定事实核查目标；
- `visual_memory`；
- 尚未验证的搜索/反向搜图候选；
- 成功的视觉、比较或网页 Evidence；
- 外部访问失败和工程错误；
- query、URL、近期动作、未解决问题和剩余预算。

每轮工具调用还带有模型自己维护的 `investigation_progress`。存在重要事实
缺口时保持 `investigating`；只有模型自己判断当前材料已经直接支持或反驳图片
表达的事实时，才改为 `decision_capable_support` 或
`decision_capable_refute`。runtime 只校验结构、保存、传回并检查主动结束请求
中的这个状态，不根据工具名称、结果字段或 `evidence_class` 替模型判断，也不
根据这个状态动态增删调查工具。既有 `evidence_class` 仍用于归档、报告和审计。
上下文同时提供总动作预算和各工具剩余预算。

完整请求、响应和状态仍保存在归档中供审计，但不会整段复制进每一轮 prompt。
刚完成的 canonical 工具结果会以完整文本传给下一轮；只排除二进制传输字段和
原始 HTML，不做字符截断或列表裁剪。

## 3. 证据边界

搜索结果、标题、摘要、URL、来源标签和猜测只是线索。只有成功的视觉/OCR 观察、
有效的图像比较或已检查网页的具体正文片段，才可能支持最终报告。相似图片或
背景网页不能单独证明图片表达的完整事实。

只有在 `investigation_progress.status=decision_capable_support` 或
`decision_capable_refute` 时，模型才能主动调用 `finish_investigation`。否则继续
使用同一组可用工具。达到总动作上限后，沿用现有流程直接进入 Judgment，不新增
其他终止路径。最终证据是否充分仍由现有 SFT 审核判断。

所有进入视觉 API 的图片统一为最长边 1024、JPEG quality 95，包括原图、裁剪图、
候选图和单图 contact sheet。

## 4. SFT 导出

完整 episode 按 Qwen 可读格式导出：

```text
system
user: 任务 + 紧凑上下文
assistant: <think>...</think>
          <tool_call>...</tool_call>
tool: 结果 + 状态增量
...
assistant: 最终报告
```

导出器保留一条完整 episode，删除重复的累计 workspace，不按动作拆成多条独立
训练样本。provider 没有实际返回 thought 文本的轨迹归入 action-only/RL 材料，
不伪造思考内容。

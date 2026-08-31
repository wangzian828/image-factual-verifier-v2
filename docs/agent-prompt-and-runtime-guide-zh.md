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

## 2. 上下文和图片

每次直接多模态请求都会临时附加一份受控压缩原图。图片不会反复追加到文本历史，
base64 也不会写入 trace。下一轮只接收有限的：

- 固定事实核查目标；
- `visual_memory`；
- 尚未验证的搜索/反向搜图候选；
- 成功的视觉、比较或网页 Evidence；
- 外部访问失败和工程错误；
- query、URL、近期动作、未解决问题和剩余预算。

完整请求、响应和状态仍保存在归档中供审计，但不会整段复制进每一轮 prompt。

## 3. 证据边界

搜索结果、标题、摘要、URL、来源标签和猜测只是线索。只有成功的视觉/OCR 观察、
有效的图像比较或已检查网页的具体正文片段，才可能支持最终报告。相似图片或
背景网页不能单独证明图片表达的完整事实。

`finish_investigation` 只结束 ReAct 调查，不直接决定 `real/fake`。随后由 Judgment
输出二分类和面向读者的事实核查报告。信息不完整时可以做收尾判断，但缺少证据
本身不是任一标签的自动依据。

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

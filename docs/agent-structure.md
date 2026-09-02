# Agent 结构

当前生产入口是 `unified-react-v1`。主流程只有一个连续的
image-grounded ReAct loop，状态由 runtime/reducer 管理，成熟工具继续负责
各自的视觉、检索和网页处理。

```text
图片 + 固定事实核查任务
          │
          ▼
┌──────────────────────────────┐
│ unified ReAct policy          │
│ 每轮：thought → 一个工具调用 │
│ 每次请求临时附加压缩原图     │
└──────────────┬───────────────┘
               │ public tool schema
               ▼
┌──────────────────────────────┐
│ runtime tool adapter         │
│ 隐藏内部字段，注入图片/上下文 │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ mature tools                 │
│ perceive / OCR / search /    │
│ visit / compare / visual ... │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ reducer + compact state      │
│ visual_memory / discoveries  │
│ evidence / failures / budget │
└──────────────┬───────────────┘
               │
        下一轮 ReAct 或 finish
               │
               ▼
┌──────────────────────────────┐
│ Judgment                     │
│ 二分类 + fact-check report   │
└──────────────┬───────────────┘
               ▼
        canonical trace / SFT / RL
```

## 运行规则

- `perceive_scene` 和 `ocr_with_position` 是普通 ReAct 工具。模型可以先调用
  任意一个，也可以在后续因新问题再次调用；没有固定 bootstrap 闸门。
- 每轮只允许一个工具调用。查询、换页面、反向搜图、视觉复查和结束调查，
  都是同一个循环中的动作。
- 视觉工具默认发送受控压缩图：最长边 2048、JPEG quality 92。`perceive_scene`
  遇到可恢复的 provider 400、传输失败或超时时，会再用最长边 1280、quality 88
  的压缩图和宽松对象 schema 尝试一次；每次尝试都写入工具结果，不能把失败伪装成
  正常观察。
- 模型只产生 thought 和公开工具参数。ID、图片路径、工具内部参数、去重、
  预算、失败记录和 state delta 由 runtime 负责。
- 每次直接多模态请求都会临时附加同一张受控压缩原图。原图不重复写入文本
  历史，也不把 base64 写入 trace。
- `investigation_progress` 是模型在每轮动作中自行维护的调查状态：
  `investigating`、`decision_capable_support` 或
  `decision_capable_refute`。runtime 只校验结构、保存和传回；不会根据工具
  名称、`stance`、`directness`、`relevance` 或 `evidence_class` 推断这个状态，
  也不会因此动态增删调查工具。主动结束时，模型必须自行声明两个方向性状态
  之一；达到总动作上限时仍按现有流程进入 Judgment。
- `finish_investigation` 只结束调查，不决定 `real/fake`；最终标签和报告由
  Judgment 收尾。

## 当前状态

当前 runtime 状态不再建立 Claim/Route/Task 图，核心字段是：

- `objective`：固定事实核查任务；
- `visual_memory`：场景、实体、属性、关系、OCR 和像素不确定性；
- `discoveries`：搜索和反向搜图候选，默认是 `unverified`；
- `evidence`：成功工具结果中可追溯的观察或正文片段；
- `failures`：工程错误、网络失败和外部页面不可访问记录；
- `attempted_queries`、`visited_urls`、`attempted_actions`：去重和调查历史；
- `recent_actions`、`open_questions`、`current_focus`：有限的当前上下文；
- `action_count`、`no_gain_streak`、`stop_reason`：预算和终止状态。

旧的 `target_facts`、`search_hypotheses`、`tasks` 等字段只存在于 legacy
回放/审计代码，不属于当前 ReAct trace。

## 训练数据

当前训练使用的是“完整 episode 一行一条”的格式。`export_trajectory_sft_example()`
会把一条 canonical trace 按真实时间顺序投影为一条 Qwen 对话；不会把每个工具动作
拆成一条独立训练样本。

```text
system
user: task + compact context
assistant: <think>...</think> + Qwen tool call
tool: result + state delta
...
assistant: final judgment/report
```

### 一行数据的组成

每行是一个 JSON 对象，核心字段如下：

```json
{
  "trajectory_version": "ifv-trajectory-sft-v3",
  "episode_id": "case-001",
  "case_id": "case-001",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<think>...</think>\n\n<tool_call>...</tool_call>", "loss": true},
    {"role": "tool", "tool_call_id": "...", "content": "{\"result\": ...}"},
    {"role": "assistant", "content": "{\"verdict\":\"fake\", ...}", "loss": true}
  ],
  "tools": "[...]",
  "token_count_estimate": 12345,
  "message_count": 6,
  "tool_call_count": 1
}
```

- `system`：训练模式和 Agent 行为边界。
- 第一条 `user`：case 标识、图片哈希和初始阶段输入；不包含 private gold。
- 后续 `assistant`：模型要学习的目标输出。ReAct 轮次是
  `<think>...</think>` 加一个 Qwen 原生格式的 `<tool_call>`；最终 Judgment
  是结构化 JSON。
- `tool`：上一轮工具的完整、可审计结果，以及必要的 state delta。它是下一轮
  assistant 的上下文，不是模型要复述的目标。
- `tools`：该 episode 实际可用的工具 schema，供 Qwen 对齐工具调用参数。
- `loss=true`：只标在 assistant 消息上，表示这些内容进入监督损失；system、user
  和 tool 是条件上下文，不作为输出目标。

### ReAct 轮次怎样拼接

一轮 ReAct 的训练顺序固定为：

```text
上一轮上下文
  → assistant: <think>...</think> + <tool_call>
  → tool: function_call_id + tool name + result + state delta
  → 下一轮 assistant
```

工具结果只在它产生后的下一轮进入对话一次。下一轮不重新拼接完整累计
workspace、旧的 provider request snapshot 或重复的原图 base64。完整 runtime 状态
仍保存在 canonical trace/archive 中，训练行只保留该轮继续决策所需的上下文。

图片本身不写成文本或 base64。运行时的原图由图像输入链路单独管理；SFT 行保留
case/image 元数据和可追溯引用，避免把大图片重复塞进每个历史消息。

### 训练目标和分桶

当前默认转换器只接受完整 `trajectory_sft` 行，不接受旧的
`ifv-policy-dataset-v2` 逐动作数据。训练目标仍然是行为克隆：在给定历史消息、
工具结果和当前状态后，学习下一条 assistant 的 thought、工具调用参数或最终
Judgment 输出。

- 有完整 provider thought 的轨迹进入 `reasoning_sft`；
- 没有可读 thought 的轨迹进入 `action_only`，不伪造 `<think>`；
- 可执行的完整轨迹可以同时登记为 `rl_candidate`，但不等于自动进入 SFT；
- 超过上下文预算、协议未恢复或审计未通过的轨迹进入 holdout/rejected 桶；
- 分桶只建立清单，不删除 canonical trace，也不改变原始工具结果。

因此，训练时学到的是“看见当前历史后如何继续调查和调用工具”，而不是背诵
runtime 内部状态、private gold、完整网页缓存或 provider 的内部交互协议。

主 prompt 的英文源文件是
`src/orchestrator/unified_prompts.py`；工具内部 prompt 仍放在各自工具源码中。
中文 prompt 文件只用于人工阅读。

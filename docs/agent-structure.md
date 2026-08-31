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
- 模型只产生 thought 和公开工具参数。ID、图片路径、工具内部参数、去重、
  预算、失败记录和 state delta 由 runtime 负责。
- 每次直接多模态请求都会临时附加同一张受控压缩原图。原图不重复写入文本
  历史，也不把 base64 写入 trace。
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

`export_trajectory_sft_example()` 将一条完整 episode 导出为一条 Qwen 对话：

```text
system
user: task + compact context
assistant: <think>...</think> + Qwen tool call
tool: result + state delta
...
assistant: final judgment/report
```

不把一条 episode 拆成逐步独立样本，也不重复拼接完整 workspace。没有 provider
实际 thought 文本的轨迹进入 action-only/RL 候选，不伪造思考文本。

主 prompt 的英文源文件是
`src/orchestrator/unified_prompts.py`；工具内部 prompt 仍放在各自工具源码中。
中文 prompt 文件只用于人工阅读。

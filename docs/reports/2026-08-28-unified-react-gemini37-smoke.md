# unified-react-v1 Gemini 3.7 真实 smoke 记录

日期：2026 年 8 月 28 日

## 运行

- 服务器：`gpu-13`
- 代码：`8a92d8227255a453e40973abd4b672a754313e85`
- 分支：`codex/gpu13-canary-20260804-plan-relaxation-01`
- 模型：`gemini-3.7-flash`
- 并发：10
- case：10 条，5 条预期 `fake`、5 条预期 `real`
- run：
  `/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/unified-react-v1-gemini37-train-smoke-20260828-p10`

## 结果

- 10/10 轨迹正常结束；
- 10/10 strict trace audit 通过；
- 工程错误：0；
- provider error：0；
- provider 重试事件：3；
- 标签正确：7/10；
- 平均耗时：415.6 秒/条；
- 平均工具动作：9.7；
- 平均模型请求：23.8；
- 最大单次 Gemini 输入：61,715 tokens；
- 最大单条累计请求输入：424,186 tokens。

新结构已被真实调用验证：

- `perceive_scene` 和 `ocr_with_position` 是 ReAct 可选工具，不是固定预执行；
- 每轮都是一个 `unified_react` thought 加一个 native tool call；
- 没有旧的 Planning、Query Replan 或 Route Replan stage；
- 10/10 轨迹均保存了可读 thought；
- `trajectory_sft.jsonl` 生成 10 条，包含 `<think>` 和原生 Qwen `<tool_call>`。

## 质量问题

这批 smoke 不足以恢复全量 rollout。三条错误标签均不是工程错误：

1. 两条 `pipeline_generated_refuted` 的公开运行输入只有图片。其 gold
   target claim 包含图像中不可见的具体地点/事件语境；Agent 只能从图片诱导
   一个较宽的视觉命题，无法知道隐藏的地点 claim。这是数据任务定义与
   `image-only` 输入契约不一致，不应通过向 Agent 泄漏 private gold 解决。
2. 一条 `web_crawled_refuted` 需要事实核查来源才能得到 `fake`，但当前
   source-access policy 不允许把“AI 生成/真实性”作为检索路线。该样本在当前
   image-only、来源策略下没有可执行的决定性反证路线。

因此当前应把问题分开处理：

- `unified-react-v1` 的工程链路可以继续修；
- 这批 label 错误不能直接归因于 Gemini 3.7 或网络；
- 需要先按任务模式拆分数据：可从图片恢复的开放世界调查样本，和必须提供
  外部 claim 的 image-plus-claim 样本；
- 在此之前不恢复 8,490 条全量 teacher rollout。

## 评分层修复

本次 smoke 还发现过程评分的一个确定性误报：bootstrap 的两个视觉工具也被
计入“最终判断后的动作”。但 `investigation_state.action_count` 从第一次真实
调查动作才开始计数，导致每条正常的路线耗尽轨迹都出现：

```text
post_determination_action_count = 2
stop_quality = 0
```

评分器现已只对非 bootstrap 的 unified ReAct 调查动作计算该指标。该修复需要
本地测试、提交并重新部署后再用真实 smoke 验证。

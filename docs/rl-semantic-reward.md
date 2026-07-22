# RL 轨迹综合评分与标准 GRPO 契约

## 角色边界

Qwen 独立完成 on-policy 调查。Gemini 只在轨迹结束后作为冻结的离线评审，不生成
Qwen rollout、不参与搜索、不建议下一步，也不读取 evaluator-private gold。

```text
同一题目
  -> G 个完全隔离的 Qwen 完整 episode（默认评测为 1，RL 建议 G=4）
  -> canonical trace + strict audit
  -> 每条轨迹恰好一次 Gemini 综合盲评
  -> ifv-semantic-reward-v2
  -> 私有 gold 的确定性正确性对齐
  -> 同题 GRPO group：每条完整轨迹一个 scalar reward
  -> rLLM / veRL 的标准 GRPO
```

`case_id` 是原始题目；`prompt_group_id` 表示同一个题目、权重和配置下的一组采样；
`episode_id` 是其中一次独立调查；`rollout_index` 从 0 开始。派生的 episode ID 从不替代
数据管线的 `case_id`，private gold 也只按原始 `case_id` 连接。

## 一次 Gemini 综合盲评

一条未命中缓存的轨迹只发出一个 standalone Gemini Interactions 请求，固定使用
`ifv-semantic-trajectory-blind-v1`。评审看到：

- 受控分辨率的原图；
- ImageClaim 的文本；
- 原始、带 ID 的 Evidence 摘录与来源信息；
- 已接受的 action → tool observation 轨迹，以及确定性记录的信息增益；
- 可引用的 Evidence ID 与 turn ID。

评审**看不到** Qwen hidden thinking、旧对话、完整网页、private gold、最终 verdict、
Claim status、Evidence stance、Finding summary 或 rejected output。最终 Judgment action
也不进入盲评包。

返回字段同时覆盖独立 `real|fake|unclear` 判断、证据充分性、搜索方向、调查进展、
Evidence 使用、可见的信念更新和总体过程质量。它引用的 Evidence/turn ID 会被确定性
代码核验；无效引用使语义审计失败。artifact 保存模型、prompt 版本、token 使用、
Interaction ID 和请求/响应 hash，但不保存图片 base64。

## Reward 与 GRPO

私有 gold 必须在同组所有 rollout 结束后才加载。确定性代码随后比较 Qwen verdict、
Gemini 的独立 verdict 和 gold；Gemini 自身永远拿不到 gold。

训练标量遵循结果优先：

```text
工程错误 / strict audit 失败 / 缺少 post-rollout 正确性  -> mask
错误 verdict                                             -> 0.0
正确 verdict                                             -> 0.5 + 0.5 × quality
quality = 0.5 × evidence_quality + 0.5 × overall_process_quality
```

因此任意正确轨迹的 reward 都不低于 0.5，任何错误轨迹为 0；Gemini 的过程评价只用于
细排正确轨迹，不能让“写得漂亮的错误调查”胜过正确结论。错误但工程有效的轨迹仍留在
同题 RL group 中，供标准 GRPO 形成组内相对信号。

每条完整轨迹仅有一个 scalar reward。这里不实现 turn reward、turn-aware estimator、
CW-GRPO 或自定义 advantage；rLLM/veRL 对同一 `prompt_group_id` 内的 scalar rewards
执行自身的标准组内标准化，并把 episode advantage 用于该轨迹的全部 policy token。
全同分、有效成员少于 2、缺 trace 或 mask 的组明确标为不可训练。

## 运行顺序

```powershell
# 1. 生成独立 episode；训练采样通常用 G=4
python -m src.eval.run_eval `
  --benchmark D:\release\runtime_input\cases.jsonl `
  --rollouts-per-case 4 `
  --base-sampling-seed 1729 `
  --output-dir D:\runs\qwen-g4

# 2. 每个 episode 恰好一次 Gemini 综合盲评
python -m src.eval.score_semantic_reward `
  --run-dir D:\runs\qwen-g4 `
  --output-dir D:\runs\qwen-g4\semantic_rewards

# 3. 在 training 独立环境中生成 ledger 与标准 GRPO group
cd training
ifv-training build-run-rewards `
  --semantic-artifacts D:\runs\qwen-g4\semantic_rewards `
  --deterministic D:\runs\qwen-g4\post_rollout_rewards.jsonl `
  --rollout-members D:\runs\qwen-g4\rollout_groups.jsonl `
  --profile configs\rl\semantic-reward-v5.json `
  --ledger-output D:\runs\qwen-g4\reward_ledgers.jsonl `
  --group-output D:\runs\qwen-g4\grpo_groups.jsonl
```

`rollout_groups.jsonl` 记录 episode 身份、seed、trace 与题目分组；
`post_rollout_rewards.jsonl` 仅在 rollout 后记录 private-gold 对齐的确定性字段；
`grpo_groups.jsonl` 是框架可消费的同题分组标量 reward 契约。
当前唯一可用的 reward profile 是 `training/configs/rl/semantic-reward-v5.json`。

冻结开发 20 例及其所有派生 episode、trace、artifact 和 ledger 只能用于评估与回归，
绝不进入 SFT、RL、教师数据或合成训练数据。实现上，`development_subset` 的成员会带
`training_prohibited=true`，training CLI 会把整个 group 标为
`training_prohibited_source`，而不是依赖人工筛选。

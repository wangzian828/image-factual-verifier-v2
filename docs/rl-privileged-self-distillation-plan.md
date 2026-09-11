# IFV Privileged Self-Distillation / RL 实施计划

更新日期：2026-09-11

## 目标

在不改变 IFV runtime、Evidence ownership、private gold 隔离和 Qwen 原生输出格式的前提下，引入 Privileged Self-Distillation（PSD）：

```text
Qwen 无 hint rollout
  -> 定位当前策略自己的可修复失败点
  -> privileged repair side 生成短 hint
  -> 带 hint 重跑同一个失败状态
  -> 确定性 verifier 验证修复
  -> teacher: state + hint
     student: state
  -> top-k KL 蒸馏局部修复动作
```

PSD 是 verifier-gated 的局部蒸馏，不是把 Gemini 成功轨迹直接当作 RL 正样本。

## 组件分工

| 组件 | 责任 |
| --- | --- |
| Qwen policy | 连续 ReAct 动作选择与最终 Judgment |
| IFV runtime | 工具执行、raw history 持久化、机械预算和严格审计 |
| repair proposer | 读取失败轨迹和私有修复信息，生成不带完整答案的短 hint |
| repair verifier | 验证局部动作和完整 episode 是否真正恢复 |
| PSD target builder | 构造无 hint student prefix、带 hint teacher prefix、completion 和 top-k target |
| PSD trainer | 对局部 assistant action 做 top-k KL；不训练 tool observation |
| GRPO | 保留为独立对照，并在 PSD 后用于 warm-start polish |

repair proposer 可以看到 private gold；Qwen 部署策略和最终 student prefix 永远不能看到 private gold。hint 只存在于 teacher 侧，不能写入 student prompt、普通 rollout context 或测试集。

## 第一阶段：数据契约和离线 target

新增 `ifv-psd-*` 数据契约，和现有 SFT、GRPO ledger 分开：

- repair row：一个经过验证的局部失败修复；
- preservation row：原策略已经正确完成的无 hint 行为；
- target row：实际 tokenizer IDs 和 teacher top-k 分布；
- manifest：来源文件 hash、target 数量、split、上下文长度、top-k、准入/拒绝统计。

准入规则：

- 只接受 `train`；
- `test`、`training_prohibited`、工程错误全部拒绝；
- 第一版主训练只接受 `causal_episode_pass`；
- `local_pass_downstream` 先进入 holdout/低权重实验；
- hint 泄漏、缺少真实 token IDs、student/teacher prefix 不一致时 fail closed；
- top-k 缓存未完成的 target 只能标记 `pending_topk`，不能直接训练。

当前已实现：

- `training/ifv_training/psd.py`
- `training/ifv_training/psd_candidates.py`
- `ifv-training audit-psd-hints`
- `ifv-training build-psd-targets`
- `ifv-training build-psd-candidates`
- `ifv-training materialize-psd-topk`
- `ifv-training collect-psd-topk`
- hint 泄漏审计；
- 从 postprocessed Qwen rollout 的真实 `policy_input` / `policy_action` / `interaction_id`
  建立 repair seed、base-pass preservation、工程错误重跑和拒绝账本；
- 未保存完整 Qwen token capture 的旧 trace 不可作为 PSD 数据，单独进入
  `token_capture_requeue`，不能靠重新编码推断成“真实 on-policy prefix”；
- repair/preservation target 构造；
- top-k target 结构校验；
- top-k cache 与 target/token hash 的严格绑定；缓存不完整或错配时不产出可训练
  `targets.jsonl`；
- Qwen native chat 在显式设置 `IFV_CAPTURE_POLICY_TOKENS=1` 时，写入实际 prompt
  token IDs、completion token IDs 和采样 logprob；默认关闭，旧 Gemini trace 不会被
  误认为具备 PSD token 数据；
- target manifest 和 rejection ledger。

## 第二阶段：真实 policy rollout 和失败定位

从 SFT checkpoint 启动 Qwen policy，不使用 Gemini 轨迹代替 Qwen rollout。

每个训练 case：

1. 生成同 prompt 的 `G=4` 条独立 rollout，记录真实 prompt token IDs、completion token IDs、sampling logprob、模型版本和 runtime commit。
2. 完整 episode 结束后，才连接 private gold。
3. 从第一个可修复失败点开始定位：
   - protocol/schema；
   - ReAct 路线无效或重复；
   - ReAct 没有按最新原始观察调整路线；
   - ReAct 对工具结果的范围或成功状态解释错误；
   - 只有前序 raw observation chain 完整时，才定位 Judgment。
4. provider、网络、工具超时和 runtime 损坏不进入 PSD，自动重跑。

失败定位必须使用真实 `policy_input` 和 `interaction_id`，不能从 canonical state 重新拼出一个不同的伪 prefix。

## 第三阶段：hint 生成和修复验证

每个失败点先生成 4 个 L1/L2 hint，按最弱、最短优先尝试。

hint 不允许包含：

- `real/fake` 正确标签；
- 完整 private target fact；
- exact query、URL、Evidence ID、image hash；
- 完整工具调用及参数；
- gold、ground truth、expected verdict 等内部字段。

验证分两层：

```text
local_pass
  当前 stage / ReAct step 恢复

causal_episode_pass
  冻结 self-teacher 在相同 prefix 上保留短 hint，完成整条正确 episode 并通过审计
```

只有 `causal_episode_pass` 进入第一版主 repair target。每个 case 可以保留多个失败点，但同一个失败点只保留最短的首个成功 hint。
额外的无 hint student continuation 只用于确认/诊断原始失败，不是 repair 准入条件。

## 第四阶段：PSD 训练

第一版只训 policy，不训 perception；优先采用 `separate_vlm`，让 policy 输入保持文本化观察和工具结果。

```text
L_psd = batch_mean(sum_over_target_tokens(
  row_weight * CE(teacher_topK(action | state + hint),
                  student(action | state))
))

L_total = L_psd + λ_preserve * L_preserve
```

正式 Qwen3.5 配置按上游公开 recipe 固定：

- top-K：20；
- LoRA rank：32；
- learning rate：`4e-5`；
- 32 个 unique target / optimizer step；
- 5 epochs，gradient clipping 1.0，seed 0；
- 8 卡 SP8、128K；
- repair/preservation 都必须存在；按上游最终公开配置逐 target 等权，默认每个
  repair target 和每个 preservation assistant-step target 权重均为 1.0，不再做
  两类 aggregate row mass 的 1:1 重平衡；
- 每个 repair site 第一版只训练第一个修复 assistant action；
- system、user、tool observation 不计 loss；
- ReAct 保留 Qwen 原生 `<think>` / `<tool_call>` 格式，不自行发明协议。

ms-swift 是后端替换，不改变 PSD 的 target 语义和公开优化器 recipe。正式训练前仍必须用真实 target 包完成 8 卡一步 memory probe，验证该后端替换在 A100 40GB 上的容量和吞吐。

## 第五阶段：对照和多轮

固定同一批训练 case 和 held-out RL-dev，比较：

1. SFT checkpoint；
2. 标准 terminal-reward GRPO；
3. PSD-only；
4. PSD 后 fresh rollout 再 PSD；
5. PSD warm-start GRPO。

每轮 PSD 都必须重新 rollout。旧模型的失败状态不能无限复用，否则会产生 exposure-bias plateau。

最终推荐顺序：

```text
SFT
  -> PSD round 1
  -> fresh rollout
  -> PSD round 2
  -> fresh GRPO
```

先不要直接做 PSD+GRPO interleave；必须先把 PSD-only、GRPO-only 的效果和工程稳定性分开测清楚。

## 评估和停止条件

每个 checkpoint 同时记录：

- verdict accuracy；
- strict audit pass rate；
- engineering error rate；
- protocol-valid rate；
- Evidence chain complete rate；
- early correct judgment / final-only judgment；
- base-pass preservation regression；
- repair local pass rate；
- causal episode pass rate；
- hint level、hint 泄漏率和 target token 长度。

最终测试集 1684 条不参与 rollout、hint、SFT 或 RL。RL-dev 从训练集内部按 case 划出，不和训练 case 重叠。

## 当前状态与剩余实证门禁

代码侧已经具备：真实 Qwen rollout candidate 投影、verifier-guided failure
localization、Gemini privileged hint constructor、冻结 Qwen continuation、严格 episode
准入、完整 hinted teacher episode 生成、零 provider replay 的离线 verifier 收口、
可恢复的 vLLM forced-token top-20 collector、target/datum preflight、ms-swift
sparse loss、SP8/128K profile、训练监控和 checkpoint resume。

剩余工作依赖正式数据和空闲 GPU，不是继续补一套训练语义：

1. 用真实训练 case 完成 5-case repair smoke；
2. 对 verifier 通过的 repair/preservation 包完成 top-20 collection；
3. 用同一包完成 8 卡 SP8 一步 memory probe；
4. 再做 20–50 case 行为 smoke 和 300 case pilot；
5. pilot 无工程错误且 held-out 指标通过后才扩大到完整训练集。

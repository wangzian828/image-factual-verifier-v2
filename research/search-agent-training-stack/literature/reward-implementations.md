# Search Agent 评分实现核查

本页只记录官方仓库中实际参与训练的 reward。论文评测指标、离线 benchmark
分数和框架理论能力不等同于训练 reward。

## 已核对实现

| 项目 | 核对提交 | 训练 reward | Reward 落点 |
|---|---|---|---|
| Search-R1 | `598e61b` | 最终答案 EM；可选格式分。官方 format recipe 中 retrieval reward 为 0 | 最后一个有效 response token |
| ReCall/ReSearch | `aaf16b3` | 最终答案 token-F1；格式正确但答案错误给 0.1 | 终局 response |
| DeepResearcher | `82c6dc2` | 最终答案 F1；标签格式错误为 -1 | 终局 response |
| rLLM | `cd9ea0e` | 内置 SearchReward 为最终答案 EM/F1 | 轨迹 reward 广播到可训练 action tokens |
| WebAgent-R1 | `354f5c1` | WebArena 环境 evaluator/LLM judge 的任务完成分，可与其他 reward function 加权 | 完成整条环境 rollout 后做组内归一化 |
| R-Search | `d7038ca` | 答案 F1 + 格式 + “所选原始证据能否回答问题” | 终局 response |
| DEFAME/AVeriTeC | 本地 `D:\DEFAME` | 标签正确性与 reference QA evidence 的 METEOR/Hungarian 匹配 | 仅离线评测，不是训练 reward |

### Search-R1

实际代码：

- `verl/utils/reward_score/qa_em.py`
- `verl/utils/reward_score/qa_em_format.py`
- `verl/trainer/main_ppo.py`
- `verl/trainer/main_ppo_format.py`

默认实现从最后一个 `<answer>` 提取答案，与 gold 做规范化 EM。format 版本可给
结构分、最终格式分和 retrieval 分，但公开的 GRPO/PPO format 脚本把
`retrieval_score=0`。reward 仍只写到最后一个有效 response token。

### ReCall 与 DeepResearcher

实际代码：

- ReCall: `src/verl/utils/reward_score/re_call.py`
- DeepResearcher: `verl/utils/reward_score/format_and_f1.py`

二者仍以最终答案 F1 为主。ReCall 给正确格式但错误答案 0.1；DeepResearcher
对格式错误给 -1。它们不评价每次搜索方向，也不判断 Evidence→Finding 的语义链。

### rLLM

实际代码：

- `rllm/rewards/search_reward.py`
- `rllm/workflows/workflow.py`
- `rllm/trainer/verl/transform.py`

内置 SearchReward 仍是最终答案 EM/F1。框架能保存 step reward，但当前 veRL
transform 的常用 broadcast 模式以 `trajectory.reward` 生成统一 advantage，传播到
该轨迹的全部可训练 action token；工具/观察 token 由 mask 排除。对于非前缀连续的
独立阶段，transform 会拆成多个 segment，但仍按 trajectory UID 共享同一 advantage。

因此 IFV 的真实接入点应是 episode/trajectory reward join，而不是把一个 terminal
step 文件视为已经接入 trainer。

### WebAgent-R1

实际代码：

- `Train/verifiers/verifiers/envs/webarena_env.py`
- `Train/verifiers/verifiers/trainers/grpo_env_trainer.py`

WebAgent-R1 在 rollout 结束后调用 WebArena evaluator，检查最终网页状态、URL、
页面内容或终局回答；随后对多个 reward function 加权并做 GRPO 组内归一化。它是
真实环境 outcome reward，不是逐步评价模型思考。

### R-Search

实际代码：

- `src/verl/utils/reward_score/qa_f1.py`
- `scripts/evidence.yaml`
- `src/verl/trainer/main_ppo.py`

这是本次最相关的实现。Agent 在终局输出 `<original_evidence>`；冻结的
Llama-3.2-3B-Instruct 只看到问题和这批所选原始证据，并从中回答问题。教师不看
gold，随后确定性 F1 比较器再把教师答案与 gold 比较。总 reward 由答案 F1、证据
可回答性和两项格式分相加，最后仍写在终局 token。

它评价的是一个可验证的中间产物，而不是隐藏思维过程：

```text
Agent 选择的原始 Evidence
  -> 冻结 verifier 仅凭 Evidence 回答问题
  -> verifier 答案与 gold 比较
  -> evidence reward
```

## 跨项目结论

1. 成熟 Search Agent 通常不细排所有正确轨迹。正确且证据足够的轨迹同分是正常
   现象；全同分 group 对 GRPO 没有梯度，需要数据动态过滤或改用更合适的估计器。
2. 公开实现很少对每一步搜索做 LLM 过程评分。所谓多奖励，多数仍是终局答案、
   格式、环境成功或终局提交的证据包。
3. 最可信的“过程信号”是可验证中间产物，而不是让教师阅读完整 thinking 后主观
   打分。
4. 格式 reward 普遍存在，但只应约束协议。把格式、搜索次数或长度当成主要 reward
   容易产生奖励投机。
5. 搜索基础设施故障必须 mask；公开代码里把异常直接记零的做法不适合 IFV 的真实
   网络工具环境。

## 对 IFV 当前 scorer 的影响

当前 Gemini blind/aware packet 仍可能包含 basis 之外的 Evidence。教师可以凭
Agent 没有选择的材料恢复正确 verdict，因而高估轨迹质量。claim status 还混合了
“画面可见内容”和“现实世界是否成立”，已证明不能作为 reward。

下一版评分应拆成三层：

```text
1. outcome
   post-rollout classification_correct；gold 不进入 Qwen 或 Gemini 上下文

2. selected-basis recoverability
   Gemini 只看待核查命题、selected visual anchors、basis 选中的原始 Evidence
   不看其他 Evidence，不看 Qwen verdict，不看 policy-generated Finding 总结
   Gemini 输出 real | fake | insufficient
   rollout 结束后再与 gold 比较

3. protocol/runtime
   strict trace audit + fatal engineering mask
```

Finding 可另做 Evidence→Finding entailment 诊断，但在完成真实排序校准前不进入标量。
搜索次数、长度、停止时机也先记录，不直接塑形。

## 接入约束

- Gemini 在 rollout 结束后异步运行，不能把结果回灌 Qwen。
- Reward artifact 继续保存多维分量与 token usage。
- rLLM gateway 按 episode ID 回连标量到 `trajectory.reward`；由当前 transform 将同一
  advantage 广播给该轨迹的 action tokens。
- 独立阶段 prompt 会形成多个 segment，必须验证它们仍共享同一 trajectory UID。
- 现有 `ifv-rllm-reward-record-v1` 只是边界记录，不等于已经完成 trainer callback。
- 冻结 20 例只做开发评测，不能进入训练 group。

## 下一轮校准

1. 冻结 Qwen checkpoint、runtime commit、采样参数和工具环境。
2. 每题同配置采样 K=4–8 条轨迹。
3. 同时记录 outcome、selected-basis recoverability、strict audit 和 fatal mask。
4. 人工盲审 top/middle/bottom，检查 basis reward 是否真的偏好证据链更完整的轨迹。
5. 先比较 REINFORCE/RLOO 与 GRPO 的有效 group 比例，再决定优化器。


# Image Factual Verifier Training

`training/` 是 v4 主仓库内保留 Git 历史的独立 Python 子项目。它负责成熟框架上的
Qwen SFT、RL 数据契约、reward 审计、checkpoint/serving 清单；不复制 v4 Agent 状态机，
不向 Agent 提供 evaluator-private gold。

根仓库与本目录必须使用独立环境：

```powershell
# 根目录：runtime
python -m pytest -q

# 本目录：training
cd training
python -m pip install -e ".[dev]"
python -m pytest -q
```

## SFT

SFT 使用经过审计的阶段样本，不把整段长期对话或历史 hidden reasoning 当作训练目标。
正式数据必须来自非冻结评测集，并经过图像、工具 schema、阶段归属、loss mask、split 和
去重门禁。优先采用成熟的 ms-swift + DeepSpeed/FSDP；本项目不自建训练循环。

## RL：标准 GRPO

同一题目生成多条完全隔离的完整 episode。Qwen 在每次参数更新后重新 on-policy 采样；
Gemini 仅作为冻结的离线评审。它对每条 episode 最多发起一次综合盲评，不生成 Qwen
轨迹，也不读取 private gold。

确定性代码在所有 rollout 结束后才按原始 `case_id` 接入 private gold：工程或 strict-audit
失败会 mask；错误轨迹 reward 为 `0.0`；正确轨迹在 `0.5–1.0` 内由 Evidence 质量与
总体调查过程质量细排。每条完整轨迹只有一个 scalar reward；rLLM/veRL 对同题
`prompt_group_id` 内的 scalar rewards 执行标准 GRPO。这里不实现逐步 reward、CW-GRPO、
turn-aware advantage 或自定义 estimator。

```powershell
# 在根仓库先得到 qwen-g4 的 rollout / semantic artifacts
cd training
ifv-training build-run-rewards `
  --semantic-artifacts D:\runs\qwen-g4\semantic_rewards `
  --deterministic D:\runs\qwen-g4\post_rollout_rewards.jsonl `
  --rollout-members D:\runs\qwen-g4\rollout_groups.jsonl `
  --profile configs\rl\semantic-reward-v5.json `
  --ledger-output D:\runs\qwen-g4\reward_ledgers.jsonl `
  --group-output D:\runs\qwen-g4\grpo_groups.jsonl
```

`grpo_groups.jsonl` 只携带每条完整 episode 的 raw scalar reward、policy step IDs 和
mask/方差诊断；框架负责 advantage。所有 `development_subset`（包括冻结 20 例）和其
派生 trace、artifact、ledger 都会被标记为 `training_prohibited`，不得进入 SFT、RL、
教师数据或合成数据。

## 本地门禁

```powershell
python -m pytest -q
python -m compileall -q ifv_training scripts
git diff --check
```

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

Qwen3.5 训练工程摘录见
[`docs/qwen35-engineering-notes-from-modern-genai-bilibili.md`](docs/qwen35-engineering-notes-from-modern-genai-bilibili.md)，
应用计划见
[`../docs/superpowers/plans/2026-08-03-qwen35-training-engineering-application-plan.md`](../docs/superpowers/plans/2026-08-03-qwen35-training-engineering-application-plan.md)；
两者记录 FSDP2、sequence parallel、padding-free、CPU offload locality、dataset cache
和 checkpoint memory gates 的可迁移经验与落地顺序。

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

## Frozen teacher SFT export

scoring release 的教师数据固定使用根仓库导出的 frozen teacher dataset：

```powershell
python scripts/trajectory/export_dataset.py `
  --run-dir D:\runs\teacher-run-a `
  --case-split D:\training\sft-v1\case-split\case_split.jsonl `
  --eligibility-dir D:\training\sft-v1\eligibility `
  --semantic-reward-dir D:\training\sft-v1\semantic_rewards `
  --minimum-accepted-cases 40 `
  --output-dir D:\training\sft-v1\accepted-dataset
```

此模式不随机重分 split；每个 case 最多选一条通过 strict、structured
eligibility 和 gold-free semantic 三层门禁的最高质量 rollout，并保留
`accepted_episodes.jsonl`、`episode_metadata.jsonl` 与门禁 artifact ID。随后转换为
标准 ms-swift 多模态数据：

```powershell
python -m ifv_training convert-accepted-perception `
  --input D:\training\sft-v1\accepted-dataset `
  --output D:\training\sft-v1\ms-swift-perception
python -m ifv_training convert-policy `
  --input D:\training\sft-v1\accepted-dataset `
  --output D:\training\sft-v1\ms-swift-policy
```

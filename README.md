# Image Factual Verifier Training

Visual Fact Discrepancy Agent v4 的独立 Qwen3.5 训练与部署工程。正式学生模型是 `/gsdata/home/wza/models/Qwen3.5-9B`；旧 Qwen3-VL 只保留为诊断基线。

本仓库负责：

- 把通过审计的 teacher 轨迹转换为 provider-neutral 的阶段样本；
- 校验图像、工具 schema、阶段归属、loss mask、split、去重与 20 例泄漏；
- 用成熟框架搭建全参数多模态 SFT 与 Agent RL；
- 生成 checkpoint、环境和 serving 审计清单；
- 通过 OpenAI-compatible endpoint 接入 v4 runtime。

本仓库不实现第二套 Agent 状态机，不读取 evaluator-private gold，也不把冻结 20 例用于训练。

## 环境边界

Serving、SFT、RL 是三套独立环境。当前 serving 使用 Python 3.11 + vLLM nightly；SFT 与 RL 使用独立的 Python 3.12 + ms-swift 4.4.2 环境，不继承历史 `qwen3vl`、`ifv-agent` 或旧训练环境。详细命令见 [gpu-13 运维文档](docs/gpu13.md)。

训练样本以一次真实阶段决策为单位，而不是整条长对话：

```text
perception | planning | react | evidence_decision |
discrepancy_decision | reflection | judgment
```

Qwen checkpoint 在 RL 更新后重新采样 on-policy 轨迹；Gemini 只作为冻结的离线教师、过程评分器和回归基线，不生成 Qwen 的 on-policy rollout。

## RL 评分链

Agent 仓库先从完成后的 canonical trace 生成 `ifv-semantic-reward-v1`。Gemini 的第一阶段
不知道 Qwen 的 verdict、Claim status 和 Evidence stance；第二阶段才检查记录的 basis，
并执行 verdict swap 与 Evidence dropout 反事实探针。

本仓库只消费冻结 artifact：校验内容 hash，合并确定性门禁和可选的 rollout 后
`classification_correct`，保留所有 reward 维度，再按 profile 组合标量。provider、工具或
运行时 fatal error 会被 mask，不会作为策略负奖励。

```powershell
ifv-training audit-semantic-reward `
  --input D:\runs\semantic_rewards\case_x.semantic_reward.json `
  --strict

ifv-training reward-ledger `
  --semantic-artifact D:\runs\semantic_rewards\case_x.semantic_reward.json `
  --deterministic D:\runs\case_x.post_rollout.json `
  --profile configs\rl\semantic-reward-v1.json `
  --output D:\runs\reward_ledgers\case_x.json

ifv-training export-reward `
  --ledger D:\runs\reward_ledgers\case_x.json `
  --framework rllm `
  --output D:\runs\reward_records\case_x.rllm.json
```

`--framework verl` 生成对应的 terminal-token reward record。两个导出器只做框架边界映射，
不复制 v4 状态机或训练循环。冻结 20 例产生的 artifact 和 ledger 仍只用于开发评估，禁止
进入 SFT、RL 或教师数据。

## 本地门禁

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q ifv_training scripts
git diff --check
```

服务器源码只通过 GitHub fast-forward；模型、数据和运行产物只写 `/gsdata`。

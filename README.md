# Image Factual Verifier Training

Visual Fact Discrepancy Agent v4 的独立 Qwen3-VL 学生训练工程。唯一正式学生是 gpu-13
已有的 `/gsdata/home/wza/models/Qwen3-VL-8B-Thinking`。

本仓库负责：

- 将通过审计的 provider-neutral teacher release 转成精确 Qwen3-VL processor/template 输入；
- 校验图像、tool schema、阶段、loss mask、split、去重和 20 例泄漏；
- 使用成熟框架完成 LLM、ViT、aligner 均不冻结的全参数多模态 SFT；
- 保存、审计、恢复和重新部署 checkpoint；
- 通过 OpenAI-compatible serving profile 接入 v4 runtime；
- 通过外部 model gateway 接入 v4 环境做 on-policy Agent RL。

本仓库不实现第二套 Agent 状态机，也不读取 evaluator-private gold。

## 数据流

```text
accepted Gemini teacher traces
  -> ifv-policy-dataset-v2 provider-neutral stage rows
  -> exact Qwen3-VL processor/template
  -> full-parameter SFT
  -> Qwen base / SFT / RL canary and frozen 20-case evaluation
  -> v4 runtime gateway + on-policy Agent RL
```

一条训练样本对应一次真实阶段决策，而不是整条长聊天：

```text
perception | planning | react | evidence_decision |
discrepancy_decision | reflection | judgment
```

## 框架策略

- Serving 首选 gpu-13 已实测的 LMDeploy 0.13.0；若协议门禁失败，按记录的最小复现切换
  现有 vLLM、SGLang 或官方 Transformers 服务。
- SFT 首选 ms-swift 4.4.1 + DeepSpeed ZeRO-3；只有 processor、全参数梯度、保存、恢复和
  重新加载实测通过后才冻结版本。
- Agent RL 首选 rLLM gateway + veRL；不兼容时比较 ms-swift RL、OpenRLHF、AReaL，
  但所有 rollout 必须由 v4 runtime 持有工具和状态。

rLLM/veRL 及其 rollout engine 使用独立 RL 环境。当前 rLLM 主线固定提交为
`cd9ea0eca082bcc701ee4df464446288ccb02c1f`；它要求的新 PyTorch/Transformers/vLLM
组合不得安装进已验证的 `qwen3vl` serving 环境。普通 Agent serving 与 RL rollout worker
是两个 profile，后者还必须通过 token ID/logprob 对齐探针。

框架可以按服务器实测调整，模型、数据契约和验收标准不能随之降低。

## 本地门禁

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q ifv_training scripts
git diff --check
```

## gpu-13 边界

- checkout：`/gs/home/wza/projects/image-factual-verifier-training`
- 数据与产物：`/gsdata/home/wza/image-factual-verifier-v2-data/training`
- 模型：`/gsdata/home/wza/models/Qwen3-VL-8B-Thinking`
- 只允许物理 GPU `4,5,6,7` 中实时空闲的卡，最多四张
- 不修改 `ifv-agent`，不终止其他用户进程
- 20 例在四例 base canary 和配置冻结前不运行，且永不进入训练数据

服务器命令见 [docs/gpu13.md](docs/gpu13.md)。统一执行计划位于 runtime 仓库的
`docs/superpowers/plans/2026-07-21-qwen3-vl-8b-deployment-and-training-infrastructure.md`。

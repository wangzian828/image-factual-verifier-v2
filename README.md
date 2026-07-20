# Image Factual Verifier Training

Visual Fact Discrepancy Agent v4 的独立 Qwen3.5 学生训练工程。运行时、训练器和
数据管线通过版本化文件契约交互，不互相导入 Python package。

主线：

```text
Qwen3.5-4B：快速环境/协议/3-step 设施检查
Qwen3.5-9B：正式 serving、全参数多模态 SFT、Agent RL
```

本仓库负责：

- 将 v4 provider-neutral teacher release 转成 Qwen3.5/ms-swift 格式；
- 校验图像、tool schema、阶段、loss mask 和 split；
- 使用 ms-swift + DeepSpeed ZeRO-3 做全参数多模态 SFT；
- 保存、审计和恢复 checkpoint；
- 生成 vLLM serving profile；
- 通过 rLLM/veRL gateway 接入现有 v4 runtime 做 Agent RL。

本仓库不实现第二套 Agent 状态机，也不读取 evaluator-private gold。

## 数据流

```text
accepted Gemini teacher traces
  -> ifv-policy-dataset-v2
  -> provider-neutral stage rows
  -> exact Qwen3.5 processor/template
  -> full-parameter SFT
  -> vLLM
  -> v4 Qwen canary / 20-case evaluation
  -> rLLM gateway + veRL on-policy RL
```

一条 SFT 样本对应一次真实阶段决策，不是整条聊天历史：

```text
perception | planning | react | evidence_decision |
discrepancy_decision | reflection | judgment
```

## 本地门禁

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q ifv_training scripts
git diff --check
```

## gpu-13 边界

- checkout：`/gs/home/wza/projects/image-factual-verifier-training`
- 大文件：`/gsdata/home/wza/image-factual-verifier-v2-data/training`
- 模型：`/gsdata/home/wza/models/Qwen3.5-4B|Qwen3.5-9B`
- 只允许物理 GPU `4,5,6,7`，最多四张；每次先检查进程所有者
- `ifv-agent` 不安装、升级或删除任何 Qwen 训练/serving 依赖

详细执行命令见 [docs/gpu13.md](docs/gpu13.md)。统一决策见 runtime 仓库的
`docs/superpowers/plans/2026-07-21-qwen35-9b-deployment-and-training-infrastructure.md`。

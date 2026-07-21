# Image Factual Verifier Training

Visual Fact Discrepancy Agent v4 的独立 Qwen3-VL 训练与部署工程。正式学生模型是服务器已有的 `/gsdata/home/wza/models/Qwen3-VL-8B-Thinking`。

本仓库负责：

- 把通过审计的 teacher 轨迹转换为 provider-neutral 的阶段样本；
- 校验图像、工具 schema、阶段归属、loss mask、split、去重与 20 例泄漏；
- 用成熟框架搭建全参数多模态 SFT 与 Agent RL；
- 生成 checkpoint、环境和 serving 审计清单；
- 通过 OpenAI-compatible endpoint 接入 v4 runtime。

本仓库不实现第二套 Agent 状态机，不读取 evaluator-private gold，也不把冻结 20 例用于训练。

## 环境边界

Serving、SFT、RL 是三套独立环境。当前 serving 使用从零创建并锁定的 Python 3.11 + vLLM 0.11.2 环境，不继承历史 `qwen3vl`、`ifv-agent` 或训练环境。详细命令见 [gpu-13 运维文档](docs/gpu13.md)。

训练样本以一次真实阶段决策为单位，而不是整条长对话：

```text
perception | planning | react | evidence_decision |
discrepancy_decision | reflection | judgment
```

Qwen checkpoint 在 RL 更新后重新采样 on-policy 轨迹；Gemini 只作为冻结的离线教师、过程评分器和回归基线，不生成 Qwen 的 on-policy rollout。

## 本地门禁

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q ifv_training scripts
git diff --check
```

服务器源码只通过 GitHub fast-forward；模型、数据和运行产物只写 `/gsdata`。

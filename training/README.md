# IFV Training

`training/` 提供数据转换、数据审计、奖励账本、GRPO 分组和 checkpoint
清单。Agent runtime 不在这里执行。

## 本地检查

```powershell
cd training
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q ifv_training scripts
git diff --check
```

## SFT 数据链路

```text
teacher rollout
  -> strict trace audit + frozen SFT judge
  -> accepted-teacher-release
  -> accepted-dataset
  -> convert-policy / convert-accepted-perception
  -> live-runtime causal audit
  -> real ms-swift processor verification v3
  -> raw-data launch gate
  -> swift sft
```

policy 行是一条完整 Qwen Agent episode，格式为：

```text
system
user: <image> + task
assistant: <think>...</think>
tool_call: {"name":"...","arguments":"{...}"}
tool_response: ...
...
assistant: <think>...</think><answer>...</answer>
```

转换结果只使用 `tools`、`messages`、`images`。正常监督消息只有 `role` 和
`content`；若历史动作不满足当前可执行契约，则保留整条轨迹，并仅给对应 thought
和 `tool_call` 写 `loss=false`。不写 `channel` 或 `chat_template_kwargs`。目标
Qwen/ms-swift template 负责生成 labels。

```powershell
python -m ifv_training convert-policy `
  --input <accepted-dataset> `
  --output <ms-swift-policy>

# 只有已经转换过、无法回到 canonical trace 的旧交付包才使用此入口。
python -m ifv_training repair-policy-contract `
  --input <legacy-ms-swift-policy> `
  --output <ms-swift-policy-v4>

python -m ifv_training convert-accepted-perception `
  --input <accepted-dataset> `
  --output <ms-swift-perception>

python -m ifv_training audit --strict --input <ms-swift-policy>
python -m ifv_training audit --strict --input <ms-swift-perception>
python scripts/probe/verify_ms_swift_agent_dataset.py `
  --model <target Qwen checkpoint> `
  --policy-dir <ms-swift-policy> `
  --perception-dir <ms-swift-perception> `
  --max-context 131072 `
  --max-pixels 262144 `
  --truncation-strategy raise `
  --padding-free true `
  --sequence-parallel-size 8 `
  --loss-scale ignore_empty_think `
  --enable-thinking false `
  --add-non-thinking-prefix false `
  --image-max-token-num 1024 `
  --output <processor-verification.json>
```

验证脚本会实际调用 processor，确认原生 `<think>` 在 labels 中、工具调用和工具
结果进入输入、图片未丢失，同时逐一确认正常动作被监督、`loss=false` 动作确实被
掩码，并使用与训练完全相同的 template 参数检查上下文长度。
报告记录输入 JSONL 的绝对路径、大小和 SHA-256，供训练启动门禁绑定。

直接用 raw JSONL 调用 `run_sft.sh` 时必须设置：

```text
IFV_PROCESSOR_VERIFICATION=<processor-verification.json>
```

启动器会再次执行 dataset `manifest.json` 严格审计，并核对 train/validation
文件哈希、模型路径和全部 template 参数。任何数据或 profile 在 processor 审计后发生
变化都会在创建 GPU 训练进程前失败。`run_teacher_sft_pipeline.sh --run-training`
会自动传递该报告。

## RL

标准 GRPO 入口消费 `post_rollout_rewards.jsonl`：

```powershell
python -m ifv_training build-run-rewards `
  --deterministic <run>/post_rollout_rewards.jsonl `
  --rollout-members <run>/rollout_groups.jsonl `
  --profile <profile.json> `
  --ledger-output <run>/reward_ledgers.jsonl `
  --group-output <run>/grpo_groups.jsonl
```

reward ledger 负责记录工程审计、正确性和过程质量；训练框架负责同 prompt 的组内
归一化。本项目不实现第二套 trainer。

当前仓库的可执行 RL 入口只有 `training/scripts/rl/run_mock_grpo.sh`，用于验证
ms-swift/Gym/GRPO 工程接线，不是正式在线 Agent RL。正式 RL 仍需独立实现并验证
真实 runtime gateway、同 prompt rollout 分组、reward ledger 绑定和训练资源配置。

RL bootstrap 会生成独立的 `environment-preflight.json`，核对固定依赖、`pip
check`、CUDA/GPU、Qwen3.5 上下文、ms-swift RL 参数契约，以及 `swift rlhf` 和
`vllm serve` 两个 CLI。旧的 ready 文件不能单独证明环境可用；新 ready 文件保存
本次通过报告的 SHA-256，任何环境修复后都必须重新生成报告和摘要。

## PSD

PSD 使用当前 round-start Qwen 的冻结副本生成 top-20 分布，Gemini 等外部模型只可
构造 privileged hint，不能提供 self-teacher logits。训练入口固定为 8 卡 SP8、128K、
LoRA rank 32；不再保留 SP1 profile：

```bash
python -m ifv_training collect-psd-topk \
  --targets <target-package>/targets.jsonl \
  --serving-profile <serving-profile.json> \
  --round-start-checkpoint-manifest <checkpoint-manifest.json> \
  --output-dir <server-topk-cache> \
  --topk 20

python -m ifv_training materialize-psd-topk \
  --targets <target-package>/targets.jsonl \
  --cache <server-topk-cache>/teacher_topk_cache.jsonl \
  --output-dir <materialized-target-package> \
  --topk 20
```

top-20 collector 对冻结 Qwen 强制输入原始 token IDs，并读取 completion 位置的
prompt logprobs；不会把文本 decode 后再 encode。它绑定 serving profile、round-start
checkpoint manifest 和逐 target token hash，每成功一条立即持久化；同一输出目录重跑
只补失败/缺失 target。

repair driver 默认逐次执行“提示 → 完整轨迹 → judge → 根据失败反馈修改提示”，
默认最多 6 次完整续跑、12 次提示提议，成功立即停止；已完成请求有绑定缓存，
`--resume` 不会为了审核失败而重采样相同判断。提示生成器不读取私有标准答案，
私有参考只用于验收和提示泄漏检查。完整历史保存在每个 `rounds/round-NN/` 下。
需要旧的单次/离线验证方式时显式指定 `--search-mode single`。
repair driver 会在每次尝试中生成完整 hinted teacher episode；局部 task verifier
完成后，用 `ifv-training finalize-psd-repair-run --run-dir <同一目录> ...` 离线收口。
finalizer 不调用模型或工具，并保留 pre-finalize 备份，避免为工程重试重复成功 case。

先把 materialized targets 转成 datums 并通过 preflight，再启动训练：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
training/scripts/train/run_psd_topk.sh \
  training/configs/models/qwen3.5-9b.env \
  training/configs/psd/qwen3.5-lora-r32-1step-8gpu-sp8-128k-memory-probe.env \
  <psd-package>/datums.jsonl \
  <memory-probe-experiment-id>

training/scripts/train/run_psd_topk.sh \
  training/configs/models/qwen3.5-9b.env \
  training/configs/psd/qwen3.5-lora-r32-5epoch-8gpu-sp8-128k.env \
  <psd-package>/datums.jsonl \
  <production-experiment-id>
```

启动器在加载模型前校验逐 target 权重、数据 hash、top-20、128K/SP8、LoRA 超参和
每步 32 个 unique target，并跑 8-rank CPU 分布式 loss smoke。自定义 loss 直接沿用
ms-swift 的 sequence/ring split 顺序，只 gather 每位置标量 loss，不 gather 全词表
logits；fp32 cross-entropy 按 active completion position 分块重算。真实 production
仍必须先通过空闲 8 卡的一步 memory probe。

## 环境边界

- SFT 和 RL 使用独立环境。
- 128K SFT 还使用独立的 long-context 环境；通过 `sft-long` bootstrap 从源码构建
  固定版本的 FlashAttention/causal-conv1d，不改写现有 SDPA 或历史环境。
- 所有目标模型变化都要重新跑真实 processor 验证。
- `action_only` 不混入 reasoning policy SFT。
- reasoning policy SFT 使用
  `training/configs/models/qwen3.5-9b.env`，并保持
  `IFV_ADD_NON_THINKING_PREFIX=false`。
- 当前保留 16K production、portable 8K、smoke-noeval，以及 8 卡 128K 的 memory
  probe、10-step canary、11-step resume 三个验收 profile；历史硬件/并发 sweep 可从
  Git 标签 `pre-deep-cleanup-20260910` 恢复。
- evaluator private gold、judge 字段和 provider 内部协议不得进入模型可见数据。
- 本目录不保存 rollout、图片或 checkpoint；这些数据放在服务器 `/gsdata`。
- `run_sft.sh` 默认记录逐卡显存、利用率、温度、功率、进程树 CPU/RSS、步耗时、
  checkpoint 和 watchdog 状态；设置 `IFV_GPU_MEMORY_TARGET_*` 后，资源目标成为
  production gate 的一部分。
- 长上下文 profile 在创建训练进程前还会生成 `environment-preflight.json`，严格核对
  Python/CUDA/核心包版本、CUDA 扩展可导入性、ms-swift template 参数、模型 128K
  上限和所选 8 张 A100 的型号与显存。
- 128K 验收 profile 还拒绝没有 120,000+ token 训练样本的 processor 报告，避免用短
  样本冒充长上下文容量测试。完整的 probe -> canary -> resume 顺序见
  `docs/gpu13.md`。

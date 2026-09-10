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
  -> real ms-swift processor verification
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

转换结果只使用 `tools`、`messages`、`images`。消息只有 `role` 和 `content`；
不写 `loss`、`channel` 或 `chat_template_kwargs`。目标 Qwen/ms-swift template
负责生成 labels。

```powershell
python -m ifv_training convert-policy `
  --input <accepted-dataset> `
  --output <ms-swift-policy>

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
结果进入输入、图片未丢失，并使用与训练完全相同的 template 参数检查上下文长度。
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

## 环境边界

- SFT 和 RL 使用独立环境。
- 128K SFT 还使用独立的 long-context 环境；通过 `sft-long` bootstrap 从源码构建
  固定版本的 FlashAttention/causal-conv1d，不改写现有 SDPA 或历史环境。
- 所有目标模型变化都要重新跑真实 processor 验证。
- `action_only` 不混入 reasoning policy SFT。
- reasoning policy SFT 使用
  `training/configs/models/qwen3.5-9b.env`，并保持
  `IFV_ADD_NON_THINKING_PREFIX=false`。
- 当前只保留 production、portable 8K 和 smoke-noeval 三个经过记录的 SFT profile；
  历史硬件/并发 sweep 可从 Git 标签 `pre-deep-cleanup-20260910` 恢复。
- evaluator private gold、judge 字段和 provider 内部协议不得进入模型可见数据。
- 本目录不保存 rollout、图片或 checkpoint；这些数据放在服务器 `/gsdata`。
- `run_sft.sh` 默认记录逐卡显存、利用率、温度、功率、进程树 CPU/RSS、步耗时、
  checkpoint 和 watchdog 状态；设置 `IFV_GPU_MEMORY_TARGET_*` 后，资源目标成为
  production gate 的一部分。
- 长上下文 profile 在创建训练进程前还会生成 `environment-preflight.json`，严格核对
  Python/CUDA/核心包版本、CUDA 扩展可导入性、ms-swift template 参数、模型 128K
  上限和所选 8 张 A100 的型号与显存。

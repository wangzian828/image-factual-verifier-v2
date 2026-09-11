# gpu-13 部署与训练环境

本文记录当前可复用的部署边界。服务器只通过 GitHub 更新代码；数据、图片、日志和
checkpoint 写入 `/gsdata`，不写入 Git checkout。

## 1. 环境

当前冻结的 SDPA SFT 环境：

```text
Python       3.12
ms-swift     4.4.2
transformers 5.12.1
torch        2.10.0
datasets     4.8.4
环境路径     /gsdata/home/wza/conda/envs/ifv-qwen35-sft-ms-swift442
```

该环境不包含 128K padding-free/SP8 必需的 FlashAttention 与
causal-conv1d CUDA 扩展，不能用于长上下文 profile。历史 32K FSDP2/SP4 实验验证过
以下扩展组合：

```text
flash-attn              2.8.3
flash-linear-attention  0.5.1
causal-conv1d           1.6.2.post1
liger-kernel            0.8.0
```

2026-09-10 已验收的正式 128K 环境：

```text
Python                  3.12.14
ms-swift                4.4.2
transformers            5.12.1
torch                   2.10.0+cu128
flash-attn              2.8.3
flash-linear-attention  0.5.1
causal-conv1d           1.6.2.post1
liger-kernel            0.8.0
环境路径                /gsdata/home/wza/conda/envs/ifv-qwen35-sft-long-ms-swift442-20260910
验收报告                /gsdata/home/wza/image-factual-verifier-v2-data/training/logs/environments/ifv-qwen35-sft-long-ms-swift442/environment-preflight.json
```

验收报告为 `passed=true`、`errors=[]`；两个源码构建的原生扩展最高只要求
`GLIBC_2.14`，兼容 gpu-13 的 glibc 2.28。重建环境时必须使用全新前缀，不修改上述
已验收环境或历史实验环境：

```bash
export IFV_QWEN35_MODEL=/gsdata/home/wza/models/Qwen3.5-9B
export IFV_QWEN35_SFT_LONG_ENV_PREFIX=<new-empty-conda-prefix>
export IFV_TRAINING_DATA_ROOT=<training-data-root>
bash training/scripts/server/bootstrap_qwen35_training_gpu13.sh sft-long
```

gpu-13 的 glibc 为 2.28，长上下文 bootstrap 会固定版本并从源码编译两个 CUDA
扩展。完成后必须存在 `.ifv-qwen35-sft-long-ready`，且
`<training-data-root>/logs/environments/ifv-qwen35-sft-long-ms-swift442/environment-preflight.json`
必须通过；构建失败时不得回退到 SDPA 启动 128K。

模型名称和 checkpoint 路径以实际任务配置为准。换模型时必须用该模型自己的
processor 重新验证，不能只复用旧环境的通过结果。

独立 RL 环境使用
`/gsdata/home/wza/conda/envs/ifv-qwen35-rl-ms-swift442-vllm0221`。其验收报告应位于
`<training-data-root>/logs/environments/ifv-qwen35-rl-ms-swift442-vllm0221/environment-preflight.json`。
`.ifv-qwen35-rl-ready` 保存该报告的 SHA-256；空的历史 marker 不构成验收依据。

## 2. 从 GitHub 更新

```bash
cd /gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01
git fetch origin
git pull --ff-only
```

不要在服务器上直接改代码；临时实验也不要把输出写进 checkout。

## 3. SFT 数据准备

```bash
ENV=/gsdata/home/wza/conda/envs/ifv-qwen35-sft-ms-swift442

"$ENV/bin/python" -m ifv_training convert-policy \
  --input <accepted-dataset> \
  --output <ms-swift-policy>

"$ENV/bin/python" -m ifv_training convert-accepted-perception \
  --input <accepted-dataset> \
  --output <ms-swift-perception>

"$ENV/bin/python" -m ifv_training audit --strict --input <ms-swift-policy>
"$ENV/bin/python" -m ifv_training audit --strict --input <ms-swift-perception>

"$ENV/bin/python" training/scripts/probe/verify_ms_swift_agent_dataset.py \
  --model <qwen-checkpoint> \
  --policy-dir <ms-swift-policy> \
  --perception-dir <ms-swift-perception> \
  --max-context 131072 \
  --max-pixels 262144 \
  --truncation-strategy raise \
  --padding-free true \
  --sequence-parallel-size 8 \
  --loss-scale ignore_empty_think \
  --enable-thinking false \
  --add-non-thinking-prefix false \
  --image-max-token-num 1024 \
  --output <processor-verification.json>
```

processor 验证必须确认：

- policy 的原生 `<think>` token 在 assistant labels 中；
- `tool_call` 和 `tool_response` 没有从编码输入丢失；
- 图片被 processor 接收；
- 每行存在可训练 assistant token；
- 编码长度没有超过目标模型上下文上限。

raw JSONL 训练还必须把本次报告传给启动器：

```bash
export IFV_PROCESSOR_VERIFICATION=<processor-verification.json>
```

训练启动器会 fail-closed 核对报告、dataset manifest、train/validation SHA-256、
模型路径和 template 参数。processor 审计后修改任何输入文件或训练 profile 都必须
重新审计。

128K probe/canary 还要求 processor 报告中的精确边界计数证明训练 split 至少有一条
编码后不短于 120,000 token 的真实样本。报告同时记录 32K、64K、96K、120K 和
128K 边界两侧的行数及最近真实行，供容量排查使用。只设置
`IFV_MAX_LENGTH=131072`、但所有样本都很短，不算 128K 容量验证。

## 4. 数据格式

policy 行：

```text
system
user: <image> + task
assistant: <think>...</think>
tool_call: {"name":"...","arguments":"{...}"}
tool_response: ...
...
assistant: <think>...</think><answer>{...}</answer>
```

顶层只使用 `tools`、`messages`、`images`。消息只含 `role` 和 `content`。
不添加 `loss`、`channel` 或 `chat_template_kwargs`；labels 由目标 ms-swift
template 生成。

## 5. 训练边界

- reasoning policy SFT 保留 provider 原生 thought。
- 没有可读 thought 的轨迹不伪装成 reasoning SFT。
- perception SFT 是独立图片观察任务，不混入 policy episode。
- `action_only` 不进入 reasoning policy SFT。
- private gold、judge 字段、provider wire 和内部缓存不得进入模型可见数据。
- 当前代码收尾不启动 8,490 条全量教师 rollout。

## 6. GPU 与并发

生产任务只使用任务明确分配且确认空闲的 GPU。并发由任务启动参数控制，不读取或
复用历史遗留的 `GEMINI_EVAL_MAX_CONCURRENCY` 作为隐式覆盖。

所有长任务必须记录：

- Git commit；
- 模型和 processor 版本；
- 数据集 manifest SHA-256；
- 并发、重试和超时设置；
- 输出目录和审计报告路径。

## 7. 训练监控与显存验收

`run_sft.sh` 自动启动 watchdog 和进程树资源采样，持续写入：

```text
<training-data-root>/logs/<experiment-id>/resource-samples.jsonl
<training-data-root>/logs/<experiment-id>/resource-summary.json
<training-data-root>/logs/<experiment-id>/monitor-latest.json
<training-data-root>/logs/<experiment-id>/monitor.log
<training-data-root>/logs/<experiment-id>/profile.json
```

8 卡 128K canary 使用以下验收目标：

```text
IFV_GPU_MEMORY_TARGET_MIN_MIB=36000
IFV_GPU_MEMORY_TARGET_MAX_MIB=38912
IFV_GPU_MEMORY_MAX_IMBALANCE_MIB=1024
IFV_GPU_UTILIZATION_TARGET_MIN_PERCENT=85
```

低于显存目标表示仍有空间可用于减少重计算；超过上限表示安全余量不足。任何一张卡
越界或卡间峰值差超过限制都会写入 watchdog 告警，配置了目标的 run 若未通过资源
验收，也不能通过最终 production gate。资源报告同时记录各卡显存比例、利用率、温度、
功率和活跃期显存差。显存目标用于调优，不替代 OOM、NaN、验证、checkpoint 与 resume
检查。

## 8. 8 卡 128K 验收顺序

三个 profile 都固定为 FSDP2 + SP8：8 张卡共同处理一条序列，数据并行度为 1；使用
FlashAttention、padding-free、gradient checkpointing 和分块交叉熵。不要直接把
memory probe 当成正式训练结果。

推荐使用一次性验收入口。它会先生成静态容量/时间预算，再逐级执行并验证数据血缘、
资源目标、eval、完整 checkpoint 和从 step 10 到 step 11 的真实恢复；已有通过的 gate
可复用，发现同名未完成目录时会停止，避免换目录重复成功 case：

```bash
bash training/scripts/train/run_128k_acceptance.sh \
  training/configs/models/qwen3.5-9b.env \
  <train.jsonl> <validation.jsonl> <processor-verification.json> \
  <run-prefix> <capacity-plan.json>
```

`capacity-plan.json` 只给出全参训练状态的分片显存下界；activation、collective 和
allocator 的实际占用必须由一步 120K+ 真样本探针证明。预计总时长也只接受已经通过
production gate 且同为 128K/world8/SP8 的 profile，不会拿 16K 或不同并行方式外推。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export IFV_ALLOWED_GPU_IDS=0,1,2,3,4,5,6,7
export IFV_OMP_NUM_THREADS=1
export IFV_CUDA_HOME=<long-context-env>
export IFV_TRAINING_DATA_ROOT=<training-data-root>
export IFV_PROCESSOR_VERIFICATION=<processor-verification.json>

# 1. 一步、无 eval、无 checkpoint，只测真实 120K+ 样本的峰值与首步稳定性
bash training/scripts/train/run_sft.sh \
  training/configs/models/qwen3.5-9b.env \
  training/configs/sft/qwen3.5-full-1step-8gpu-fsdp2-sp8-flash-128k-memory-probe.env \
  <train.jsonl> <validation.jsonl> <unique-memory-probe-id>

# 2. 十步、带 eval 和完整训练状态 checkpoint
bash training/scripts/train/run_sft.sh \
  training/configs/models/qwen3.5-9b.env \
  training/configs/sft/qwen3.5-full-10step-8gpu-fsdp2-sp8-flash-128k-canary.env \
  <train.jsonl> <validation.jsonl> <unique-canary-id>

# 3. 从 checkpoint-10 恢复并前进一步，写入新的实验目录
bash training/scripts/train/run_sft.sh \
  training/configs/models/qwen3.5-9b.env \
  training/configs/sft/qwen3.5-full-11step-8gpu-fsdp2-sp8-flash-128k-resume.env \
  <train.jsonl> <validation.jsonl> <unique-resume-id> \
  <canary-checkpoint-10>
```

第一阶段的 `profile.json` 必须明确是 `smoke_only`，它只决定是否值得进入第二阶段。
第二、三阶段都必须通过环境、数据、资源、验证、checkpoint I/O 和 resume 门禁。正式
epoch 数、是否减少 activation 重计算以及最终吞吐目标，要等最终数据的 token 长度分布
和上述实测结果后再冻结；当前不提供未经真实 128K canary 验证的全量训练 profile。

# gpu-13 Qwen 训练与服务

Gemini 环境 `ifv-agent` 不安装、升级或删除任何包。Qwen 使用：

```text
ifv-qwen-train
ifv-qwen-serve
```

ms-swift 固定为 `4.4.1`。训练入口直接调用 `swift sft` 和 `swift rlhf`；
没有自定义 Trainer。

环境采用 Python 3.11。该选择与 Gemini 的 `ifv-agent` 无关；它是独立环境，
并符合 ms-swift 4.4.1 当前安装建议。训练环境不会安装进 `ifv-agent`。

## 1. 选择 GPU

```bash
cd /gs/home/wza/projects/image-factual-verifier-training
bash scripts/server/select_idle_gpus.sh
nvidia-smi
export CUDA_VISIBLE_DEVICES=<确认空闲卡1>,<确认空闲卡2>
```

候选列表只按显存和利用率排序，必须再看进程所有者。所有项目进程使用：

```bash
export OMP_NUM_THREADS=1
```

## 2. 环境

先检查 gpu-13 驱动，再显式指定对应的 PyTorch wheel：

```bash
export IFV_TORCH_INDEX_URL=<官方 PyTorch CUDA wheel index>
export IFV_TORCH_PACKAGES='torch==<version> torchvision==<version>'
bash scripts/server/bootstrap_gpu13.sh
```

脚本不会接触 `ifv-agent`。它只安装：

- `ms-swift==4.4.1`
- DeepSpeed（训练环境）
- vLLM（训练 rollout 与服务）
- Qwen VL utilities

## 3. 数据转换

```bash
conda run --no-capture-output -n ifv-qwen-train \
  python -m ifv_training convert-policy \
  --input "$POLICY_DATASET" \
  --output "$DERIVED_ROOT/policy-v1"

conda run --no-capture-output -n ifv-qwen-train \
  python -m ifv_training audit \
  --input "$DERIVED_ROOT/policy-v1" \
  --strict
```

真实训练前必须用实际 Qwen processor 验证模板、loss 和图像 token：

```bash
conda run --no-capture-output -n ifv-qwen-train \
  python scripts/probe/ms_swift_template.py \
  --model /gsdata/home/wza/models/Qwen3-VL-8B-Thinking \
  --dataset "$DERIVED_ROOT/perception-v1/train.jsonl"
```

## 4. 双卡 LoRA smoke

训练集和验证集必须非空。当前 20 条 preview 太小，只用于基础设施 smoke，
不能用于证明训练有效。

```bash
conda run --no-capture-output -n ifv-qwen-train \
  bash scripts/train/run_sft.sh \
  configs/models/qwen3-vl-8b-thinking.env \
  configs/sft/qwen3-vl-lora-smoke.env \
  "$TRAIN_JSONL" \
  "$VAL_JSONL" \
  qwen3-vl-8b-ifv-sft-smoke-001
```

恢复：

```bash
conda run --no-capture-output -n ifv-qwen-train \
  bash scripts/train/run_sft.sh \
  configs/models/qwen3-vl-8b-thinking.env \
  configs/sft/qwen3-vl-lora-smoke.env \
  "$TRAIN_JSONL" \
  "$VAL_JSONL" \
  qwen3-vl-8b-ifv-sft-resume-001 \
  "$CHECKPOINT"
```

## 5. 导出和服务

```bash
conda run --no-capture-output -n ifv-qwen-train \
  bash scripts/export/merge_lora.sh \
  "$CHECKPOINT" \
  qwen3-vl-8b-ifv-sft-smoke-001 \
  "$DERIVED_ROOT/manifest.json"

export CUDA_VISIBLE_DEVICES=<服务卡1>,<服务卡2>
conda run --no-capture-output -n ifv-qwen-serve \
  bash scripts/serve/start_vllm.sh \
  "$EXPORTED_MODEL" \
  ifv-qwen-sft-smoke \
  8899 \
  2 \
  "$CHECKPOINT_MANIFEST"
```

服务探针：

```bash
conda run --no-capture-output -n ifv-qwen-serve \
  python scripts/probe/openai_endpoint.py \
  --base-url http://127.0.0.1:8899/v1 \
  --image "$CANARY_IMAGE"
```

服务只监听 loopback，不占用 Jupyter 8333，也不读取 Gemini 环境变量。

## 5.1 Mixed curriculum

只有五个阶段的 train/validation 文件都非空时才启动：

```bash
conda run --no-capture-output -n ifv-qwen-train \
  bash scripts/train/run_curriculum_sft.sh \
  configs/models/qwen3-vl-8b-thinking.env \
  configs/sft/qwen3-vl-lora-smoke.env \
  "$DERIVED_ROOT/perception-v1" \
  "$DERIVED_ROOT/policy-v1" \
  qwen3-vl-8b-ifv-curriculum-001
```

该入口使用 ms-swift 原生 `--interleave_prob`，不会复制数据或实现自定义
DataLoader。

## 6. GRPO smoke

仅在 SFT checkpoint 通过真实 Agent canary 后运行。第一阶段是 ms-swift
`GYMScheduler` 的确定性 mock 环境，不访问网络：

```bash
conda run --no-capture-output -n ifv-qwen-train \
  bash scripts/rl/run_mock_grpo.sh \
  configs/models/qwen3-vl-8b-thinking.env \
  configs/rl/qwen3-vl-grpo-mock.env \
  qwen3-vl-8b-ifv-grpo-mock-001
```

真实工具环境以后仍通过 `external_plugins` 接入，不替换 ms-swift 的 GRPO、
vLLM rollout、DeepSpeed 或 checkpoint 实现。

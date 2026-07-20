# gpu-13 Qwen3.5 部署与训练

只使用物理 GPU `4,5,6,7` 中实时空闲的卡，最多四张。所有命令先执行：

```bash
cd /gs/home/wza/projects/image-factual-verifier-training
bash scripts/server/select_idle_gpus.sh
nvidia-smi
export OMP_NUM_THREADS=1
```

不得修改 `ifv-agent`，不得终止其他用户进程。

## 1. 环境

gpu-13 已验证 PyTorch 2.13/cu130。建立独立 Python 3.12 环境：

```bash
export IFV_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
export IFV_TORCH_PACKAGES='torch==2.13.0 torchvision==0.28.0'
bash scripts/server/bootstrap_gpu13.sh
```

生成环境清单并检查 Qwen3.5 关键依赖：

```bash
conda run -n ifv-qwen35-serve python -m ifv_training environment-manifest \
  --repo-root "$PWD" \
  --output /gsdata/home/wza/image-factual-verifier-v2-data/training/logs/qwen35-serve-environment.json

conda run -n ifv-qwen35-sft python -c \
  'import torch, transformers, swift; print(torch.__version__, transformers.__version__, swift.__version__)'
```

## 2. 下载模型

4B 只做快速启动，9B 是正式学生：

```bash
conda run -n ifv-qwen35-serve python scripts/server/download_model.py \
  --model Qwen/Qwen3.5-4B \
  --local-dir /gsdata/home/wza/models/Qwen3.5-4B \
  --manifest /gsdata/home/wza/image-factual-verifier-v2-data/training/models/Qwen3.5-4B.json

conda run -n ifv-qwen35-serve python scripts/server/download_model.py \
  --model Qwen/Qwen3.5-9B \
  --local-dir /gsdata/home/wza/models/Qwen3.5-9B \
  --manifest /gsdata/home/wza/image-factual-verifier-v2-data/training/models/Qwen3.5-9B.json
```

## 3. 4B serving 快速门禁

```bash
export CUDA_VISIBLE_DEVICES=4
conda run --no-capture-output -n ifv-qwen35-serve \
  bash scripts/serve/start_vllm.sh \
  /gsdata/home/wza/models/Qwen3.5-4B \
  ifv-qwen35-4b-base 8899 1 32768 false
```

另一个控制会话运行：

```bash
conda run --no-capture-output -n ifv-qwen35-serve \
  python scripts/probe/openai_endpoint.py \
  --base-url http://127.0.0.1:8899/v1 \
  --image "$SMOKE_IMAGE" \
  --rounds 8
```

4B 通过后将模型路径换成9B，原样重跑。完整20例只使用9B。

## 4. Processor 与全参数 SFT smoke

先生成不属于评测集的合成图像数据：

```bash
SMOKE_ROOT=/gsdata/home/wza/image-factual-verifier-v2-data/training/datasets/qwen35-smoke-v1
conda run -n ifv-qwen35-sft python scripts/server/build_smoke_dataset.py \
  --output "$SMOKE_ROOT"

conda run -n ifv-qwen35-sft python scripts/probe/ms_swift_template.py \
  --model /gsdata/home/wza/models/Qwen3.5-4B \
  --dataset "$SMOKE_ROOT/train.jsonl"
```

4B 3-step 快速检查：

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
conda run --no-capture-output -n ifv-qwen35-sft \
  bash scripts/train/run_sft.sh \
  configs/models/qwen3.5-4b.env \
  configs/sft/qwen3.5-full-3step.env \
  "$SMOKE_ROOT/train.jsonl" \
  "$SMOKE_ROOT/validation.jsonl" \
  qwen35-4b-full-3step-001
```

9B 按 1-step、3-step、20-step 递进；每次使用新的 experiment ID。20-step 指优化器
步数，不是20例评测。完成后执行 checkpoint audit、显式恢复3步和vLLM重新加载。

## 5. 运行产物

```text
/gsdata/home/wza/image-factual-verifier-v2-data/training/
  datasets/
  derived/
  checkpoints/
  exports/
  rollouts/
  rewards/
  logs/
  models/
```

每次运行必须保存 environment manifest、Git commit、GPU映射、stdout/stderr、checkpoint
manifest和恢复来源。服务只监听loopback，不占用Jupyter 8333。

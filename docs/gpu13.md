# gpu-13 Qwen3-VL-8B-Thinking 部署与训练

只使用物理 GPU `4,5,6,7` 中实时空闲的卡，最多四张。每个任务先执行：

```bash
cd /gs/home/wza/projects/image-factual-verifier-training
bash scripts/server/select_idle_gpus.sh
nvidia-smi
export OMP_NUM_THREADS=1
```

不得修改 `ifv-agent`，不得终止其他用户进程。

## 1. 已验证资产

```text
model: /gsdata/home/wza/models/Qwen3-VL-8B-Thinking
env:   /gs/home/wza/anaconda3/envs/qwen3vl
stack: Python 3.10.18, torch 2.6.0+cu124, transformers 4.57.6,
       qwen-vl-utils 0.0.14, LMDeploy 0.13.0, Ray 2.55.1
```

先生成只读环境清单：

```bash
conda run -n qwen3vl python -m ifv_training environment-manifest \
  --repo-root "$PWD" \
  --output /gsdata/home/wza/image-factual-verifier-v2-data/training/logs/qwen3vl-serve-environment.json
```

gpu-13 当前没有可用外网代理。不要重复下载已有权重，也不要在网络未恢复时无意义地反复
安装。SFT 环境需要 ms-swift/DeepSpeed；有可用代理或本地 wheelhouse 后执行：

```bash
# serving 可完全复用已安装依赖
bash scripts/server/bootstrap_gpu13.sh serve

# SFT 需要可用代理或本地 wheelhouse
# export IFV_WHEELHOUSE=/gsdata/home/wza/wheels
bash scripts/server/bootstrap_gpu13.sh sft
```

该脚本从现有 `qwen3vl` 克隆 Python 3.10/CUDA 基础，建立隔离的
`ifv-qwen3vl-serve` 与 `ifv-qwen3vl-sft`，不污染 `ifv-agent`。

## 2. LMDeploy serving 门禁

8B BF16 先使用一张空闲 A100。LMDeploy 显式使用 PyTorch engine 和 `uni` executor，避免
历史默认 Ray executor 因节点整体 RSS 达到 95% 而杀死 worker。

```bash
export CUDA_VISIBLE_DEVICES=4
conda run --no-capture-output -n qwen3vl \
  bash scripts/serve/start_lmdeploy.sh \
  /gsdata/home/wza/models/Qwen3-VL-8B-Thinking \
  ifv-qwen3-vl-8b-thinking 8899 1 32768 agent
```

另一控制会话运行：

```bash
conda run --no-capture-output -n qwen3vl \
  python scripts/probe/openai_endpoint.py \
  --base-url http://127.0.0.1:8899/v1 \
  --model ifv-qwen3-vl-8b-thinking \
  --image "$SMOKE_IMAGE" \
  --rounds 8
```

必须分别记录文本、图像、结构化 JSON、一次 native tool call、`role=tool` continuation、
八轮短链、并发、取消和干净退出。该 Thinking checkpoint 的 chat template 固定以
`<think>` 开始，`enable_thinking=false` 不改变模板；因此不做伪 hybrid A/B，而由
LMDeploy `reasoning-parser=default` 将 reasoning 与 canonical content 隔离。若 LMDeploy
协议不通过，保存最小复现后按统一计划测试成熟替代框架。

RL rollout worker 使用单独角色启动以开放 raw logprobs；普通 Agent serving 不承担训练采样：

```bash
export CUDA_VISIBLE_DEVICES=4
conda run --no-capture-output -n qwen3vl \
  bash scripts/serve/start_lmdeploy.sh \
  /gsdata/home/wza/models/Qwen3-VL-8B-Thinking \
  ifv-qwen3-vl-8b-thinking-rollout 8899 1 32768 rollout
```

## 3. Processor 与全参数 SFT smoke

先生成不属于 20 例的合成小数据：

```bash
SMOKE_ROOT=/gsdata/home/wza/image-factual-verifier-v2-data/training/datasets/qwen3-vl-smoke-v1
conda run -n ifv-qwen3vl-sft python scripts/server/build_smoke_dataset.py \
  --output "$SMOKE_ROOT"

conda run -n ifv-qwen3vl-sft python scripts/probe/ms_swift_template.py \
  --model /gsdata/home/wza/models/Qwen3-VL-8B-Thinking \
  --dataset "$SMOKE_ROOT/train.jsonl"
```

四张卡同时空闲后依次运行 1、3、20 optimizer step；每次使用新的 experiment ID：

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
conda run --no-capture-output -n ifv-qwen3vl-sft \
  bash scripts/train/run_sft.sh \
  configs/models/qwen3-vl-8b-thinking.env \
  configs/sft/qwen3-vl-full-1step.env \
  "$SMOKE_ROOT/train.jsonl" "$SMOKE_ROOT/validation.jsonl" \
  qwen3vl-8b-full-1step-001
```

从 1-step checkpoint 显式恢复并使用 3-step profile。门禁必须证明 LLM、ViT、aligner 都有
有限非零梯度，checkpoint 包含优化器、调度器、RNG 和 manifest，独立进程能重新加载并通过
文本、图像、JSON 与工具协议。20-step 是设施稳定性检查，不是 20 例评测。

## 4. Agent RL

先用 mock tool gym 验证 gateway、多轮 rollout、tool provenance、reward、logprob/loss mask、
取消、checkpoint 和恢复，再接真实 v4 工具。Qwen checkpoint 每次更新后重新采样；Gemini
只做冻结的离线过程评分，不生成 on-policy rollout。

当前首选是 rLLM gateway + veRL。只有 gpu-13 实测协议与显存门禁通过后才冻结；不通过时
按统一计划比较成熟替代框架，不在本仓库自写 trainer 或第二套 Agent 状态机。

RL 环境独立安装 `requirements/rl.txt`，不得复用或升级 `qwen3vl` serving 环境。rollout
endpoint 在进入 gateway 前必须通过：

```bash
python scripts/server/probe_rl_compatibility.py \
  --base-url http://127.0.0.1:8899/v1 \
  --model ifv-qwen3-vl-8b-thinking-rollout \
  --output "$IFV_TRAINING_DATA_ROOT/logs/rl-serving-compatibility.json"
```

native tool call、prompt/completion token IDs、逐 token logprobs 及长度对齐缺一不可。普通
LMDeploy serving 即使 Agent 协议通过，也不能在该探针失败时冒充 RL rollout worker。

## 5. 产物

```text
/gsdata/home/wza/image-factual-verifier-v2-data/training/
  datasets/ derived/ checkpoints/ exports/ rollouts/ rewards/ logs/ models/
```

每次运行保存 environment manifest、Git commit、物理/逻辑 GPU 映射、stdout/stderr、
checkpoint manifest、恢复来源和停止命令。服务只监听 loopback，不占用 Jupyter 8333。

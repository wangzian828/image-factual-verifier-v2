# Qwen / ms-swift 部署与验证

本文件描述在任意新服务器上接收 IFV 数据并验证 Qwen/ms-swift 训练输入。机器专属
CUDA、驱动、挂载点、代理和模型路径必须由部署者配置，不能从现有服务器照搬。

## 1. 环境边界

建议基础版本：

```text
Python       3.12
ms-swift     4.4.2
transformers 5.12.1
datasets     4.8.4
```

`torch` 与 CUDA 必须按目标服务器驱动和目标 Qwen checkpoint 的支持矩阵安装。
环境文件使用 `configs/runtime.env.example`，复制到：

```text
~/.config/image-factual-verifier/runtime.env
```

至少配置：

```text
IFV_DATA_ROOT
IFV_MODEL_ID
IFV_OMP_NUM_THREADS=1
```

API key、代理和本地模型 endpoint 按实际用途补充。真实配置文件权限设为 `600`，
不得提交到 Git。

## 2. 快捷安装和预检

```bash
git clone <repository-url> image-factual-verifier
cd image-factual-verifier
mkdir -p ~/.config/image-factual-verifier
cp configs/runtime.env.example \
  ~/.config/image-factual-verifier/runtime.env
# 编辑 runtime.env
chmod 600 ~/.config/image-factual-verifier/runtime.env

source scripts/server/ifv_env.sh
scripts/server/bootstrap_runtime.sh
scripts/server/run_ifv.sh python scripts/server/doctor.py --json
```

训练环境单独创建，避免和 rollout runtime、vLLM 服务环境混装：

```bash
python3.12 -m venv .venv-sft
source .venv-sft/bin/activate
python -m pip install --upgrade pip
# 先按本机驱动安装 torch/CUDA wheel
python -m pip install "ms-swift==4.4.2" \
  "transformers==5.12.1" "datasets==4.8.4"
python -m pip install -e ./training
python -m pip check
python scripts/server/doctor.py --require-training --json
```

## 3. 训练包结构

正式包至少包含：

```text
ms-swift-policy/
ms-swift-perception/
accepted-dataset/
accepted-release/
audits/
MANIFEST.json
```

policy 一行是一条完整 episode。顶层只使用：

```json
{
  "tools": "[...]",
  "messages": [{"role": "...", "content": "..."}],
  "images": ["data:image/jpeg;base64,..."]
}
```

ReAct assistant turn 保留教师原生 `<think>`；工具调用由 `tool_call`、工具结果由
`tool_response` 表示。图片 marker 和 `images` 数量严格相等。private gold、judge
字段、runtime 内部状态、原始 HTML 和 provider wire 数据不进入模型可见行。

## 4. 严格审计

先做结构审计：

```bash
python -m ifv_training audit --strict --input <package>/ms-swift-policy
python -m ifv_training audit --strict --input <package>/ms-swift-perception
```

再用最终要训练的真实 checkpoint 做 processor 编码：

```bash
python training/scripts/probe/verify_ms_swift_agent_dataset.py \
  --model "$IFV_MODEL_ID" \
  --policy-dir <package>/ms-swift-policy \
  --perception-dir <package>/ms-swift-perception \
  --output <package>/audits/processor-verification.json \
  --max-context 131072
```

探针必须确认：

1. 每行可编码且长度未超上限；
2. `<think>` 位于可训练 assistant labels；
3. JSON 工具调用被 Qwen 模板渲染为真实 function/parameter token；
4. 工具结果进入编码输入；
5. 所有图片被 processor 接收；
6. 每行存在非空训练 label。

换 checkpoint、processor、chat template 或 ms-swift 版本后必须重新验证。

## 5. 启动边界

测试集评测入口：

```bash
source scripts/server/ifv_env.sh
scripts/server/start_gemini_eval.sh \
  --benchmark <evaluation-release/runtime_input/cases.jsonl> \
  --output-dir "$IFV_DATA_ROOT/runs/eval/<run-id>" \
  --concurrency <N>

scripts/server/poll_eval.sh <run-id>
```

不带 `evaluation_gold` 的教师轨迹输入使用：

```bash
scripts/server/start_teacher_rollout.sh \
  --benchmark <runtime-cases.jsonl> \
  --output-dir "$IFV_DATA_ROOT/runs/eval/<run-id>" \
  --concurrency <N> \
  --rollouts-per-case 1
```

训练入口继续使用 `training/scripts/train/run_sft.sh`。模型路径从
`IFV_MODEL_ID`/模型 profile 读取，训练数据路径由命令行提供。不要在启动器里写死
用户名、挂载点、模型目录、GPU 编号或代理。

完整新服务器交付流程见
[`portable-deployment-handoff.md`](portable-deployment-handoff.md)。

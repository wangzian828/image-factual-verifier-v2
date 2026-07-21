# gpu-13 部署与训练运维

本页只记录当前有效入口。Serving、SFT、RL 使用彼此独立的环境；任何安装失败都新建环境，不在半成品上继续修补。

## Qwen3-VL serving 基线

当前 serving 基线是：

```text
模型       /gsdata/home/wza/models/Qwen3-VL-8B-Thinking
环境       /gsdata/home/wza/conda/envs/ifv-qwen3vl-vllm0112-locked
引擎       vLLM 0.11.2
Python     3.11
GPU        物理 4、5，TP=2
地址       http://127.0.0.1:8901/v1
上下文     131072 token
reasoning  qwen3
tools      qwen3_xml
schema     xgrammar，在 reasoning 结束后约束最终输出
```

旧 `qwen3vl`、`ifv-agent`、`ifv-qwen3vl-vllm0110` 和 `ifv-qwen3vl-vllm0251` 均不是这套服务的依赖或安装目标。

## 从零构建一次

服务器源码只通过 GitHub 更新：

```bash
cd /gs/home/wza/projects/image-factual-verifier-training
git fetch origin
git pull --ff-only

export http_proxy=http://100.10.1.210:47899
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export NO_PROXY=127.0.0.1,localhost
export CUDA_VISIBLE_DEVICES=4,5
export OMP_NUM_THREADS=1

bash scripts/server/bootstrap_vllm_qwen3vl_gpu13.sh
```

构建脚本具有以下边界：

- 使用 `conda create -p ... python=3.11`，不 clone 旧环境；
- 目标目录一旦存在就拒绝修改；
- 按 `requirements/serve-vllm-qwen3vl.txt` 和 constraints 一次解析安装；
- 执行 `pip check`、精确版本、CUDA、模型/processor/tokenizer、Qwen3-VL registry 和 parser 检查；
- 把 `pip freeze`、SHA256、环境 manifest 写入 `/gsdata/.../training/logs/environments/`；
- 权重已经在本地，构建与启动均设置 Hugging Face 离线模式，不重复下载。

如果构建失败，保留日志和失败目录用于诊断。确认后为下一次构建指定一个全新的 `IFV_VLLM_ENV_PREFIX`；不要重新运行脚本去修改失败目录。

## 启动、查看与停止

启动器会复查 GPU 只能来自 `4,5,6,7` 且当前空闲，只监听 loopback：

```bash
cd /gs/home/wza/projects/image-factual-verifier-training
export CUDA_VISIBLE_DEVICES=4,5
bash scripts/serve/manage_vllm_qwen3vl.sh start
bash scripts/serve/manage_vllm_qwen3vl.sh status
bash scripts/serve/manage_vllm_qwen3vl.sh logs
```

停止时仅向 PID 文件所指、且命令行已核验为本环境/本模型/本端口的进程组发送 SIGTERM：

```bash
bash scripts/serve/manage_vllm_qwen3vl.sh stop
```

脚本不会自动 SIGKILL，也不会按模糊进程名清理其他任务。

## Serving 协议门禁

门禁使用真实 Thinking、图像、复杂 Planning schema 和 native tools，不用 minimal 提示替代真实协议：

```bash
ENV=/gsdata/home/wza/conda/envs/ifv-qwen3vl-vllm0112-locked
SMOKE_IMAGE=/path/to/an/existing/evaluation/image.jpg

"$ENV/bin/python" scripts/probe/vllm_qwen3vl_endpoint.py \
  --base-url http://127.0.0.1:8901/v1 \
  --model ifv-qwen3-vl-8b-thinking-vllm \
  --image "$SMOKE_IMAGE" \
  --rounds 8 \
  --expected-context 131072 \
  --output /gsdata/home/wza/image-factual-verifier-v2-data/training/logs/serving/qwen3vl-vllm0112-gates.json
```

该探针必须同时验证：

1. `/health`、`/v1/models` 和服务公开的 `max_model_len=131072`；
2. 文本与单图请求的 reasoning 非空，且 final content 不含 `<think>`；
3. reasoning 后生成简单 JSON schema；
4. reasoning 后生成完整 `ImageAccountPlanningOutput` 结构；
5. required native tool call 和 `role=tool` continuation；
6. 八次彼此独立的短链均成功。

全部通过后才运行 Queen 的真实 Planning 请求。Queen 无工程错误后，依次运行 Andreea、Pillars、Monarch；此前不启动 20 例。

## SFT 与 RL

SFT 和 RL 不使用上述 serving 环境。它们继续使用各自的成熟框架、requirements 和独立环境。20 例永远只用于冻结评测，不进入 SFT 或 RL 数据。

每个任务只使用物理 GPU `4,5,6,7` 中当时空闲的卡，最多四张；不终止其他用户进程。所有数据、checkpoint、rollout、环境清单与日志写入 `/gsdata`，不写入 Git checkout。

## gpu-13 NCCL 兼容设置

gpu-13 的 R580 驱动与 NCCL 2.27 默认 cuMem host 分配路径存在已实测的 `libcuda.so` 崩溃。服务固定设置 `NCCL_CUMEM_HOST_ENABLE=0`；构建门禁会在物理 GPU 4、5 上执行真实的双 rank NCCL all-reduce，不能用单卡 CUDA import 代替这项检查。

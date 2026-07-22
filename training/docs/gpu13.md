# gpu-13 部署与训练运维

本页只记录当前有效入口。Serving、SFT、RL 使用彼此独立的环境；任何安装失败都新建环境，不在半成品上继续修补。

## Qwen3.5 serving 基线

当前 serving 基线是：

```text
模型       /gsdata/home/wza/models/Qwen3.5-9B
环境       /gsdata/home/wza/conda/envs/ifv-qwen35-vllm-nightly
引擎       vLLM 0.23.1rc1.dev1348+g47f1b47a7
Python     3.11
GPU        物理 4、5，TP=2
地址       http://127.0.0.1:8901/v1
上下文     131072 token
reasoning  qwen3
tools      qwen3_coder
schema     xgrammar，在 reasoning 结束后约束最终输出
```

旧 `qwen3vl`、`ifv-agent` 和所有 Qwen3-VL serving 环境均不是这套服务的依赖或安装目标。

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

bash scripts/server/freeze_vllm_qwen35_gpu13.sh
```

构建脚本具有以下边界：

- 使用 `conda create -p ... python=3.11`，不 clone 旧环境；
- 目标目录一旦存在就拒绝修改；
- 校验已经独立创建的 Qwen3.5 vLLM nightly 环境，不修改它；
- 执行 `pip check`、精确版本、CUDA、模型/processor/tokenizer、Qwen3.5 registry 和 parser 检查；
- 把 `pip freeze`、SHA256、环境 manifest 写入 `/gsdata/.../training/logs/environments/`；
- 权重已经在本地，构建与启动均设置 Hugging Face 离线模式，不重复下载。

如果构建失败，保留日志和失败目录用于诊断。确认后为下一次构建指定一个全新的 `IFV_VLLM_ENV_PREFIX`；不要重新运行脚本去修改失败目录。

## 启动、查看与停止

启动器会复查 GPU 只能来自 `4,5,6,7` 且当前空闲，只监听 loopback：

```bash
cd /gs/home/wza/projects/image-factual-verifier-training
export CUDA_VISIBLE_DEVICES=4,5
bash scripts/serve/manage_vllm_qwen35.sh start
bash scripts/serve/manage_vllm_qwen35.sh status
bash scripts/serve/manage_vllm_qwen35.sh logs
```

停止时仅向 PID 文件所指、且命令行已核验为本环境/本模型/本端口的进程组发送 SIGTERM：

```bash
bash scripts/serve/manage_vllm_qwen35.sh stop
```

脚本不会自动 SIGKILL，也不会按模糊进程名清理其他任务。

## Serving 协议门禁

门禁使用真实 Thinking、图像、复杂 Planning schema 和 native tools，不用 minimal 提示替代真实协议：

```bash
ENV=/gsdata/home/wza/conda/envs/ifv-qwen35-vllm-nightly
SMOKE_IMAGE=/path/to/an/existing/evaluation/image.jpg

"$ENV/bin/python" scripts/probe/vllm_qwen35_endpoint.py \
  --base-url http://127.0.0.1:8901/v1 \
  --model ifv-qwen3.5-9b \
  --image "$SMOKE_IMAGE" \
  --rounds 8 \
  --expected-context 131072 \
  --output /gsdata/home/wza/image-factual-verifier-v2-data/training/logs/serving/qwen35-gates.json
```

该探针必须同时验证：

1. `/health`、`/v1/models` 和服务公开的 `max_model_len=131072`；
2. Thinking 开/关按请求生效，且 reasoning 不污染 final content；
3. reasoning 后生成简单 JSON schema；
4. required native tool call 和 `role=tool` continuation；
5. 八次彼此独立的短链均成功。

全部通过后才运行 Queen 的真实 Planning 请求。Queen 无工程错误后，依次运行 Andreea、Pillars、Monarch；此前不启动 20 例。

## SFT 与 RL

SFT 和 RL 不使用上述 serving 环境。它们继续使用各自的成熟框架、requirements 和独立环境。20 例永远只用于冻结评测，不进入 SFT 或 RL 数据。

Qwen3.5 使用 ms-swift 4.4.2 的成熟 SFT/GRPO 入口，Transformers 锁到其支持范围内的 5.12.1。FLA 与 CUDA 扩展要求 Python 3.12；SFT 与 RL 均从空目录建立，失败环境不原地修补：

```bash
cd /gs/home/wza/projects/image-factual-verifier-training
bash scripts/server/bootstrap_qwen35_training_gpu13.sh sft
bash scripts/server/bootstrap_qwen35_training_gpu13.sh rl
```

默认环境分别为：

```text
/gsdata/home/wza/conda/envs/ifv-qwen35-sft-ms-swift442
/gsdata/home/wza/conda/envs/ifv-qwen35-rl-ms-swift442-vllm0221
```

每个任务只使用物理 GPU `4,5,6,7` 中当时空闲的卡，最多四张；不终止其他用户进程。所有数据、checkpoint、rollout、环境清单与日志写入 `/gsdata`，不写入 Git checkout。

## gpu-13 NCCL 兼容设置

gpu-13 的 R580 驱动与 NCCL 2.27 默认 cuMem host 分配路径存在已实测的 `libcuda.so` 崩溃。服务固定设置 `NCCL_CUMEM_HOST_ENABLE=0`；构建门禁会在物理 GPU 4、5 上执行真实的双 rank NCCL all-reduce，不能用单卡 CUDA import 代替这项检查。

vLLM 的 custom all-reduce 在本机预热时会返回 CUDA `invalid argument`，因此启动参数固定使用 `--disable-custom-all-reduce`，TP 通信统一走上述已验证的 NCCL 路径。

Thinking checkpoint 的原始模板会在 prompt 末尾预填 `<think>`，但 vLLM 0.11.2 的 `qwen3` parser 只解析生成结果中同时存在的 `<think>...</think>`。启动器因此从 checkpoint 原始模板精确派生一份服务模板，仅去掉生成前的 `<think>` 预填，让模型自行生成起始标记；模板及 SHA256 写入服务 profile 目录。正文中出现思考或缺少独立 reasoning 字段都视为门禁失败。

结构化输出固定设置 xgrammar 的 `disable_any_whitespace=true`。该 checkpoint 在允许任意 JSON 空白时会持续生成数千行空白直至 16384 token 长度上限；禁用 grammar 空白只约束序列化形式，不改变 Planning schema 或调查语义。

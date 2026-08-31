# Qwen3.5-9B on gpu-07 V100: research log

## Baseline inventory

- `gpu-07` exposes eight Tesla V100-SXM2 GPUs with 32 GiB each and SM70 compute capability. All eight were confirmed free before the study began.
- The previous `wza` service was a single four-GPU Transformers process: `transformers serve /gsdata/home/wza/models/Qwen3.5-9B --device auto --dtype float16 --no-continuous-batching`. It used GPUs 4-7 and was stopped before this study.
- Existing candidate environments include a Transformers/vLLM 0.8.5 setup, an LMDeploy 0.15.0 setup, and a Qwen3.5 vLLM nightly environment with PyTorch CUDA 13.0 wheels and vLLM 0.23.1rc1.dev1348.

## External implementation survey

- 1Cat-vLLM identifies itself as a V100/SM70-focused vLLM fork. Its public release path ships prebuilt Python 3.12 wheels with bundled SM70 extensions and a V100 FlashAttention backend.
- The screenshot's `1Cat-vLLM-0.0.3.dev10+gcldce8324` is a development build, not a normal upstream vLLM version. The public 1Cat release line has since advanced beyond the early `0.0.3` release.
- The current public 1Cat release selected for preflight is `v1.3.0` with wheel digest `sha256:2bdb14a9c44f83ee6a766d88ed0d85b11390d6f5d65747e8dbe80a8e2d5d63e0`.
- 1Cat documents SM70 AWQ, a V100 FlashAttention path, long-context runtime defaults, and experimental FP8/MTP features. Those features are hypotheses to test; no performance or correctness conclusion is transferred from its published Qwen3.6-27B-AWQ examples to IFV's Qwen3.5-9B checkpoint.

## Initial engineering implications

- The previous V100 service was FP16. The existing gpu-13 launcher uses BF16, which must not be copied blindly into the V100 experiment matrix.
- One model replica per GPU, TP=2 replicas, TP=4 replicas, and an eight-GPU aggregate deployment optimize different workloads. The winning layout will be chosen from measured IFV request mix, not from nominal GPU count.

## 2026-08-31 compatibility gates

- Both public 1Cat wheels were rejected before serving:
  - v1.3.0 requires `GLIBC_2.38`;
  - the older V100-focused v0.0.3 still requires `GLIBC_2.32` for
    `flash_attn_v100` and `GLIBC_2.38` for `vllm/_C`;
  - gpu-07 is Rocky Linux 8.10 with `GLIBC_2.28`.
  This is an operating-system ABI incompatibility, not a V100, model, or
  inference-quality result. Both wheel files and their checksums remain under
  the remote experiment directory for audit.
- The pre-existing CUDA-13 vLLM nightly was rejected after an actual TP1 start:
  its PyTorch binary does not include SM70. Engine initialization failed with
  `cudaErrorNoKernelImageForDevice`. It must not be used on gpu-07.
- The CUDA 12.4 Transformers environment is verified to contain `sm_70`.
  A simple FP16 tensor operation on GPU 0 succeeded.
- LMDeploy 0.15 recognizes Qwen3.5 and has local Qwen3.5 model code, but its
  default TileLang Qwen3.5 gated-delta kernel fails on V100. An already
  installed Dao causal-conv1d fallback was enabled and moved the failure to an
  LLVM layout crash. LMDeploy is therefore not a serving baseline until its
  SM70 gated-delta kernels are repaired or replaced.

## 2026-08-31 working functional baseline

- `transformers 5.14.1` / `torch 2.6.0+cu124` / FP16 / GPU 0 was launched
  on port 8925 with continuous batching, paged KV blocks, 85% cache budget,
  and CUDA graphs deliberately disabled for the first stability probe.
- It loaded the local `Qwen3_5ForConditionalGeneration` checkpoint and passed:
  - text completion with `enable_thinking=false`;
  - separated reasoning output with `enable_thinking=true`;
  - an OpenAI-style image request (a red 256x256 PNG was answered `Red`);
  - API service startup and local OpenAI-compatible request routing.
- The service occupied about 18.3 GiB on GPU 0 immediately after startup.
  That is only a baseline allocation; later concurrent-load experiments must
  measure safe KV growth and throughput before deciding a production memory
  budget.

## 2026-08-31 experiment pause

- At the user's request, all services started by this study were stopped after
  retaining their logs and JSON probe outputs.
- A post-stop `nvidia-smi` check verified GPUs 0 through 7 at `0 MiB` and
  `0%` utilization with no compute applications reported.
- The next action is a paper-structure discussion rather than another
  deployment attempt.

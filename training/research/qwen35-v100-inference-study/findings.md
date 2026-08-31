# Findings: Qwen3.5-9B inference on 8xV100

## Current understanding

The earlier deployment left substantial throughput on the table because it was a four-GPU `transformers serve` process with continuous batching explicitly disabled. Its observed GPU placement does not establish a performance ceiling for this host.

`gpu-07` is materially different from `gpu-13`: it has eight Volta SM70 V100 32 GiB GPUs. Engine features must therefore be verified on this hardware rather than copied from the A100 deployment profile.

The strongest external candidate remains the 1Cat-vLLM V100 fork, but neither
public prebuilt wheel can run on this particular host because their shared
objects require a newer glibc than Rocky Linux 8.10 provides. A local source
build is therefore an engineering path, not an immediately deployable binary.

The CUDA-13 vLLM nightly is also ruled out: its PyTorch build does not carry
SM70 kernels, so it fails with `cudaErrorNoKernelImageForDevice` before model
service can begin.

The reliable baseline is now established: the local CUDA-12.4 Transformers
environment has SM70 support and serves the Qwen3.5 multimodal checkpoint on
a V100 with continuous batching enabled. It passed separate text,
thinking-off, thinking-on, and image-input checks. This replaces the earlier
four-GPU, no-continuous-batching service as the correctness control.

LMDeploy recognizes the model and successfully loads its weights, but its
Qwen3.5 gated-delta runtime is not currently stable on SM70: the default
TileLang path cannot initialize a CUDA target and its installed Dao fallback
hits an LLVM layout crash. This is a targeted kernel compatibility problem,
not evidence that the model cannot run on V100.

## Open questions

1. Can a locally built 1Cat SM70 runtime be made ABI-compatible with the
   available CUDA 12.4 / PyTorch-SM70 toolchain without compromising Qwen3.5
   multimodal behavior?
2. Does the Transformer continuous-batching baseline preserve schema and
   tool-call behavior as well as text, thinking, and image behavior?
3. Which layout maximizes the actual workload objective: eight TP1 replicas,
   TP2 replicas, TP4 replicas within NVLink islands, or a larger TP service?
4. At what safe KV/cache allocation and request concurrency does each layout
   maximize aggregate throughput without producing quality or stability
   regressions?

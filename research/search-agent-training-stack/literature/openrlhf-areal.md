# OpenRLHF and AReaL

Sources:

- https://github.com/OpenRLHF/OpenRLHF
- https://github.com/inclusionAI/AReaL

## OpenRLHF

OpenRLHF is a mature Ray + DeepSpeed RLHF stack with VLM, asynchronous multi-turn
agents, external environments over HTTP, vLLM/SGLang rollouts, and multiple
REINFORCE/GRPO-family estimators. It publishes a four-GPU Qwen3.5-2B VLM example and
an OpenAI-compatible agent executor.

The generic executor assumes an append-only, prefix-stable conversation when it
stitches multiple calls. IFV uses independent stage prompts and schemas, so the
example cannot wrap the runtime unchanged. A custom executor could remove that
assumption, making OpenRLHF the strongest conservative fallback if rLLM proves too
unstable.

Its public VLM example freezes the visual encoder. Full Qwen3-VL-8B with vision,
aligner, and language weights all trainable on four 40GB GPUs remains unproven.

## AReaL 2.0

AReaL 2.0 has a strong external-runtime story: an existing black-box agent can point
its OpenAI client at a model gateway while the runtime remains independent. It also
documents search agents and Qwen3-VL support, and separates rollout, inference,
training, and reward services.

The tradeoff is maturity and operational weight. Version 2.0 is very recent, uses a
new microservice architecture, and expects a modern CUDA/PyTorch stack. It is a
promising scale-up option, not the first gpu13 integration target.


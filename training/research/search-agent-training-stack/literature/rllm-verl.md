# rLLM and veRL

Sources:

- https://github.com/rllm-org/rllm
- https://github.com/volcengine/verl
- https://rllm-project.readthedocs.io/

veRL is the common distributed optimization substrate behind Search-R1 and many
later agents. It supports full-parameter FSDP/Megatron training, multimodal rollouts,
multi-turn tool configurations, vLLM/SGLang rollout engines, checkpointing, and
multiple policy-gradient estimators.

rLLM adds an agent/workflow layer and is used by the strongest open multimodal
search projects. Its current model-gateway direction is especially relevant:
existing agents can call an OpenAI-compatible policy endpoint while rLLM captures
each policy call as a training step. This is a better fit for IFV than rewriting the
runtime as a framework-owned chat loop.

Risks:

- current rLLM is a pre-release line and changes quickly;
- the successful multimodal search projects vendor or fork rLLM, veRL, Megatron, and
  bridge code rather than consuming a small stable package;
- released Qwen3-VL examples assume more GPUs than the current machine allocation.

If selected, rLLM must be pinned to an exact commit and isolated in its own
environment. The IFV adapter should be a workflow/environment boundary, never a
custom optimizer or trainer.


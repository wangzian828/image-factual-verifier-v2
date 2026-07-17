# Survey Protocol

## Inclusion criteria

A project is included when it releases at least one of:

- trainable search/deep-research agent code;
- a reproducible SFT, DPO, PPO, GRPO, or related post-training recipe;
- a reusable multi-turn tool environment or distributed rollout framework.

Inference-only wrappers are recorded but cannot establish training-stack maturity.

## Evaluation dimensions

1. exact training framework and distributed backend;
2. SFT, preference optimization, and online RL algorithms;
3. full-parameter versus adapter-only support;
4. VLM and specifically Qwen3-VL support;
5. environment abstraction and tool execution boundary;
6. step-level versus monolithic rollout semantics;
7. reward and credit-assignment mechanisms;
8. checkpoint, resume, serving, and rollout recovery;
9. compatibility with an external deterministic runtime;
10. practical fit for a single node with up to four A100 40GB GPUs.

## Planned sample

- Search-R1
- ReSearch
- ReCall
- WebThinker
- WebSailor / WebAgent
- DeepResearcher
- Agent-R1
- verl-agent
- AReaL
- OpenRLHF
- ms-swift


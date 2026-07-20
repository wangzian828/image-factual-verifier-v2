# ms-swift, Agent-R1, and verl-agent

Sources:

- https://github.com/modelscope/ms-swift
- https://github.com/AgentR1/Agent-R1
- https://github.com/langfengQ/verl-agent

## ms-swift

ms-swift is the strongest Qwen-specific SFT choice in the current project:

- official Qwen3-VL templates and processors;
- full-parameter multimodal training;
- DeepSpeed ZeRO-3 and offload;
- checkpoint/resume and serving integration;
- multi-turn Gym-style GRPO.

The RL limitation is architectural rather than feature absence. Its scheduler wants
to own the model/environment loop. Wrapping IFV would require reproducing the
existing staged state machine inside an ms-swift Env. Official 7B VLM GRPO examples
also use eight GPUs for LoRA, while full 7B agent GRPO separates seven training GPUs
and one rollout GPU.

Conclusion: keep ms-swift for full-parameter SFT, processor alignment, and checkpoint
smokes. Do not make it own IFV online RL merely to reduce the number of environments.

## Agent-R1

Agent-R1's step-level MDP is conceptually aligned with VisualFact/Task/Evidence
transitions and process rewards. It exposes per-step observations, actions, rewards,
and context policies.

Adopting it would require translating the whole IFV runtime into Agent-R1 Env/Tool
objects. The repository also has dependency and compatibility issues around its veRL
pin. It is a valuable design reference for later step-level credit assignment, not
the first integration substrate.

## verl-agent

verl-agent explicitly supports long-horizon Qwen3-VL workflows and configurable
per-step context/history. Like Agent-R1, it is most useful when the framework owns
the environment loop. It would duplicate more of IFV than a model-gateway approach.


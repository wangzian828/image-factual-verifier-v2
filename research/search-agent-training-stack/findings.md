# Search-Agent Training Stack Findings

## Research Question

Which mature stack should train a Qwen3-VL student for IFV while preserving the
existing deterministic runtime and Gemini teacher environment?

## Current Understanding

The initial scan suggests that Qwen multimodal SFT and long-horizon search-agent RL
are different infrastructure problems. This is exploratory until the repository and
paper matrix is complete.

## Key Results

Pending.

## Patterns and Insights

Pending.

## Lessons and Constraints

- Framework marketing is insufficient; count only capabilities visible in released
  code, configs, or documentation.
- Distinguish model training from inference-only agent orchestration.
- Distinguish generic multi-turn chat from a real environment that executes tools,
  returns observations, and controls stopping.
- Do not infer Qwen3-VL support from text-only Qwen support.
- Do not infer full-parameter training from LoRA examples.

## Open Questions

- Which framework has the smallest adapter boundary around an external deterministic
  runtime?
- Which framework can preserve per-step image inputs and loss/reward masks?
- Can one 8B VLM full-parameter policy and rollout engine fit safely on four A100 40GB
  GPUs, or must SFT and RL use different parallel strategies?

## Optimization Trajectory

No framework selection has been frozen.


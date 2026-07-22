# Search-Agent Training Stack Findings

## Research Question

Which mature stack should train a Qwen3-VL student for IFV while preserving the
existing deterministic runtime and Gemini teacher environment?

## Current Understanding

Qwen multimodal SFT and long-horizon search-agent RL are different infrastructure
problems and should use separate stacks.

The strongest direct precedents agree:

- Vision-DeepResearch uses Megatron-SWIFT for Qwen3-VL SFT and
  rLLM/veRL/Megatron/SGLang for RL.
- OpenSearch-VL uses LLaMA-Factory/DeepSpeed for full SFT and
  rLLM/veRL/Megatron/SGLang for RL.
- LiteResearcher uses LLaMA-Factory for cold-start SFT and veRL for staged online RL.

For IFV, ms-swift remains the best SFT choice because the existing repository already
uses its Qwen3-VL processor/template contracts and it supports full multimodal
DeepSpeed training.

rLLM with the veRL backend is the best initial RL candidate because its model gateway
can wrap an existing OpenAI-compatible agent runtime and record each independent LLM
call as a Step. IFV therefore keeps deterministic state reduction, tools, stopping,
and trace persistence.

## Key Results

- H1 supported: the split stack has the best evidence and lowest rewrite cost.
- H2 refuted: a single framework is possible, but it either duplicates IFV's state
  machine or has weaker direct multimodal-search precedent.
- H3 supported: IFV's stage boundaries are naturally represented as separate policy
  Steps inside one episode.
- The public 7B/8B multimodal search-RL recipes use at least eight large-memory GPUs.
  Four A100-40GB GPUs are enough to attempt full SFT with ZeRO-3 offload, but do not
  establish practical full online RL throughput.

## Patterns and Insights

- Cold-start SFT before online search RL is the dominant successful pattern.
- Modern agents do not rely on plain GRPO alone. They add RLOO/REINFORCE baselines,
  DAPO/GSPO variants, adaptive rollout allocation, dynamic filtering, or staged
  environments.
- Provider and search infrastructure failures need fatal-aware masking so the policy
  is not punished for an OSS timeout or inaccessible page.
- Cached/local-web environments precede live-web RL.
- External model gateways are a better fit than framework-owned Env loops when an
  auditable runtime already exists.
- IFV's online policy calls are text-only. Perception and image-comparison model calls
  belong to a frozen environment endpoint, even when both endpoints use Qwen.
- Search-R1, ReCall, DeepResearcher, rLLM SearchReward, and WebAgent-R1 all rely
  primarily on terminal outcome or environment-success rewards. Released code does
  not establish a mature generic per-search-step semantic reward.
- R-Search provides the closest useful process precedent: the policy submits selected
  raw Evidence, a frozen verifier answers using only that Evidence, and a deterministic
  comparator checks the verifier answer against gold.
- Current rLLM/veRL transforms broadcast `trajectory.reward` across trainable action
  tokens. IFV therefore needs an episode/trajectory join; a terminal-step JSON record
  alone is not a trainer integration.

## Lessons and Constraints

- Framework marketing is insufficient; count only capabilities visible in released
  code, configs, or documentation.
- Distinguish model training from inference-only agent orchestration.
- Distinguish generic multi-turn chat from a real environment that executes tools,
  returns observations, and controls stopping.
- Do not infer Qwen3-VL support from text-only Qwen support.
- Do not infer full-parameter training from LoRA examples.
- Do not send VLM tool-internal calls through the trainable policy gateway.
- Do not claim that four A100-40GB GPUs reproduce public 8B search-RL recipes.
- Do not start with live-web RL or plain sparse GRPO on the current small dataset.
- Do not force the independent IFV stage prompts into an append-only chat transcript.
- Do not let the reward judge see Evidence outside the Agent-selected verdict basis.
- Do not score ImageClaim status agreement until visual-observation and world-fact
  semantics are separated.
- Do not add per-step cost or search-count shaping before ranking calibration.

## Open Questions

- Does rLLM's current gateway fully preserve IFV native tool calls and JSON-schema
  outputs under the selected Qwen serving engine?
- Can Qwen3-VL-8B full multimodal SFT save/resume on four A100-40GB GPUs with
  ZeRO-3 offload?
- Can an 8B language-policy RL smoke fit with three training GPUs plus one rollout GPU,
  or must the first RL optimizer smoke use a smaller checkpoint?
- Which initial estimator is most stable for the available rollout group size:
  RLOO, REINFORCE baseline, or DAPO without group-variance normalization?
- Does selected-basis recoverability correlate with human ordering on fixed-policy
  K-sample groups, or does it only reproduce outcome correctness?

## Optimization Trajectory

The evidence-backed provisional selection is:

```text
full multimodal SFT:
  ms-swift 4.4.1 + DeepSpeed ZeRO-3/offload

online search-agent RL:
  rLLM model gateway + veRL backend + SGLang/vLLM rollout

fallback:
  OpenRLHF external agent executor

later scale-up candidate:
  AReaL 2.0
```

The RL choice remains conditional on a real gateway protocol smoke and gpu13 memory
probe.

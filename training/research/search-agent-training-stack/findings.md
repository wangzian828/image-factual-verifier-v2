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
- First-generation systems including Search-R1, ReCall, DeepResearcher and the built-in
  rLLM SearchReward rely primarily on terminal outcome. That observation must not be
  generalized to recent search-agent training: StepSearch, TIPS, LOTAPO, RewardFlow,
  CW-GRPO, FaithMed and SearchEyes/HaPO all expose a real process-credit path in code
  and training configuration.
- The recent implementations span several non-equivalent mechanisms: gold retrieval
  coverage, answerability potentials, counterfactual turn attribution, semantic step
  judges, state-graph propagation and outcome-advantage redistribution.
- R-Search's selected-Evidence verifier remains the right terminal evidence-quality
  component, but it does not replace turn-level process credit.
- Gemini's best-supported role is a frozen post-rollout process teacher and calibrator.
  It should score bounded action/observation transitions with evidence citations, not
  generate the policy's only trajectories or act as the sole truth source.
- PRInTS demonstrates that long-context management and process evaluation should be
  designed together: recursively preserve findings, uncertainty and plans while
  evaluating the latest tool response and current step.
- Current rLLM/veRL transforms broadcast `trajectory.reward` across trainable action
  tokens. IFV therefore needs both an episode join and a custom turn/segment-aware
  transform or advantage estimator; a terminal-step JSON record alone cannot train
  the investigated process behavior.

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
- Do not ask a teacher for an unrestricted scalar over hidden thinking. The process
  packet must identify pre-state, action, observation, state delta and cited IDs.
- Do not add per-step cost or search-count shaping before same-policy ranking
  calibration.

## Open Questions

- Does rLLM's current gateway fully preserve IFV native tool calls and JSON-schema
  outputs under the selected Qwen serving engine?
- Can Qwen3-VL-8B full multimodal SFT save/resume on four A100-40GB GPUs with
  ZeRO-3 offload?
- Can an 8B language-policy RL smoke fit with three training GPUs plus one rollout GPU,
  or must the first RL optimizer smoke use a smaller checkpoint?
- Does Gemini turn-level evidence-gain/direction/belief-update/regression labeling
  agree with human labels and counterfactual contribution on fixed-policy K-samples?
- Can an IFV-specific outcome-gated estimator preserve useful prefixes of failed
  trajectories without giving positive total credit to incorrect episodes?
- Is a custom rLLM transform sufficient, or should distinct turn advantages be
  implemented directly in the veRL-derived trainer?
- Does selected-basis recoverability add trajectory-ranking signal beyond outcome
  correctness when combined with process credit?

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

## 2026-07-22 implementation decision

The reviewed turn-credit systems remain useful evidence that process quality matters,
but are not adopted as an IFV training-algorithm contribution. The production contract
is deliberately narrower: generate G isolated full episodes for the same prompt, call
Gemini once per episode for a blind overall trajectory assessment, join private outcome
correctness only after rollout, and pass one scalar reward per episode to standard
GRPO. rLLM/veRL owns group normalization; no turn-level reward, counterfactual probe,
CW-GRPO redistribution or custom estimator is implemented.

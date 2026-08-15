# RL Reward and Optional Semantic Diagnostics

## Contract

The default RL reward is deterministic and outcome-dominant. It does not call an
LLM judge, does not read hidden reasoning, and does not require recovering one
canonical evidence chain.

```text
same case
  -> G isolated complete Qwen episodes
  -> canonical traces + strict audit
  -> post_rollout_rewards.jsonl
  -> reward_ledgers.jsonl
  -> same-prompt grpo_groups.jsonl
  -> standard GRPO in rLLM / veRL
```

`case_id` is the original item, `prompt_group_id` is the same-case rollout group,
and `episode_id` is one isolated attempt. Private gold is joined only after every
episode completes, and only to record deterministic correctness and process fields.

## Reward rule

The scalar reward follows:

```text
engineering error / strict audit failure / missing correctness -> mask
wrong verdict                                                  -> 0.0
correct verdict                                                -> 0.5 + 0.5 * process_quality
```

`process_quality` is a weighted average over deterministic fields already emitted in
`trajectory_scores.jsonl`, such as Evidence-chain recovery, discrepancy alignment,
stop quality, bridge quality, basis minimality, and calibrated stopping. Missing
process components are simply ignored. If no process component is present, a correct
strict-audit-passing episode receives `1.0`.

The default profile is:

```powershell
training/configs/rl/deterministic-process-v1.json
```

The legacy filename `training/configs/rl/semantic-reward-v5.json` is kept only as a
compatibility alias and contains the same deterministic profile.

## Build GRPO groups

```powershell
cd training
ifv-training build-run-rewards `
  --deterministic D:\runs\qwen-g4\post_rollout_rewards.jsonl `
  --rollout-members D:\runs\qwen-g4\rollout_groups.jsonl `
  --profile configs\rl\deterministic-process-v1.json `
  --ledger-output D:\runs\qwen-g4\reward_ledgers.jsonl `
  --group-output D:\runs\qwen-g4\grpo_groups.jsonl
```

Every complete episode receives one scalar reward. Incorrect but engineering-valid
episodes remain in their same-prompt group with reward `0.0`, so GRPO still sees
within-group contrast instead of silently dropping failures.

## Optional semantic diagnostics

`python -m src.eval.score_semantic_reward` may still be run for offline analysis,
ablation notes, or manual calibration. Its artifacts are content-addressed and can be
joined as diagnostics:

```powershell
ifv-training build-run-rewards `
  --deterministic D:\runs\qwen-g4\post_rollout_rewards.jsonl `
  --rollout-members D:\runs\qwen-g4\rollout_groups.jsonl `
  --semantic-artifacts D:\runs\qwen-g4\semantic_rewards `
  --ledger-output D:\runs\qwen-g4\reward_ledgers.jsonl `
  --group-output D:\runs\qwen-g4\grpo_groups.jsonl
```

Semantic artifacts never change scalar reward, trainability, positive-buffer
eligibility, or SFT export eligibility. They are not model-visible. SFT teacher
positives are screened separately by the frozen LLM `sft_eligibility` judge plus
deterministic hard constraints; non-fatal deterministic quality issues are recorded
as red flags and can break ties between multiple passing rollouts for the same case.
The v2 SFT gate uses a generic image-level fact target. It may accept a semantically
equivalent sub-fact when the supplied Evidence decisively establishes the same image
verdict; it does not require a pre-registered decision path or a specific Claim
wording. No human-review queue is produced, and this gate never changes RL reward.

## Reference-chain metrics

`reference_chain_metrics.jsonl` is also diagnostic-only. It measures whether a
trajectory recovered the pre-registered reference facts and acceptable evidence. It
does not judge whether an alternative evidence chain is reasonable, and it must not
be used as RL reward.

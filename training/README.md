# Image Factual Verifier Training

`training/` is a small, provider-neutral subproject for dataset conversion,
reward-ledger construction, GRPO group contracts, checkpoint manifests, and serving
manifests. It does not run the Agent loop and does not expose evaluator-private gold
to the runtime model.

## Local gates

```powershell
cd training
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q ifv_training scripts
git diff --check
```

## RL: standard GRPO

The default RL path is deterministic. It consumes `post_rollout_rewards.jsonl`
directly; no LLM judge is required for training reward.

```powershell
cd training
ifv-training build-run-rewards `
  --deterministic D:\runs\qwen-g4\post_rollout_rewards.jsonl `
  --rollout-members D:\runs\qwen-g4\rollout_groups.jsonl `
  --profile configs\rl\deterministic-process-v1.json `
  --ledger-output D:\runs\qwen-g4\reward_ledgers.jsonl `
  --group-output D:\runs\qwen-g4\grpo_groups.jsonl
```

Reward policy:

```text
engineering error / strict audit failure / missing correctness -> mask
wrong verdict                                                  -> 0.0
correct verdict                                                -> 0.5 + 0.5 * deterministic process quality
```

Every episode receives one scalar reward. rLLM / veRL handles standard same-prompt
normalization; this project does not implement turn-level reward, custom advantage,
or CW-GRPO.

Optional `--semantic-artifacts` may be supplied for offline diagnostics. Those
artifacts are recorded in the ledger but never change reward, trainability, or
positive-buffer eligibility.

## Frozen teacher SFT export

Frozen SFT has one canonical path:

```text
teacher rollouts + frozen LLM judge
  -> stage_accepted_teacher_release.py
  -> canonical accepted release
  -> export_dataset.py --accepted-release
  -> ifv-training converters
  -> run_sft.sh
```

The accepted release is authoritative for the selected trace, eligibility decision,
and provider-neutral policy/perception rows. The final exporter validates trace SHA-256,
eligibility identity, split membership, and model-visible schema; it does not reread
or regenerate policy rows from the original rollout directories.

First stage all accepted teacher candidates:

```powershell
python scripts/trajectory/stage_accepted_teacher_release.py `
  --source D:\runs\teacher-run-a D:\runs\teacher-run-a\sft-eligibility-v3 `
  --source D:\runs\teacher-run-b D:\runs\teacher-run-b\sft-eligibility-v3 `
  --minimum-accepted-cases 40 `
  --output-dir D:\training\sft-v1\accepted-teacher-release
```

Then export only from that canonical release:

```powershell
python scripts/trajectory/export_dataset.py `
  --accepted-release D:\training\sft-v1\accepted-teacher-release `
  --case-split D:\training\sft-v1\case-split\case_split.jsonl `
  --minimum-accepted-cases 40 `
  --output-dir D:\training\sft-v1\accepted-dataset
```

`--semantic-reward-dir` remains optional and diagnostic-only. Deterministic fatal
reasons such as `incorrect_result`, `engineering_error`, and
`legacy_core_ownership` are rejected during staging; non-fatal process issues remain
red flags in the canonical release.

Then convert accepted data for ms-swift:

```powershell
python -m ifv_training convert-accepted-perception `
  --input D:\training\sft-v1\accepted-dataset `
  --output D:\training\sft-v1\ms-swift-perception
python -m ifv_training convert-policy `
  --input D:\training\sft-v1\accepted-dataset `
  --output D:\training\sft-v1\ms-swift-policy
```

## Checkpoint validation and epoch selection

The training process itself remains the existing `swift sft` path. The launcher
accepts either the framework-native `IFV_MAX_STEPS` or
`IFV_NUM_TRAIN_EPOCHS`; it does not implement a second trainer.

After a run, validate every saved checkpoint that has a matching validation
metric:

```powershell
python -m ifv_training validate-checkpoints `
  --train-log D:\training\logs\exp-1\train.log `
  --checkpoint-root D:\training\checkpoints\exp-1 `
  --output D:\training\logs\exp-1\checkpoint-validation.json `
  --train-rows 50000
```

The report maps `global_step` to epoch using the observed global batch, records
missing checkpoint/evaluation pairs, and selects the lowest `eval_loss` by
default. This is a selection report, not an additional SFT eligibility gate.

If the existing external behavior evaluator has already evaluated checkpoints,
it may provide JSONL rows keyed by `global_step` with an explicit
`behavior_score` (or `selection_score`):

```json
{"global_step": 1000, "metrics": {
  "behavior_score": 0.81,
  "verdict_accuracy": 0.84,
  "evidence_chain_complete_rate": 0.77,
  "protocol_valid_rate": 0.99
}}
```

Then use `--behavior-metrics`; `--selection auto` will use that explicit
external score when present and otherwise fall back to `eval_loss`. The tool
does not invent weights for task metrics and does not treat missing behavior
metrics as a hard training failure.

Development subsets and their derived traces/artifacts/ledgers remain
`training_prohibited` and must not enter SFT, RL, teacher data, or synthetic training
data.

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

Frozen teacher export uses a fixed split plus the frozen LLM SFT eligibility gate.
Deterministic trajectory features are still used as hard-safety constraints for
clearly invalid teacher positives, as red-flag diagnostics, and as same-case
tie-breakers. It no longer requires semantic reward artifacts.

```powershell
python scripts/trajectory/export_dataset.py `
  --run-dir D:\runs\teacher-run-a `
  --case-split D:\training\sft-v1\case-split\case_split.jsonl `
  --eligibility-dir D:\training\sft-v1\eligibility `
  --minimum-accepted-cases 40 `
  --output-dir D:\training\sft-v1\accepted-dataset
```

`--semantic-reward-dir` remains optional and diagnostic-only. When supplied, artifact
IDs are preserved for audit trails, but they are not eligibility gates. Fatal
deterministic reasons such as `incorrect_result`, `engineering_error`,
`protocol_rejections`, and `legacy_core_ownership` still reject a teacher positive;
non-fatal deterministic reasons are retained as `deterministic_red_flags`.

Then convert accepted data for ms-swift:

```powershell
python -m ifv_training convert-accepted-perception `
  --input D:\training\sft-v1\accepted-dataset `
  --output D:\training\sft-v1\ms-swift-perception
python -m ifv_training convert-policy `
  --input D:\training\sft-v1\accepted-dataset `
  --output D:\training\sft-v1\ms-swift-policy
```

Development subsets and their derived traces/artifacts/ledgers remain
`training_prohibited` and must not enter SFT, RL, teacher data, or synthetic training
data.

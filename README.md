# Image Factual Verifier v4

Discrepancy-first, image-only factual investigation runtime.

```text
ImageOnlyRuntimeCase
  -> Gemini perception + positioned OCR
  -> deterministic visual facts and retrieval anchors
  -> Image Account Planning (1-3 ImageClaims + SearchHypotheses)
  -> account/hypothesis-owned native Gemini ReAct
  -> sparse multimodal Discrepancy Decision
  -> deterministic Coverage and claim/discrepancy/Evidence basis
  -> constrained discrepancy-first-v4 Judgment
  -> real | fake
```

The default decision policy is `discrepancy-first-v4`. The v3 runtime is frozen at
tag `runtime-v3-final-20260717`; legacy schemas may remain temporarily for read-only
trace replay but do not enter the v4 default path.

## Repository boundary

This repository owns the runtime, tools, canonical traces, strict audit,
post-rollout process scoring, and policy export. Benchmark construction, private gold,
classification scoring, licenses, and source snapshots belong to the separate data
pipeline repository and enter only through the immutable release contract.

Each public row contains exactly `case_id`, `image_path`, and `image_sha256`.
Evaluator-private gold is loaded only after every rollout.

The repository is a monorepo: `training/` is imported with its Git history, but
remains an independently installable Python project. Runtime dependencies and
training dependencies must not be mixed. Run root gates from this directory and
training gates from `training/`.

## Local setup

Python 3.11 is required.

```powershell
python -m pip install -e ".[dev]"
$env:GEMINI_API_KEY = "..."
$env:SERPER_API_KEY = "..."
python -m src path\to\image.jpg
```

Gemini sees the original image during perception and Planning. Later semantic stages
receive an explicit context packet assembled from canonical state and the immutable
runtime archive; focused visual tools can inspect the saved image again. Planning,
Discrepancy Decision, and Judgment are standalone requests. Each ReAct action uses
`previous_interaction_id` only inside its native
`function_call -> function_result` round trip; that short chain does not cross into
the next action or stage.

Search titles, snippets, and reverse-image matches are Discovery only. Verdict
Evidence must preserve exact fetched text or a successful visual observation,
provenance, artifact hashes, and successful function-call ownership. Provider or
protocol failure is an engineering error, never a factual verdict.

## Evaluation outputs

```powershell
python -m src.eval.run_eval `
  --benchmark path\to\release\runtime_input\cases.jsonl `
  --output-dir path\to\new-run
```

Outputs include predictions, run diagnostics, canonical traces, process metrics,
reference-chain metrics, v4 teacher scores, and `ifv-policy-v2` trajectories. The v4
training gate focuses on Evidence-chain recovery, discrepancy alignment, stop quality,
protocol validity, and absence of post-verdict actions.

For on-policy RL, run `--rollouts-per-case 4`. Each member receives a unique
`episode_id` and sampling seed while retaining the public `case_id`; every complete
episode owns its workflow, mutable state, interaction lifecycle, runtime archive, and
trace. The run writes `rollout_groups.jsonl` and `post_rollout_rewards.jsonl`;
`training/ifv-training build-run-rewards` consumes those deterministic post-rollout
records directly and emits standard GRPO groups. `python -m src.eval.score_semantic_reward`
is optional offline diagnostics only; it is not a reward or eligibility gate. See
[docs/rl-semantic-reward.md](docs/rl-semantic-reward.md).

## Validation and acceptance

```powershell
python -m pytest -q
python -m compileall -q src scripts
git diff --check
python scripts/audit_real_trace.py path\to\traces --json --strict-scheduler
```

```powershell
cd training
python -m pytest -q
python -m compileall -q ifv_training scripts
```

Local tests are deterministic gates, not production acceptance. v4 completion also
requires replaying the four frozen historical traces, then one real Gemini canary
with manual trace review, followed by three to four heterogeneous canaries. Do not
run a full 20-case batch or Qwen trajectory production before those gates pass.

# Runtime Release Contract

This repository consumes only immutable v0.3 image-only releases produced by the
separate data-pipeline repository.

## Ownership

The data project owns:

- release construction, review, and content-addressed images;
- `manifest.json`;
- `runtime_input/cases.jsonl`;
- optional evaluator-private source-access policy;
- evaluator-private gold and acceptable evidence;
- classification and process-reference protocols;
- classification scorer, checksums, licenses, and source snapshots.

The v4 runtime owns:

- strict release and row parsing;
- path and image-hash validation;
- Agent rollout and canonical traces;
- VisualFacts, tasks, Findings, Reflection, Coverage, and verdict basis;
- post-rollout process scoring;
- policy trajectory and dataset export/audit.

The repositories do not import each other.

## Required manifest

The evaluator entrypoint is:

```text
<release-root>/runtime_input/cases.jsonl
```

`manifest.json` must declare:

```json
{
  "schema_version": "ifv-image-only-benchmark-release-v0.3",
  "runtime_contract_version": "ifv-image-only-runtime-v1",
  "input_mode": "image_only",
  "decision_policy_version": "reinspect-v2",
  "runtime_contract": {
    "allowed_keys": ["case_id", "image_path", "image_sha256"],
    "private_keys_absent": true
  }
}
```

The manifest `decision_policy_version=reinspect-v2` identifies the data-pipeline
release protocol. It is not an Agent implementation selector. The release adapter
validates that source contract, while `run_eval` independently configures the Agent
as `discrepancy-first-v4`. Canonical traces therefore declare
`discrepancy-first-v4`, and the run manifest records both the source release policy
and the effective Agent policy in separate `benchmark` and `agent` objects.

The manifest `artifacts` object supplies release-relative paths for:

```text
agent_input
evaluation_gold
classification_protocol
process_reference_protocol
licenses
```

When `source_access_policy.active=true`, it also supplies an existing release-relative
policy path. When inactive, no path may be declared.

All artifact paths must remain inside the release root. `artifacts.agent_input` must
equal the evaluator entrypoint.

## Public cases

Every JSONL row has exactly:

```json
{
  "case_id": "case_0123456789abcdef",
  "image_path": "assets/sha256/ab/abcdef.jpg",
  "image_sha256": "abcdef..."
}
```

Forbidden public content includes:

- factual labels or gold;
- decisive facts or acceptable evidence;
- source provenance and construction metadata;
- runtime VisualFacts, tasks, Findings, trajectories, or scores;
- any extra or nullable placeholder field.

`image_path` is relative to `runtime_input/`, cannot escape that directory, and is
re-hashed before execution.

## Rollout isolation

Before rollout, v4 may read:

- public manifest and protocols;
- public cases and images;
- an active evaluator-private source-access policy.

It must not read `evaluator_private/gold.jsonl`.

After all rollouts, v4 loads gold, validates the one-to-one `case_id` join, computes
process metrics and trajectory scores, and records the gold artifact hash. Gold is
never copied into model inputs, predictions, or canonical Agent state.

For grouped training sampling, one public case expands into several independent
episodes. `case_id` remains the immutable release identity; `episode_id` is an
Agent-owned attempt identity and `prompt_group_id` is the same-case sampling group.
Each episode has its own workflow/orchestrator, state, interaction lifecycle, archive,
runtime event stream, trace and derived seed. Content-addressed tool caches may be
shared, but mutable Agent state may not. Gold is still loaded only after every episode
in the run has stopped, then joined by the original `case_id`.

## Evaluation artifacts

### `predictions.jsonl`

Contains successful classifications only:

```json
{"case_id": "case_...", "verdict": "fake"}
```

The field set is exactly `case_id` and `verdict`; verdict is
`real|fake`.

Engineering-error cases are absent from this file. The data-owned classification
scorer treats the missing case as wrong.

For multi-rollout generation, `predictions.jsonl` still contains only rollout index 0
for each public case, so it remains scorer-compatible. `episode_predictions.jsonl`
contains every completed episode and adds `episode_id`, `prompt_group_id`, and
`rollout_index`; it is a training/evaluation diagnostic, not a data-pipeline scorer
input.

### `run_results.jsonl`

Contains runtime diagnostics:

```text
case_id, verdict/error, confidence, verdict_basis, termination,
time/call/token accounting, stage timings, trace_path
```

`error` is an execution status, not a classification verdict.

### Process and trajectory outputs

```text
process_metrics.jsonl
reference_chain_metrics.jsonl
trajectory_scores.jsonl
trajectory_sft.jsonl
```

These are post-rollout evaluator/training artifacts. They are never model-visible.
`trajectory_sft.jsonl` contains one complete episode per row; it is the default SFT
artifact. The old step-level `policy_trajectories.jsonl` name is legacy-only and is
not emitted by the current evaluator.
`reference_chain_metrics.jsonl` contains exactly four metrics: fact recovery recall,
evidence recovery recall, complete chain recovery recall, and final-basis reference
precision. It only evaluates the decisive facts and acceptable evidence retained in
evaluator-private gold. URL, span, snapshot, and SHA identity are data-audit
properties, not Agent capability metrics. Off-chain material is neutral unless it
enters the final verdict basis. These scores are canonical-reference diagnostics only:
failing to recover the pre-registered chain does not mean the trajectory used
unreasonable evidence, and the scores do not override classification, reward, or
teacher eligibility.

### Grouped rollout outputs

When `--rollouts-per-case > 1`, `traces/<episode_id>.json` never overwrites another
attempt of the same public case. The run additionally emits:

```text
rollout_groups.jsonl
post_rollout_rewards.jsonl
```

The first file has one member record per episode with `prompt_group_id`, original
`case_id`, `episode_id`, `rollout_index`, group size, seed and trace path. The second
file is written only after all rollout completion and carries deterministic private-gold
alignment (`classification_correct`), engineering/audit gates, policy step IDs, and
deterministic process components. A later training-only step consumes it directly and
writes `grpo_groups.jsonl`, whose members retain raw scalar rewards for the framework's
standard within-group normalization. Optional semantic reward artifacts may be joined
as diagnostics, but they do not change reward, trainability, or SFT eligibility.
Frozen teacher SFT eligibility is governed by the frozen LLM `sft_eligibility` gate
plus deterministic hard-safety constraints. Non-fatal deterministic process issues
are recorded as red flags and can influence same-case teacher selection.
The v2 judge evaluates a generic image-level fact target. Runtime `ImageClaim`
identifiers remain lineage metadata, but the judge does not require Claim wording,
relation slots, a specific URL, an original-image match, or a registered evidence
path. It receives all successful Evidence rows and may select any decisive,
image-grounded Evidence or compatible sub-fact. Fatal safety, engineering, and
invalid-reference failures reject SFT eligibility. The frozen judge also receives a
compact rejected-turn history and rejects trajectories that repeat blocked behavior
or leave it unresolved; a single materially corrected turn is a warning rather than
an automatic veto. No audit result creates a human-review queue.

### Other outputs

```text
run_manifest.json
summary.json
traces/*.json
```

`run_manifest.json` records public artifact hashes before rollout and private gold hash
only after rollout.

## Compatibility policy

The consumer fails closed on any schema, input mode, runtime contract, or decision
policy other than the values above. There is no v0.2 or claim-mode fallback.

## Read-only archive collection

For teacher trajectory collection from the benchmark-pipeline historical archive,
the lightweight `src.eval.run_cases` runner also accepts:

```bash
python -m src.eval.run_cases \
  --archive-root /path/to/archive \
  --profile teacher-gemini \
  --output-dir /path/to/run
```

The archive adapter reads only `human-review-candidates.jsonl` and resolves each
row's `archive_image_path`. It projects the candidate into the three runtime
fields `case_id`, `image_path`, and `image_sha256`; labels, claims, evidence,
source metadata, and construction metadata are not passed to the Agent. The
archive is never rewritten.

This mode is a rollout-collection path, not a v0.3 release and not a scoring
release. It writes traces and minimal run diagnostics under the independent run
directory. Classification accuracy, process scoring, and SFT eligibility must be
performed later with an explicit evaluator-side gold/review package.

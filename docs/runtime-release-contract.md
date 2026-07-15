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

The v3 runtime owns:

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

`reinspect-v2` is the current v3 verdict policy. It does not mean the repository
supports a v2 runtime.

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

Before rollout, v3 may read:

- public manifest and protocols;
- public cases and images;
- an active evaluator-private source-access policy.

It must not read `evaluator_private/gold.jsonl`.

After all rollouts, v3 loads gold, validates the one-to-one `case_id` join, computes
process metrics and trajectory scores, and records the gold artifact hash. Gold is
never copied into model inputs, predictions, or canonical Agent state.

## Evaluation artifacts

### `predictions.jsonl`

Contains successful classifications only:

```json
{"case_id": "case_...", "verdict": "fake"}
```

The field set is exactly `case_id` and `verdict`; verdict is
`real|fake|unverifiable`.

Engineering-error cases are absent from this file. The data-owned classification
scorer treats the missing case as wrong.

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
policy_trajectories.jsonl
```

These are post-rollout evaluator/training artifacts. They are never model-visible.
`reference_chain_metrics.jsonl` contains exactly five metrics: fact recovery recall,
complete chain recovery recall, frozen-exact evidence recall, semantic evidence
recall, and final-basis reference precision. It only evaluates the decisive facts and
acceptable evidence retained in evaluator-private gold. Off-chain material is neutral
unless it enters the final verdict basis. These scores do not override classification
or teacher eligibility.

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

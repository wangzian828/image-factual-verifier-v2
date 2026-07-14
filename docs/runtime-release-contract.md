# Runtime Release Contract

This repository consumes only immutable v0.3 image-only benchmark releases produced
by the separate `image-factual-verifier-data-pipeline` project. It does not import
construction-pipeline Python modules and does not maintain the old claim-mode release
contract.

## Ownership

The data project owns:

- `manifest.json`;
- `runtime_input/cases.jsonl` and content-addressed images;
- optional `evaluator_private/source_access_policy.json`;
- `evaluator_private/gold.jsonl`;
- classification and process-reference protocols;
- checksums, licenses, construction records, and release review.

The runtime owns:

- strict manifest and public-row parsing;
- release-relative path resolution and image SHA-256 verification;
- Agent rollout and canonical traces;
- post-rollout private-gold join validation;
- `InvestigationBrief`, `VisualFact`, `ResearchTask`, `Finding`, Reflection,
  coverage, and `verdict_basis`;
- process scoring and later trajectory export.

## Required Release

The benchmark entrypoint is:

```text
<release-root>/runtime_input/cases.jsonl
```

The release root must contain `manifest.json` with:

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

Every public row contains exactly:

```json
{
  "case_id": "case_0123456789abcdef",
  "image_path": "assets/sha256/ab/abcdef.jpg",
  "image_sha256": "abcdef..."
}
```

Public rows must not contain claim placeholders, factual labels, decisive facts,
acceptable evidence, construction metadata, source provenance, or runtime-generated
investigation state.

## Manifest Artifacts

`manifest["artifacts"]` supplies release-relative paths for:

```text
agent_input
evaluation_gold
classification_protocol
process_reference_protocol
licenses
```

All paths must remain inside the release root. `artifacts.agent_input` must resolve to
the exact benchmark path passed to the evaluator.

`source_access_policy.active=false` means no policy file is required. When active, the
manifest must provide a release-relative path to an existing policy file. The policy
is loaded before retrieval and is never exposed to the model.

## Rollout Isolation

Before rollout the runtime may read:

- the public manifest;
- public protocols;
- public runtime rows;
- image assets;
- an active evaluator-private source-access policy.

It must not read `evaluator_private/gold.jsonl` until all Agent rollouts finish.
After rollout, gold is loaded only to validate the `case_id` join and later drive
classification/process scoring. Gold values must not be copied into predictions or
canonical Agent traces.

## Predictions

`predictions.jsonl` contains at least:

```json
{"case_id": "case_...", "verdict": "real"}
```

Allowed verdicts are:

```text
real | fake | unverifiable
```

Runtime diagnostics such as confidence, `verdict_basis`, trace path, termination,
costs, and engineering error may be additive. Labels, factual status, decisive gold
facts, and acceptable evidence are forbidden.

Classification scoring remains owned by the data project. The runtime's
`summary.json` reports execution status and cost; it does not duplicate benchmark
accuracy or Macro-F1.

## Current Activation State

The v0.3 release consumer is active:

- manifest and row validation;
- path-boundary validation;
- image hash verification;
- optional source policy;
- post-rollout gold join;
- scorer-compatible predictions.

The image-only Agent itself is still under implementation. Until VisualFact bootstrap
and `reinspect-v2` execution are active, the default workflow returns an explicit
engineering error:

```text
image-only runtime input is valid, but VisualFact bootstrap and reinspect-v2
execution are not active yet
```

It must not silently convert image-only input into an embedded or external claim.

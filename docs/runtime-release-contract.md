# Runtime Release Contract

This repository is the runtime and evaluation consumer. Benchmark construction,
review, release finalization, source collection, and image generation belong to the
separate `image-factual-verifier-data-pipeline` project.

The projects communicate only through an immutable release directory. They must not
import each other's Python modules.

## Ownership

The data project owns:

- `runtime_input/cases.jsonl` and content-addressed image assets;
- `evaluator_private/source_access_policy.json`;
- `evaluator_private/run_eval.jsonl`;
- `evaluation_gold/gold.jsonl`;
- release manifests, checksums, licenses, snapshots, and construction records.

This runtime owns:

- parsing public runtime rows into `VerificationCase`;
- resolving release-relative image paths;
- loading the source-access policy before retrieval without exposing it to the model;
- running the Agent and persisting canonical traces;
- joining evaluator-private rows and gold only after rollout;
- deriving `InvestigationBrief`, `VisualFact`, `ResearchTask`, `Finding`, Reflection,
  coverage, and verdict-basis state.

The data project must not generate runtime investigation state. Runtime state must not
be added to `runtime_input/cases.jsonl`.

## Active v0.2 Contract

Every public row must explicitly contain:

```json
{
  "case_id": "case_0123456789abcdef",
  "image_path": "assets/sha256/ab/abcdef.jpg",
  "image_sha256": "abcdef...",
  "claim_mode": "external_claim",
  "user_claim": "A public factual proposition.",
  "claim_surface": null,
  "claim_source_region": null,
  "claim_observed_at": "2024-01-02",
  "decision_policy_version": "reinspect-v1"
}
```

The active adapter accepts `external_claim` and `embedded_claim` with
`decision_policy_version=reinspect-v1`. It rejects mixed legacy/release rows, missing
policy versions, unknown policies, and `image_only`.

The runtime input must not contain labels, target claims, acceptable evidence, source
URLs used to construct the case, intervention metadata, world IDs, or review data.

## Planned v0.3 Contract

`image_only` becomes publishable only after all of the following are true:

1. the runtime `ClaimMode` supports `image_only`;
2. the runtime projects it into an immutable `InvestigationBrief`;
3. `reinspect-v2` Coverage and Judgment validation are active;
4. cross-repository fixture tests pass against a finalized release;
5. the data project removes its current image-only release gate.

The intended public row is:

```json
{
  "case_id": "case_0123456789abcdef",
  "image_path": "assets/sha256/ab/abcdef.jpg",
  "image_sha256": "abcdef...",
  "claim_mode": "image_only",
  "user_claim": null,
  "claim_surface": null,
  "claim_source_region": null,
  "claim_observed_at": null,
  "decision_policy_version": "reinspect-v2"
}
```

`VisualFact`, task, Finding, Reflection, and verdict-basis records remain runtime
outputs and do not become public release input fields.

## Compatibility Matrix

| Release input | Decision policy | Runtime status |
|---|---|---|
| v0.2 external claim | `reinspect-v1` | Supported |
| v0.2 embedded claim | `reinspect-v1` | Supported |
| external/embedded | unknown or `reinspect-v2` | Rejected until explicitly activated |
| image-only | `reinspect-v1` | Invalid combination |
| planned v0.3 image-only | `reinspect-v2` | Rejected until VisualFact Phase 3 acceptance |

Any interface change must first update the fixture tests in `test_release_adapter.py`
and `test_eval_artifacts.py`. The producing data project may enable a new release mode
only after those consumer tests are green.

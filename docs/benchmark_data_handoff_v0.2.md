# Benchmark Data Handoff v0.2

This document is the concrete handoff contract between benchmark construction,
Agent runtime evaluation, and later SFT/RL trajectory generation. It describes data
origins and files, not the construction engine internals.

## 1. Scope

The first v0.2 seed is real-only. It does not add generated or edited samples merely
to balance the three verdict classes. Class counts may be unequal.

The seed serves two purposes:

1. run the production Agent on real images and real open-web evidence;
2. expose data-quality and retrieval failures before controlled T2I training worlds
   are introduced.

Controlled T2I worlds are a later training track. They must be versioned separately
and must not be silently mixed into the real seed.

## 2. Concrete Data Sources

### 2.1 AVerImaTeC transfer candidates

AVerImaTeC supplies real image-text fact-check cases and source provenance. For this
project it is a candidate source, not ready-made gold.

Use it as follows:

- download the official release and its image archive on gpu-13;
- retain the original claim and image association;
- use the fact-check URL only as evaluator-private provenance and candidate discovery;
- independently recover original, official, or reputable news evidence;
- never expose the fact-check domain, URL, verdict, rationale, or gold evidence to the
  Agent;
- require the benchmark-wide source access policy during every runtime evaluation.

An AVerImaTeC label may initialize a review queue, but it is not accepted as this
benchmark's final label until evidence-only review and adjudication pass.

### 2.2 Publicly licensed original images

Preferred sources, in order:

1. public-domain or explicitly open official sources such as NASA, NOAA, USGS, UN,
   and government media repositories;
2. Wikimedia Commons assets with an auditable license and source page;
3. official newsroom or press-kit images whose reuse terms are reviewed per asset;
4. self-collected images with recorded permission.

These sources provide authentic camera images, original captions, event context, and
hard negative cases such as old images reused with a new claim. A camera image is not
automatically `real`; its claim can still be refuted by event, time, place, identity,
or propagation context.

### 2.3 Fact-check sites

Fact-check sites are discovery-only and evaluator-private. They may identify a case
worth reconstructing, but the Agent-facing investigation must use independently
recovered evidence. Their complete domains and URLs are included in the release-wide
source access policy.

### 2.4 Later controlled T2I track

This is not part of the real seed. After the Agent and data contract are stable, a
separate release may contain Gemini T2I images generated from Internet-grounded worlds:

- `supported`: the visual specification agrees with the frozen world facts;
- `refuted`: exactly one decisive, visible fact slot is changed;
- `unverifiable`: used only when the evidence/world contract genuinely makes the
  decisive proposition unresolvable, not as a generation failure bucket.

Record `image_origin=camera|t2i_generated` independently from the factual label.

## 3. Physical Release Layout

```text
benchmark_v0.2/
  release_manifest.json
  runtime_input/                 # only this subtree is mounted for the Agent
    cases.jsonl
    assets/sha256/<prefix>/<sha256>.<ext>
  construction_record/          # builder/reviewer private
    cases.jsonl
    sources.jsonl
    licenses.jsonl
    snapshots/sha256/
  evaluation_gold/              # evaluator private
    gold.jsonl
    adjudication.jsonl
  trajectories/                 # Teacher/Student rollout artifacts
    episodes.jsonl
    steps.jsonl
  evaluator_private/
    source_access_policy.json
    run_eval.jsonl
  checksums/SHA256SUMS
```

`runtime_input`, `construction_record`, and `evaluation_gold` must be separate files
and separate mount/permission boundaries. Ignoring private fields in one combined JSONL
is not sufficient isolation.

## 4. Runtime Cases

Every row in `runtime_input/cases.jsonl` is UTF-8 without BOM and contains exactly one
JSON object. Required fields:

```json
{
  "case_id": "case_01J...",
  "image_path": "assets/sha256/ab/ab...cd.jpg",
  "image_sha256": "ab...cd",
  "claim_mode": "external_claim",
  "user_claim": "A factual proposition supplied by the user.",
  "claim_surface": null,
  "claim_source_region": null,
  "decision_policy_version": "reinspect-v1"
}
```

Supported modes:

- `external_claim`: `user_claim` is required and is the decisive evaluation target.
- `embedded_claim`: `user_claim` is null; a factual assertion is visibly embedded in
  the image and the runtime recovers `claim_surface` from pixels/OCR.
- `image_only`: no user or embedded claim is assumed. The Agent produces an image
  account first; the evaluator maps to three classes only when a decisive, auditable
  proposition can be recovered. This mode must be explicit and must not be represented
  as `embedded_claim` merely because `user_claim` is absent.

The current production state model supports the first two modes. A release containing
`image_only` cannot be declared runtime-compatible until that enum, planning contract,
and output adapter are implemented.

Runtime rows must not contain labels, fact-check URLs, world IDs, intervention details,
acceptable evidence, target regions, answer explanations, or evaluator comments.

## 5. Evaluation Gold

`evaluation_gold/gold.jsonl` is joined to runtime cases by `case_id` only after the
Agent run ends:

```json
{
  "case_id": "case_01J...",
  "factual_status": "refuted",
  "target_claim": "Normalized decisive proposition.",
  "acceptable_evidence": [
    {
      "source_id": "source_...",
      "canonical_url": "https://...",
      "exact_span": "verbatim passage",
      "stance": "refute"
    }
  ],
  "unverifiable_reasons": [],
  "image_origin": "camera"
}
```

Internal factual statuses map to the product labels as follows:

- `supported` -> `real`;
- `refuted` -> `fake`;
- unresolved, conflicting, access-limited, or saturated -> `unverifiable` with typed
  reasons.

`error` is an engineering outcome and never a factual label. `AI-generated` is an
image-origin attribute and never automatically maps to `fake`.

## 6. Label Construction

For every accepted real case:

1. normalize the decisive claim without using the source site's verdict wording;
2. recover an independent evidence bundle and preserve exact page spans, URL,
   retrieval time, and artifact hash;
3. Reviewer A assigns a status using only that evidence bundle;
4. Reviewer B independently assigns a status using the same bundle;
5. disagreement goes to adjudication;
6. freeze the accepted status and evidence before any model evaluation.

The seed does not need equal class sizes. Sampling reports must publish per-class and
per-source counts so imbalance is visible.

## 7. Asset and Leakage Gates

Before release, the finalizer must:

- recompute every asset SHA-256 and compare it with the declared hash;
- reject symlinks, path traversal, unreadable images, invalid magic, decode failures,
  and unsupported dimensions;
- run exact-hash and perceptual near-duplicate checks across splits and events;
- keep IDs, paths, and filenames label-neutral;
- verify license and provenance records for every asset;
- build `source_access_policy.json` from the complete frozen release, never a selected
  subset;
- confirm that runtime mounts contain no private files or fields;
- run one `external_claim` and one `embedded_claim` dry run with the complete policy;
- reject a release on any engineering error, image hash mismatch, or private value
  appearing in a trace.

## 8. Evaluator Adapter

`evaluator_private/run_eval.jsonl` is generated from the frozen release and may contain
the server-absolute `image_path`, `ground_truth`, bucket, and provenance URL required by
the current evaluator. It is evaluator-private and is not a source dataset.

Minimum adapter row:

```json
{
  "sample_id": "case_01J...",
  "image_path": "/gsdata/home/wza/.../ab...cd.jpg",
  "user_claim": "A factual proposition supplied by the user.",
  "ground_truth": "fake",
  "bucket": "real_seed_external_claim",
  "source_article_url": "https://evaluator-private.example/case"
}
```

New datasets must not use the legacy `runtime_claim` alias. The adapter and its hash are
recorded in the evaluation manifest.

## 9. Trajectory Handoff

Teacher and Student receive exactly the same runtime input, tool schemas, observation
store, and budget. Neither receives evaluation gold.

Each trajectory step minimally records:

```yaml
episode_id:
step_id:
runtime_observation_ref:
policy_input_token_ids: []
policy_action_token_ids: []
action_json: {}
tool_result_ref:
action_valid:
terminated:
```

Each episode additionally records `final_answer`, tool calls, token cost, wall time,
and the post-run evaluator result. SFT loss applies only to policy-generated reasoning,
function-call, and final-answer tokens. System text, user input, image placeholders,
tool observations, schema errors, and all evaluator-private values are masked.

Training trajectories that contain `propose_visual_check` must distinguish
`proposal_origin=policy` from production `proposal_origin=runtime`. Only policy-proposed
checks count toward the Evidence-to-Vision core-loop metric.

## 10. Server Handoff

All datasets and run artifacts live under:

```text
/gsdata/home/wza/image-factual-verifier-v2-data/
```

Recommended incoming and frozen paths:

```text
benchmarks/incoming/<team>/<delivery-id>/
benchmarks/releases/v0.2/<release-id>/
```

Code is edited and committed locally, pushed through GitHub, and only pulled on
gpu-13. Data is downloaded or generated directly on the server; it is never transferred
through the SSH control endpoint.

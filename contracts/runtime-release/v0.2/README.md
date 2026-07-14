# Runtime Release Contract Fixtures v0.2

This directory is a vendored snapshot of the data producer's canonical
machine-readable v0.2 handoff.

Producer baseline:

```text
repository: image-factual-verifier-data-pipeline
commit: dc3c625
```

The public fixtures are:

- `external_claim.json`
- `embedded_claim.json`

The evaluator-private fixtures are:

- `source_access_policy.json`
- `evaluator_private.json`
- `evaluation_gold.json`

The public rows intentionally contain no factual labels, target claims, evidence,
source URLs, world/intervention metadata, review data, or runtime-generated
investigation state.

The active combinations are `external_claim|embedded_claim` with
`decision_policy_version=reinspect-v1`. `image_only` and `reinspect-v2` require a new
versioned fixture directory and must not change these files.

The producer and consumer repositories must not import each other's Python packages.

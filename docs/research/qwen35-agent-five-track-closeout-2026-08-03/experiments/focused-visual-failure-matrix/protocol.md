# H6 protocol: focused-visual controlled failure matrix

Status: locked before execution.

## Question

Does every bounded focused-visual failure stop before a source-only follow-up
Decision, preserve an auditable failure reason, and avoid inventing pixel
Evidence or semantic support?

## Frozen fault set

Exercise the real visual-reinspection reducer and replay summary with controlled
tool outcomes. No retrieval or free-running model rollout is permitted.

1. outer tool timeout;
2. provider HTTP 429 after bounded retries;
3. provider HTTP 5xx after bounded retries;
4. malformed visual response/schema;
5. successful envelope with an empty visual observation;
6. explicit `budget_exhausted`;
7. error payload returned normally rather than raised;
8. Decision protocol-correction exhaustion after Evidence review.

## Required assertions

For focused-visual failures:

- one stable structured failure code is recorded;
- no `image_region` pixel Evidence or visual Finding is created;
- the pending visual reinspection becomes failed and its task is exhausted;
- no second source-only Decision is run;
- source Evidence is not promoted to support because the visual tool failed;
- replay output names the exact failed stage, failure IDs/codes, and the
  source-only follow-up block;
- fallback is bounded, nonterminal, and non-factual.

For Decision correction exhaustion:

- reviewed Evidence may be retained only as an `insufficient` assessment;
- the fallback cannot support/refute a Claim, create a discrepancy, retire a
  route, or propose `real`/`fake`;
- the result is explicitly marked as deterministic exhaustion fallback.

## Acceptance

The complete matrix passes locally and on the clean remote clone. Existing v4
focused suites and replay tests must remain green.

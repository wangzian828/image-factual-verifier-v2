# Image-Centric Runtime Research Log

## 2026-07-22 — provider-result source-policy boundary

- The first frozen batch on `7e62ab0` completed six cases with passing strict
  audits, then stopped at case `case_4c91d7d82956386e`. Serper returned a
  PolitiFact candidate that was not one of the release's exact excluded URLs;
  the runtime policy therefore allowed it into Discovery, and a later visit made
  the known fact-check domain visible in Evidence and policy context. The strict
  audit correctly reported 18 hard URL leaks. The batch was stopped before the
  next case performed a model call.
- The generic boundary now blocks known fact-check domain markers whenever an
  evaluation SourceAccessPolicy is active, including provider rows and query or
  content references. It filters results before workspace insertion and leaves
  query text and fact semantics untouched. New regressions cover an unlisted
  PolitiFact result; local gates pass (`485 passed`).
- The next step is a single rerun of `case_4c91d7d82956386e`, followed by a fresh
  full-20 launch only after strict audit passes.

## 2026-07-22 — Planning source-policy regression accepted

- `qwen35-monarch-planning-policy-7e21884-r1` completed with zero engineering
  errors after 17 investigation actions and stopped normally as
  `meaningful_routes_exhausted`.
- Strict scheduler audit passed with zero scheduler, protocol, or route-control
  rejection. Two fact-check-oriented Planning proposals were rejected before
  atomic commit and execution, remained correction warnings, and did not enter
  canonical Hypothesis/Task queries or SFT policy actions.
- The final `real` verdict is semantically wrong. Decision twice proposed `fake`,
  but the cited material mixed support/refute directions or was not owned by the
  affected Claim, so the deterministic Evidence qualification gate correctly
  rejected it. This remains a batch-level Evidence-promotion analysis item rather
  than a case-specific rule.
- The next run freezes commit `7e21884` and restarts the complete 20-case set from
  a fresh concurrency-1 directory with incremental strict audit.

## 2026-07-22 — Monarch gate accepted and full 20 launched

- `qwen35-monarch-dedup-cd0c04f` completed with correct `fake`, zero
  engineering errors, two investigation actions, seven model calls, and a
  provenance-complete evidence-determined basis in 121.42 seconds.
- Strict scheduler audit passed with no rejection and one successfully recovered
  correction warning. The high-salience Claim was refuted, its decisive discrepancy
  was established, and the runtime stopped immediately as `verdict_determined`.
- The frozen `cd0c04f` runtime is now running the sole concurrency-1 batch
  `qwen35-full20-cd0c04f`. Any engineering error stops that batch for generic root
  cause repair; the 20 cases remain evaluation-only.
- The first three completed cases all have successful canonical traces and pass
  strict scheduler audit with zero scheduler, protocol, or route-control rejection.
  They exercise immediate evidence determination, meaningful-route exhaustion, and
  the 24-action cap. The largest observed provider request remains below 128K; the
  roughly 0.8M prompt-token totals are sums across independent short requests rather
  than one inherited conversation. A separate incremental audit guard now stops only
  this batch if a later final trace contains an engineering error or fails strict
  audit.

## 2026-07-22 — repeated canonical discrepancy

- `qwen35-monarch-atomic-9ac16e2` accepted an earlier medium-Claim biological
  discrepancy and continued on the still-open high Claim. At a later checkpoint,
  Qwen repeatedly submitted that prior discrepancy as a new update and exhausted
  corrections while trying to recreate its already-recorded assessment.
- Duplicate detection now uses only structured identity: affected Claim IDs,
  Evidence IDs, materiality, and status. It does not compare prose or infer
  semantics. The duplicate is rejected with its canonical discrepancy ID and the
  rest of the output is checked as though that repeated field were omitted, avoiding
  misleading cascade errors.
- No prior state is deleted, no model output is silently rewritten, and Evidence,
  verdict, and retry gates remain unchanged.

## 2026-07-22 — Monarch atomic Decision correction

- `qwen35-monarch-querycap-eef68a0` confirmed that Planning now persists valid
  hypotheses and derived text-search capability, then failed at its first sparse
  Decision after two investigation actions.
- The first response had several independent ownership/anchor errors. The second
  correctly retained only the qualified high-Claim refutation but omitted the
  required discrepancy; the final response added a valid decisive discrepancy yet
  left `verdict_proposal=continue`.
- The fixed retry count and all Evidence gates remain unchanged. Validator feedback
  now states the coupled atomic consequence once: high-Claim refutation, decisive
  established discrepancy, and `fake` belong in the same complete JSON object.

## 2026-07-22 — full-20 Planning contract failure

- `qwen35-full20-e82bd88` completed three cases without engineering errors, then
  stopped on the fourth case after Planning exhausted its fixed corrections.
- The hypothesis contained valid web queries but omitted `text_search` from a
  second, redundant tool list. Qwen then appended the literal tool name to query
  strings instead of repairing the cross-field duplication.
- The generic repair makes non-empty `queries` the single declaration of a
  text-search route. The reducer derives the bounded runtime capability without
  rewriting query text; schema and semantic gates remain unchanged. A final
  parseable but schema-invalid JSON object is now retained verbatim with its exact
  Pydantic rejection reason.

## 2026-07-22 — first frozen 20-case launch stopped after two failures

- The frozen `qwen35-full20-c1797c1` launch was stopped after its first two cases
  both exhausted the fixed Discrepancy Decision correction budget. No completed
  work was discarded; both failure traces remain durable.
- One output repeatedly assessed a medium Claim whose reviewed Evidence had no
  Finding chain for that Claim. The other first misspelled a Claim ID, while the
  fail-fast top-level check hid two additional assessment errors and an illegal
  `real` proposal with open high-salience routes.
- The repair completes, rather than weakens, contract diagnostics: all detectable
  Claim IDs, Evidence/Finding directions, discrepancy links, and verdict/route
  preconditions are returned together. Final semantically rejected JSON is now
  persisted verbatim as `output_rejected`. Correction count and semantic gates are
  unchanged.

## 2026-07-22 — heterogeneous Decision contract failure

- In `qwen35-heterogeneous3-753090e`, Andreea and Pillars completed without
  engineering errors and passed strict audit; Monarch failed after three rejected
  Discrepancy Decision outputs.
- The first Monarch output simultaneously cited visual/hypothesis IDs as Evidence,
  inverted a support Evidence stance into `refuted`, and tried to establish a
  decisive discrepancy without a qualified refute chain. The fail-fast reducer
  exposed only one error per correction, exhausting the fixed correction budget.
- The generic repair defines each planned ImageClaim as the positive real-world
  proposition communicated by the image, forbids ready-made fact-check queries,
  and reports all independently detectable reference, ownership, direction, and
  Finding-chain errors in one correction response. Evidence semantics and the
  correction budget remain unchanged.

## 2026-07-22 — profile-owned endpoint canary accepted

- `qwen35-case04886-profile-6568a61` ran with all Qwen endpoint/model environment
  overrides unset and still reached a bounded `real` Judgment in 319.65 seconds.
- The run had zero engineering errors, exactly 24 investigation actions, 40 model
  calls, 12 successful Evidence calls, and an explicit three-Claim unresolved basis.
- Strict scheduler audit passed with no warnings or rejections. All 27 context
  requests completed; the only correction had a valid parent request, and the
  maximum provider input was 56,158 tokens, below the 128K request limit.
- This accepts request ancestry, profile-owned endpoint resolution, and the clean
  forced boundary for the first heterogeneous case. Andreea, Pillars, and Monarch
  remain the pre-20-case heterogeneous gate.

## 2026-07-22 — Qwen correction-lineage canary

- `qwen35-case04886-basis-d08e6e0` completed with `real`, zero engineering errors,
  24 investigation actions, 44 model calls, and a non-empty bounded basis.
- Maximum provider input for one request was about 53.8K tokens, below the 128K
  target; protected context coverage remained complete in observed handoffs.
- Strict audit rejected four semantically recovered Decision attempts because the
  auditor required Gemini Interaction ancestry. The locked repair records Qwen
  request lifecycle and parent request IDs, then audits correction chains
  transitively without relaxing semantic validators.
- The first post-fix launch exposed an old 8899 fallback while Qwen3.5 served on
  8901. After an explicit endpoint launch, a second run reached 14 actions but a
  duplicate-call correction loop ended in a tool-bearing forced-output request.
  The next locked repair makes endpoints profile-owned, returns exact duplicate
  arguments, and suppresses tools on the schema-bound final boundary.

## 2026-07-17 — bootstrap

- Baseline commit: `5f4ac7e`.
- Structural diagnosis: original-image information is compressed before Target
  Planning and is unavailable to later policy stages.
- H1 is locked before implementation: attach the original image to every semantic
  and action-selection boundary without changing retrieval or evidence policy.
- H2 remains conditional: unify target revision, evidence assessment, replanning,
  and stopping only if H1 confirms that frozen target ownership is the remaining
  failure.

## 2026-07-17 — H1 implementation adjustment

- Literature and API review rejected repeated image upload on every policy call.
- Target Planning now attaches the original image once and creates one stored main
  Interactions chain.
- ReAct, Evidence Decision, Reflection, and Judgment continue that chain through
  `previous_interaction_id`.
- Query Concept Extraction, Query Replan, OCR, webpage extraction, and tool-internal
  VLM calls remain independent.
- A pending tool `function_result` is combined with the next stage as an official
  `user_input` step.
- Persisted policy snapshots replace binary image data with a `runtime_image`
  reference.
- Deterministic result: `332 passed`.

## 2026-07-22 — frozen full20 route-selection exhaustion

- `qwen35-full20-cd0c04f` completed four canonical cases before the incremental
  guard stopped the batch. All four had zero engineering errors and passed strict
  scheduler audit with zero scheduler/protocol/route-control rejection. Their
  exits covered evidence-determined `real`, meaningful-route exhaustion, a
  24-action `real`, and a 24-action `fake` with an image/ecology discrepancy.
- The fifth case, `case_3b6245ba631c3d94`, failed after 20 accepted investigation
  actions. Qwen repeatedly selected already executed search/recall routes in one
  ReAct segment. Runtime correctly rejected the duplicates, but after four
  corrections the old forced-output request rejected the model's checkpoint output
  with `min_tool_calls=1` and escalated the segment to an engineering error.
- The locked generic repair changes no Prompt, query, Evidence, duplicate, action,
  or verdict gate. v4 ReAct alone may close an exhausted route-selection correction
  chain with a deterministic boundary: no additional provider request, no invented
  tool action, and no `action_count` increment. The active Task becomes `blocked`,
  its hypothesis becomes `exhausted`, and a non-recoverable `protocol_error` Failure
  returns canonical state to standalone Discrepancy Decision.
- Strict audit accepts the boundary only through explicit rejected request IDs and
  reports it as a warning. Scoring and policy export still exclude every episode
  containing protocol rejection, so this fallback cannot become preferred SFT data.
  The same case rerun on `fa05685` completed with `termination=success`,
  `num_errors=0`, 24 investigation actions, 36 LLM calls, and strict audit passed
  with zero scheduler/protocol/route-control rejection. This run did not need the
  fallback: it reached normal `hard_budget_exhausted` binary Judgment after the
  former repeated-route segment, confirming the old engineering error is gone.

## 2026-07-22 — full20 Decision correction exhaustion

- The fresh `qwen35-full20-d05d672` batch was stopped at the third case as soon as
  its runtime emitted an `engineering_error` snapshot. The first two cases had
  reached final-state snapshots with no such event; no later cases were allowed to
  run.
- `case_17bcbf3a0705ce2e` reached a Discrepancy Decision with several independent
  validator errors. After the model corrected most fields, its final retry left
  `material_discrepancy.statement` empty, so schema validation rejected the update
  and the old pipeline raised `Discrepancy Decision did not produce a valid atomic
  update`. This is semantic-checkpoint protocol exhaustion, not provider or
  retrieval failure.
- The generic repair opts v4 Discrepancy Decision into the same deterministic
  bounded boundary used by ReAct. After the last rejected schema/semantic retry it
  sends no third provider request and returns a no-op `continue` Decision with an
  explicit rationale. It cannot invent IDs, Evidence, discrepancies, or a binary
  verdict; normal route-exhaustion and 24-action settlement remain unchanged.
- Local StageRunner, reducer, audit, compile, and full-test gates are the next
  prerequisite. Then only this failed case is rerun on the new commit; a clean
  single-case audit is required before restarting full20 from a new directory.

## 2026-07-22 — Decision-boundary case regression accepted

- `qwen35-case17-decision-boundary-b5b6f1d` completed with `termination=success`,
  `num_errors=0`, predicted `fake`, 24 investigation actions, and 59 model calls.
- Strict scheduler audit passed with zero failures and zero
  scheduler/protocol/route-control rejection. The trace contains two accepted
  Decision checkpoints and ten explicit bounded ReAct fallback warnings; no
  provider or runtime exception escaped the controlled lifecycle.
- The new no-op Decision boundary therefore handles the original empty
  `material_discrepancy.statement` failure without weakening any Evidence,
  ownership, duplicate, or binary-verdict gate. Because protocol rejections are
  present, scorer/exporter correctly excludes this trace from preferred training
  data. A fresh full20 run is now allowed.

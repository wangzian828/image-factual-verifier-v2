# Image-Centric Runtime Research Log

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

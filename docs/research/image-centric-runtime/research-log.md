# Image-Centric Runtime Research Log

## 2026-07-22 — Qwen correction-lineage canary

- `qwen35-case04886-basis-d08e6e0` completed with `real`, zero engineering errors,
  24 investigation actions, 44 model calls, and a non-empty bounded basis.
- Maximum provider input for one request was about 53.8K tokens, below the 128K
  target; protected context coverage remained complete in observed handoffs.
- Strict audit rejected four semantically recovered Decision attempts because the
  auditor required Gemini Interaction ancestry. The locked repair records Qwen
  request lifecycle and parent request IDs, then audits correction chains
  transitively without relaxing semantic validators.

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

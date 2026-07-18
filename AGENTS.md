# Image Factual Verifier v4 Contributor Guide

## Source of truth

Read these active documents before changing the runtime:

- `docs/superpowers/plans/2026-07-17-discrepancy-first-v4.md`
- `docs/superpowers/plans/2026-07-17-v4-discrepancy-first-implementation-plan.md`
- `docs/gemini-interaction-sequence.md`
- `docs/architecture.md`
- `docs/runtime-release-contract.md`

The July 14 implementation plan is a superseded historical record. The implementation
and contract tests win when an old research note disagrees.

## Supported boundary

The default benchmark path is v0.3 `image_only` with
`decision_policy_version=discrepancy-first-v4`.

Public rows contain exactly `case_id`, `image_path`, and `image_sha256`.
Do not add claim fields, claim modes, nullable placeholders, construction metadata, or
gold. Do not restore the removed claim-driven runtime.

The tagged v3 runtime is frozen at `runtime-v3-final-20260717`. Legacy v3 models and
reducers may remain for deterministic historical replay while migration finishes,
but they must not enter the v4 default path.

## Required control flow

1. Validate and hash-check `ImageOnlyRuntimeCase`.
2. Run Gemini `perceive_scene` and positioned OCR.
3. Deterministically bootstrap literal visual facts and retrieval anchors.
4. Create the main Interaction root with the original image and run Image Account
   Planning for one to three ImageClaims plus bounded SearchHypotheses.
5. Create claim/hypothesis-owned ResearchTasks and execute one bounded native tool
   call per ReAct action.
6. Keep Discovery separate from provenance-complete Evidence.
7. Run sparse multimodal Discrepancy Decision checkpoints after qualified Evidence,
   at material boundaries, and before unresolved termination.
8. Apply claim assessment, discrepancy, hypothesis, visual-reinspection, and verdict
   proposals atomically through deterministic reducers.
9. Stop immediately on an admissible decisive discrepancy, close all supported
   high-salience claims for real, or close unresolved high-salience routes for
   unverifiable.
10. Compile a claim/discrepancy/Evidence basis and require v4 Judgment to match it.

## Non-negotiable invariants

- Gemini LLM and vision use the Interactions API. Never switch wire protocols or
  providers after an error.
- Tool-bearing turns use native `function_call` / `function_result` with
  `previous_interaction_id`; at most one tool call is accepted per v4 action turn.
- Gemini sees the image in `perceive_scene`. Positioned OCR is a separate observation;
  low-confidence text is isolated and decisive small text may require focused visual
  verification. Manual image inspection is not part of the runtime.
- A tool result is a JSON object with `status=success|error`. Tool errors and malformed
  output are not Evidence.
- Search snippets, titles, reverse-image matches, and generated summaries are
  Discovery only.
- Web Evidence requires a fetched exact span, offsets, canonical URL, artifact SHA-256,
  retrieval time, directness, stance, and successful function-call provenance.
- A Finding must link one ResearchTask and owned fact/evidence IDs.
- A verdict basis must follow
  `VisualFact -> Finding -> Evidence -> successful tool call`.
- No external SearchHypothesis owns a verdict. ImageClaims and accepted
  MaterialDiscrepancies are the v4 semantic state.
- Reliable text may close ecological, geographic, temporal, or other world
  relations. A same-capture bridge is mandatory only when the semantic decision says
  the conclusion depends on binding a source assertion to this exact input image.
- `fake` requires an established decisive, Evidence-backed, visually anchored
  discrepancy affecting a high-salience claim. `real` requires every high-salience
  claim supported and its meaningful routes closed. `unverifiable` requires an
  unresolved high-salience claim and exhausted meaningful routes.
- A Discrepancy Decision may add or retire bounded SearchHypotheses and request one
  focused visual reinspection, but it cannot silently expand an ImageClaim.
- General VLM consistency/anomaly opinions are diagnostic and cannot create verdict
  Evidence.
- `text_search` accepts one query, `visit` one URL, and
  `reverse_image_search` one explicit Lens or semantic branch per action.
- If all investigation tools fail, or a provider/protocol/runtime boundary fails, stop
  with an engineering error before Judgment.
- Keep evaluator-private gold out of runtime/model state. Load it only after all
  rollouts for scoring.
- Apply `SourceAccessPolicy` before blocked rows or pages can enter tool output or model
  context.
- Persist redacted canonical JSON. HTML is derived diagnostics only.
- Keep credentials in environment variables or untracked `.env` files.
- Do not add face detection, embeddings, biometric matching, or a face-identity store.

## Active modules

- `src/workflow.py`: default v4 public workflow and trace persistence.
- `src/orchestrator/pipeline.py`: perception, Image Account Planning, v4 ReAct,
  Discrepancy Decision, and Judgment.
- `src/orchestrator/state.py`: canonical image-only runtime state.
- `src/orchestrator/investigation_models.py`: strict VisualFact state schemas.
- `src/orchestrator/bootstrap.py`: deterministic brief/fact/task bootstrap.
- `src/orchestrator/task_store.py`: atomic v4 reducers, action reduction, and routes.
- `src/orchestrator/discrepancy_coverage.py`: v4 Coverage and verdict basis.
- `src/orchestrator/stage_runner.py`: native Interactions protocol and tool execution.
- `src/eval/release_adapter.py`: immutable v0.3 release consumer.
- `src/eval/run_eval.py`: rollout, post-rollout scoring, and artifacts.
- `src/trajectory/`: policy export and process scoring.
- `scripts/audit_real_trace.py`: strict image-only trace audit.

Use `STAGE_TOOLS` in `src/orchestrator/tool_registry.py` as the active tool registry.

## Validation discipline

Run focused tests while editing, then:

```powershell
python -m pytest -q
python -m compileall -q src scripts
git diff --check
```

Pytest and scripted trajectories are not live acceptance. Runtime acceptance requires
`scripts/run_real_canary.py` with real Gemini/search/upload/browse/visual providers and
`scripts/audit_real_trace.py --strict-scheduler`.

Preserve unrelated user changes, keep commits scoped, and never modify the separate
data-pipeline repository from this worktree.

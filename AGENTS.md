# Image Factual Verifier v3 Contributor Guide

## Source of truth

Read these active documents before changing the runtime:

- `docs/architecture.md`
- `docs/agent-prompt-and-runtime-guide.md`
- `docs/runtime-release-contract.md`
- `docs/superpowers/plans/2026-07-16-search-control-and-tool-correctness-audit.md`

The July 14 implementation plan is a superseded historical record. The implementation
and contract tests win when an old research note disagrees.

## Supported boundary

The only supported benchmark path is v0.3 `image_only` with
`decision_policy_version=reinspect-v2`.

Public rows contain exactly `case_id`, `image_path`, and `image_sha256`.
Do not add claim fields, claim modes, nullable placeholders, construction metadata, or
gold. Do not restore the removed claim-driven runtime.

`reinspect-v2` is the current v3 decision-policy name, not compatibility with an older
project version.

## Required control flow

1. Validate and hash-check `ImageOnlyRuntimeCase`.
2. Run Gemini `perceive_scene` and layered `ocr_with_position` (configured PP-OCR
   service with EasyOCR fallback).
3. Deterministically build an immutable `InvestigationBrief`, pixel-grounded
   `VisualEntity`/`VisualFact` records, retrieval anchors, and at most four initial
   `ResearchTask` records.
4. Let Target Planning establish one image-grounded `CoreVerdictFact`; do not run a
   mandatory first reverse-image search.
5. Let Gemini choose one native Interactions function call per action turn. One
   policy action performs one bounded semantic operation.
6. Deterministically reduce each accepted action into separate Discovery, Evidence,
   Finding, Failure, task, and fact state.
7. After each accepted action, deterministically decide whether a sparse semantic
   Evidence Decision checkpoint is needed. Run it after potentially decisive
   inspected Evidence, before Reflection, or before an unresolved terminal outcome.
8. Let Gemini classify the active proposition as
   `supported|refuted|conflicted|insufficient`, then run deterministic Coverage.
9. Run structured Reflection after cumulative actions 4, 8, 12, 16, 20, and 24. It
   may reorder or add bounded routes for the same core fact, but cannot change verdict
   ownership.
10. Stop immediately when the core fact resolves, when no executable core route
   remains, after two action checkpoints without qualified core progress, or at the
   24-action cap.
11. Compile the only allowed verdict and basis, then require Gemini Judgment to match
   them exactly.

## Non-negotiable invariants

- Gemini LLM and vision use the Interactions API. Never switch wire protocols or
  providers after an error.
- Tool-bearing turns use native `function_call` / `function_result` with
  `previous_interaction_id`; at most one tool call is accepted per v3 action turn.
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
- Exactly one `CoreVerdictFact` owns the verdict. Optional title, creator, date,
  platform, asset ID, second-source, and general visual-integrity details are
  supporting by default and cannot delay a resolved verdict.
- Reliable text may close ecological, geographic, temporal, or other world
  relations. A same-capture bridge is mandatory only when the semantic decision says
  the conclusion depends on binding a source assertion to this exact input image.
- `fake` requires the core fact to be refuted after conflict adjudication; `real`
  requires the core fact to be supported with its required source/image binding; all
  other valid factual outcomes are `unverifiable` because evidence is insufficient.
- Reflection cannot replace the core fact. At most one Evidence Decision may narrow
  an unknown visible subject/place/event slot while preserving the original relation,
  same salient subject, pixel/OCR anchors, and newly reviewed Evidence. It cannot
  replace location/event scope or promote creator/title/date/platform metadata.
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

- `src/workflow.py`: v3-only public workflow and trace persistence.
- `src/orchestrator/pipeline.py`: perception, ReAct, Reflection, Coverage, Judgment.
- `src/orchestrator/state.py`: canonical v3 runtime state.
- `src/orchestrator/investigation_models.py`: strict VisualFact state schemas.
- `src/orchestrator/bootstrap.py`: deterministic brief/fact/task bootstrap.
- `src/orchestrator/task_store.py`: action reducer and bounded Reflection transitions.
- `src/orchestrator/coverage.py`: one-core-fact coverage and verdict basis.
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

# Image Factual Verifier v3 Contributor Guide

## Source of truth

Read these active documents before changing the runtime:

- `docs/architecture.md`
- `docs/agent-prompt-and-runtime-guide.md`
- `docs/runtime-release-contract.md`
- `docs/superpowers/plans/2026-07-14-visual-fact-search-agent.md`

The implementation and contract tests win when an old research note disagrees.

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
2. Run Gemini `perceive_scene` and EasyOCR `ocr_with_position`.
3. Deterministically build an immutable `InvestigationBrief`, pixel-grounded
   `VisualEntity`/`VisualFact` records, retrieval anchors, and at most four initial
   `ResearchTask` records.
4. Run one initial reverse-image search as Discovery only.
5. Let Gemini choose one native Interactions function call per action turn.
6. Deterministically reduce each real action into separate Discovery, Evidence,
   Finding, Failure, task, and fact state.
7. Run structured Reflection after cumulative actions 4, 8, 12, 16, 20, and 24.
8. Audit decisive facts and stop on coverage, two low-gain Reflection intervals, or
   the 24-action cap.
9. Compile the only allowed verdict and basis, then require Gemini Judgment to match
   them exactly.

## Non-negotiable invariants

- Gemini LLM and vision use the Interactions API. Never switch wire protocols or
  providers after an error.
- Tool-bearing turns use native `function_call` / `function_result` with
  `previous_interaction_id`; at most one tool call is accepted per v3 action turn.
- Gemini sees the image in `perceive_scene`. EasyOCR is a separate deterministic OCR
  tool. Manual image inspection is not part of the runtime.
- A tool result is a JSON object with `status=success|error`. Tool errors and malformed
  output are not Evidence.
- Search snippets, titles, reverse-image matches, and generated summaries are
  Discovery only.
- Web Evidence requires a fetched exact span, offsets, canonical URL, artifact SHA-256,
  retrieval time, directness, stance, and successful function-call provenance.
- A Finding must link one ResearchTask and owned fact/evidence IDs.
- A verdict basis must follow
  `VisualFact -> Finding -> Evidence -> successful tool call`.
- `fake` requires a decisive refuted fact; `real` requires every decisive fact to be
  supported; all other valid factual outcomes are `unverifiable`.
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
- `src/orchestrator/coverage.py`: decisive-fact coverage and verdict basis.
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

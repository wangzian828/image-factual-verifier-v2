# Image Factual Verifier v2 Contributor Guide

## Source Of Truth

Read `docs/architecture.md` before changing the workflow. It documents the active runtime contract, configuration, tests, and trace format. Use the [Agent Prompt and Runtime Guide](docs/agent-prompt-and-runtime-guide.md) for the end-to-end flow and the boundary between model prompts, deterministic orchestration, tool-internal model calls, and validation gates. Prefer the implementation and contract tests when a comment or an old report disagrees with those documents.

The supported production path is a multi-stage Gemini agent using the Gemini Interactions API. Generic backend adapters may remain for isolated tests or non-production integrations, but they are not a runtime fallback for the active workflow.

## Required Architecture

Preserve this control flow:

1. **Case and Perception**: construct or accept a strict `VerificationCase`, verify its image SHA-256, then run scene perception and positioned OCR to produce `PerceptionReport`.
2. **Planning**: have Gemini produce a validated `VerificationPlan` with unique question IDs, immutable declarative `claim_text` values, and at least one priority-1 question; compile claim text, not open-ended search questions, into atomic claim slots.
3. **Iterative native ReAct and ReInspect**: let Gemini select tools through Interactions `function_call` items, execute them locally, return `function_result` plus the deterministic observation update, and continue through `previous_interaction_id`.
4. **Coverage Audit / Replanning loop**: audit decisive claim slots through their linked plan questions. If coverage is incomplete and budget remains, revise only unresolved questions and retain prior steps, ledgers, and evidence.
5. **Ledger Judgment**: after either all decisive claims resolve or the bounded investigation budget ends, validate a `LedgerJudgment` against exact claim/evidence IDs. Unresolved factual slots yield typed `unverifiable`; engineering failures still raise before Judgment.

This is an agent, not a fixed tool script. Do not replace verification with a predetermined tool sequence or a single model call. Perception is intentionally deterministic; planning, tool choice, replanning, and judgment remain model-driven within validated boundaries.

## Non-Negotiable Invariants

- Use Gemini Interactions for the active Gemini LLM and vision path. Gemini vision must fail when the configured wire protocol is not `interactions`.
- Use native Interactions function calling in tool-bearing agent stages. Send tool schemas on the first interaction, execute returned calls, send `function_result`, and preserve the chain with `previous_interaction_id`.
- Do not translate native calls into prompt tags, silently switch to Chat Completions or Responses, change providers, or synthesize a heuristic plan/judgment after a protocol failure.
- Retry only the same Interactions request for documented transient HTTP statuses or transport errors. Exhaustion, malformed success payloads, missing interaction IDs, invalid required-action responses, and invalid final structured output are hard failures.
- Reject ungrounded conclusions. Verification needs a successful tool result; evidence must map to an actual tool step and an investigation question. A tool error is not evidence.
- Require every tool to return a JSON object whose `status` is exactly `success` or `error`; an error result also needs a non-empty `error`. Reject malformed or statusless results instead of inventing success semantics.
- Keep `VerificationCase` free of benchmark gold. `external_claim` requires runtime `user_claim`; `embedded_claim` must recover `claim_surface` from pixels/OCR. Always verify `image_sha256`, and keep `decision_policy_version` aligned with the active `reinspect-v1` validator.
- Treat `claims`, `sources`, `evidence`, `discoveries`, and `failures` as separate machine-verifiable collections. Preserve stable IDs, exact tool-call provenance, source family/risk metadata, artifact hashes, retrieval times, and recovery links.
- Keep discovery separate from evidence. Search snippets, result titles, reverse-image matches, and generated summaries are leads only. They cannot close a claim or be cited by Judgment.
- Promote web evidence only from a fetched page and an existing selected passage. Require an exact span and matching offsets, canonical URL, artifact SHA-256, ISO-8601 retrieval time, directness, valid stance/relevance, `evidence_eligible=true`, and no injection flags. A multi-query, multi-page search call may promote multiple independently eligible passages, at most one selected passage per fetched page; do not collapse them into one summary. Never accept model-invented or merged passage text.
- Keep ReInspect evidence-conditioned. A search-created `VisualQuestion` must identify exactly one source evidence/discovery ID, normalized target box, expected property, and allowed real visual action. Matching OCR/crop/count/reference calls resolve or fail it. A pending ReInspect may hold only a linked non-`external_fact` claim open; an external fact question is never ReInspect-gated and still requires eligible direct web evidence.
- Keep coverage and judgment deterministic. Claim status depends on direct evidence and source-family independence. `LedgerJudgment` must decide every decisive claim using only matching ledger IDs; `real`, `fake`, and typed `unverifiable` follow the `reinspect-v1` policy.
- Require an explicit `question_id` on verification function calls. Never silently assign a call to a question.
- In benchmark evaluation, enforce the provenance-derived `SourceAccessPolicy` before search results are enriched or returned and before any direct page/reference fetch. Do not expose excluded URLs/domains or hidden gold to the model. Product mode remains unrestricted.
- Reject benchmark search queries that explicitly target an excluded fact-check domain before calling the search provider; the model must reformulate toward independent open-web sources.
- Ground browse stance to immutable `claim_text`, keep model queries as retrieval parameters only, require every active priority-1 question to receive one tool attempt before resampling, and give every priority-2 question one real attempt before further P1 resampling or iteration output.
- Select upload, visual-search, and browse-fetch providers explicitly. A selected provider's failure must propagate; do not fall through to another provider.
- Sanitize credentials, secret fields, and signed-URL authentication parameters before writing traces, HTML, or cache entries.
- Persist canonical JSON traces by default. HTML is a derived diagnostic view and is generated only through `src.render_trace_html` when requested.
- Keep the disk tool cache opt-in. It is disabled by default and must remain bounded by TTL and namespace when enabled.
- Read credentials only from environment variables or an untracked `.env`. Never place keys in source, configuration, prompts, traces, tests, or documentation.
- Do not add a dedicated face detector, face embedding store, biometric recognition model, or biometric similarity tool. Person-identity claims remain in scope when investigated through reverse-image search, original-source captions, public reporting, visible non-biometric cues, and event context.

## Active Modules

- `src/workflow.py`: public workflow wrapper and canonical JSON trace export.
- `src/orchestrator/pipeline.py`: stage ordering, verification iterations, coverage audit, replanning, and output validation.
- `src/orchestrator/stage_runner.py`: bounded ReAct loop and native Interactions function-call round trips.
- `src/orchestrator/state.py`: `VerificationCase`, ledger records, stage outputs, and aggregate trace state.
- `src/orchestrator/ledger.py`: case construction/hash checks, ledger compilation, claim status, and typed insufficiency reasons.
- `src/orchestrator/investigation_state.py`: per-observation belief deltas, visual questions, regional observations, and stopping assessments.
- `src/orchestrator/source_provenance.py`: URL canonicalization, source-family/class assignment, and risk flags.
- `src/orchestrator/context.py`: compact context passed between stages.
- `src/orchestrator/tool_registry.py`: active stage-to-tool allowlists and tool health.
- `src/integrations/gemini/interactions.py`: environment-authenticated Interactions REST contract and retry policy.
- `src/integrations/browse/jina_reader.py`: selected fetch provider, deterministic passage candidates, exact-span extraction, and injection gating.
- `src/trace_viewer.py` and `src/render_trace_html.py`: standalone trace rendering.

Do not infer the active tool set from every file under `src/tools/`; use `STAGE_TOOLS` in `src/orchestrator/tool_registry.py`.

## Configuration And Credentials

The normal local setup is:

```powershell
python -m pip install -e ".[dev]"
$env:GEMINI_API_KEY = "..."
$env:SERPER_API_KEY = "..."
python -m src path\to\image.jpg
```

`GOOGLE_API_KEY` is accepted as the Gemini credential alias. `GEMINI_WIRE_API` should be unset or `interactions`; `AGENT_LLM_WIRE_API` and `VISION_LLM_WIRE_API` must not redirect the active Gemini path to another protocol. `GEMINI_INTERACTIONS_URL` may override the endpoint for a compatible deployment. Search and browsing integrations may require `SERPER_API_KEY` and `JINA_API_KEY`; proxy settings come from `HTTPS_PROXY` / `HTTP_PROXY`.

Set the integration selectors explicitly in deployed environments:

```dotenv
IMAGE_UPLOAD_PROVIDER=oss
VISUAL_SEARCH_PROVIDER=serper_lens
BROWSE_FETCH_PROVIDER=jina
```

Accepted values are `oss|custom|temp`, `serper_lens|zhipu_image_search`, and `jina|direct`, respectively. The defaults shown above preserve local defaults but do not permit provider fallback.

The disk cache is off unless explicitly enabled:

```dotenv
TOOL_CACHE_ENABLED=0
TOOL_CACHE_DIR=.cache/tool_results
TOOL_CACHE_TTL_SECONDS=3600
# Optional override; otherwise the namespace includes the tool contract, models, and provider selectors.
TOOL_CACHE_NAMESPACE=
```

Verification defaults to 12 native ReAct turns per iteration and `MAX_VERIFICATION_ITERATIONS=4`, with `MIN_VERIFICATION_ITERATIONS=2` and `LOW_INFORMATION_GAIN_PATIENCE=2`. The outer loop stops only on decisive coverage plus required P2 service, two consecutive low-gain iterations with no pending ReInspect, or the hard cap; traces distinguish these outcomes. `GEMINI_VERIFICATION_MAX_OUTPUT_TOKENS` defaults to `16384`, and the forced schema serialization uses `GEMINI_VERIFICATION_FINAL_MAX_OUTPUT_TOKENS=32768`. Every active Gemini agent, browse, and visual Interactions call uses `thinking_level=minimal`; record thought-token usage and treat any non-zero count as a configuration defect. Gemini calls made inside tools are separate API calls and their prompt, completion, and thought tokens must be included in aggregate call/token accounting even though their private runtime metrics are removed from model-facing tool JSON. Do not retry a failed call by changing thinking level, model, provider, or protocol. `GEMINI_VISION_MIN_OUTPUT_TOKENS` defaults to `8192`, `GEMINI_VISION_TIMEOUT_SECONDS` to `240`, and `BROWSE_EXTRACT_MAX_OUTPUT_TOKENS` to `4096`.

Keep all secrets environment-only even when adding controls.

## Validation

Run the focused suite for orchestration and Gemini protocol changes:

```powershell
python -m pytest -q test_unit.py test_evidence_grounding.py test_failure_contracts.py test_provider_failure_propagation.py test_native_interactions.py test_gemini_interactions_contract.py test_gemini_vlm_interactions.py test_full_native_agent_trace.py test_trace_viewer.py
python test_workflow_smoke.py
python scripts/probe_gemini_interactions.py --model gemini-3-flash-preview
```

The focused tests cover multi-round tool use, exact evidence grounding and passage selection, discovery/failure exclusion, case/ledger judgment, coverage-driven replanning and typed insufficiency, P1/P2 service, adaptive stopping, ReInspect state, native function-call round trips, required question IDs, scoped `thinking_level=minimal`, environment-only credentials, retry/error behavior, canonical JSON persistence, and on-demand HTML rendering. `test_full_native_agent_trace.py` remains a controlled two-iteration fixture, not the production limit or a real-world fact-check result.

For a real run, inspect the canonical JSON under `outputs/traces/`. Generate a standalone HTML diagnostic view only when needed:

```powershell
python -m src.render_trace_html outputs\traces --output-dir outputs\trace_html
```

Audit a real canonical trace, including aggregate accounting and scheduler invariants, before treating it as a valid end-to-end run:

```powershell
python scripts/audit_real_trace.py outputs\traces\example.json --json --strict-scheduler
```

Generated traces, caches, and logs are ignored. Do not place committed test fixtures under ignored output paths.

## Change Discipline

- Keep changes scoped and preserve concurrent work already present in the worktree.
- Add or update tests whenever stage contracts, function schemas, evidence validation, coverage logic, or trace serialization changes.
- Propagate failures with enough context to diagnose the failing endpoint, interaction, stage, or tool. Do not hide them behind an `unverifiable` result.
- Keep `AGENTS.md`, `CLAUDE.md`, and `docs/architecture.md` aligned with the active contract.

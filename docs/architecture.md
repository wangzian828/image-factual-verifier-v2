# Active Agent Architecture

## Runtime Contract

Image Factual Verifier v2 evaluates an image's apparent factual claim against visual and external evidence. The production path is a bounded multi-stage Gemini agent with this control flow:

```text
Perception
  -> Planning
  -> iterative native ReAct
  -> deterministic Coverage Audit
       -> decisive claims resolved: Ledger Judgment
       -> incomplete and budget remains: Replanning -> native ReAct -> Audit
       -> incomplete and budget exhausted: Ledger Judgment -> typed unverifiable
       -> engineering/protocol failure: error; no Judgment
```

`VerificationState` retains the versioned `VerificationCase`, perception, current plan and plan history, verification output, coverage audits, machine-verifiable ledgers, incremental `InvestigationState`, final judgment, stage steps, tool health, timings, token use, and call counts.

## Runtime Case Contract

Every run receives or constructs one strict `VerificationCase`:

```yaml
case_id: string
image_path: string
image_sha256: 64-character lowercase SHA-256
claim_mode: external_claim | embedded_claim
user_claim: string | null
claim_surface: string | null
claim_source_region: [x1, y1, x2, y2] | null
decision_policy_version: string
```

The workflow verifies `image_sha256` against the file before Perception. An `external_claim` requires a non-empty `user_claim`. An `embedded_claim` forbids `user_claim`; after Perception, visible OCR text is copied into `claim_surface` for Planning. A supplied `claim_source_region` is a normalized, non-empty box.

The runtime case must never contain a hidden benchmark label or gold `primary_claim`. For an embedded claim, the agent recovers the claim only from pixels, OCR, and scene observations. The active decision policy is `reinspect-v1`; versioning the field does not authorize a different policy without matching validator changes.

## Stages

### 1. Perception

`Orchestrator._run_perception()` directly runs:

- `perceive_scene` for visible entities, scene, and image type;
- `ocr_with_position` for visible text and coordinates.

Their merged `PerceptionReport` is a deterministic visual inventory for Planning. Gemini-backed perception uses Interactions only. The project has no face detection, face recognition, biometric analysis, or identity-matching functionality.

### 2. Planning

Planning receives compact perception context and returns a schema-validated `VerificationPlan`. It must contain at least one question, at least one priority-1 question, and a unique non-empty `question_id` for every question. Suggested tools and queries guide investigation but do not prescribe a fixed sequence.

Planning has no tools. For the active ledger contract, each decision-relevant question becomes an atomic `ClaimRecord` with ID `claim-<question_id>`; priority 1, 2, and 3 map to `decisive`, `supporting`, and `contextual`. `image_intent` remains a compact planning field, not a substitute for claim slots. Invalid structured output after bounded correction fails the stage; there is no heuristic plan fallback.

### 3. Iterative Native ReAct

`StageRunner` exposes only the verification allowlist from `src/orchestrator/tool_registry.py`:

| Capability | Tools |
| --- | --- |
| Runtime context | `current_time` |
| Discovery and retrieval | `reverse_image_search`, `text_search`, `visit` |
| Focused image work / ReInspect | `ocr_with_position`, `crop_and_search`, `crop_and_inspect`, `count_objects` |
| Comparison and consistency | `compare_with_reference`, `check_consistency`, `analyze_visual_anomalies` |

Within round and per-tool budgets, Gemini chooses each call from the plan and accumulated results. Every verification call needs a valid plan `question_id`; the runner records it as `__question_id` and never guesses it. Duplicate calls and exhausted tool budgets are rejected.

Every tool returns a JSON object with `status` exactly `success` or `error`. An error result also has a non-empty `error`. Malformed or statusless results fail the tool-result contract. Only successful recorded calls can ground evidence; errors remain diagnostic observations.

The active native loop is:

1. Create a Gemini interaction with context, native function schemas, and `store=true`.
2. Read returned `function_call` items.
3. Validate call IDs, names, arguments, question IDs, duplicates, and budgets.
4. Execute local tools and record each call or protocol error as a `StageStep`.
5. Send `function_result` items using the returned `previous_interaction_id` chain.
6. Continue until Gemini emits valid structured output or the round budget is exhausted.

Verification cannot finish without a successful tool call. Premature output is rejected within the same interaction chain when required tool work or priority coverage is missing.

#### Machine-Verifiable Ledgers

After every validated verification observation, and again at the end of an iteration, `compile_runtime_ledgers()` deterministically rebuilds `VerificationLedgers` from the case, plan, accepted evidence view, and immutable tool steps. Model prose is not allowed to mutate these records.

| Collection | Required role |
| --- | --- |
| `claims` | Atomic claim ID and question ID, criticality, `open|supported|refuted|conflicted|unverifiable` status, and the remaining distinction. |
| `sources` | Stable source ID, canonical URL/domain, source family and class, artifact SHA-256, retrieval time, dependency IDs, and risk flags. |
| `evidence` | Stable evidence ID linked to one claim, source, and successful `function_call_id`; exact web span, image region, or runtime anchor; artifact hash, time, stance, quality, and directness. |
| `discoveries` | Search or visual-reference candidates linked to a claim and call. They may point to a later promoted evidence ID, but are not verdict evidence themselves. |
| `failures` | Failed call ID, tool, optional claim, typed code, severity, recoverability, message, and optional recovery call. Failures never enter the evidence ledger. |

Evidence insertion checks that the claim and source exist, the call succeeded, the same call is not in the failure ledger, and the evidence artifact hash matches its source. Direct evidence can close a claim when it is a moderate/strong image observation, a moderate/strong official source, or corroboration from at least two independent non-UGC, non-risky source families. Opposing direct evidence makes the claim `conflicted`; one non-primary source family leaves it open.

`VerificationResult` remains the structured agent output and compatibility view, but Coverage and Judgment are gated by the compiled ledgers. Search snippets, titles, and generated summaries cannot be promoted merely because the model repeats them.

#### Discovery And Passage Selection

`reverse_image_search` results are discovery only. `text_search` and `crop_and_search` also record their search candidates as discoveries, but may separately yield evidence because they visit selected result pages. A standalone `visit` can promote a candidate URL by fetching the page and selecting a source passage.

The browse extractor normalizes page text, divides it into deterministic passages of at most 700 characters, and gives each passage an integer ID and exact character offsets. The extractor may select one existing `passage_id` or `-1`; it cannot provide, merge, edit, or invent the evidence text. The runtime copies the selected passage and computes its offsets and artifact SHA-256 itself. Unknown passage IDs, invalid relevance/stance/directness values, malformed JSON, empty responses, and failed fetches are errors rather than synthetic evidence.

A web record is ledger-eligible only when all of these checks pass:

- it has `evidence_eligible=true`, no prompt-injection flags, and `directness=direct`;
- the exact passage is non-empty and its offsets match its length;
- the artifact hash is a 64-character SHA-256 and retrieval time is valid ISO-8601;
- its canonical URL matches the evidence source and its stance/relevance are valid.

Blocked or injection-flagged pages remain failure or URL-only discovery observations. Their text must not steer the root claim, plan IDs, tool policy, schema, or verdict.

#### Incremental Belief And ReInspect

On the production Gemini Interactions path, `InvestigationReducer` reduces each tool result into an auditable `ObservationAssessment`, `BeliefDelta`, optional `VisualQuestion`, optional `RegionObservation`, and `StoppingAssessment`. The update is persisted on the step and returned to Gemini with the corresponding `function_result`.

Discovery produces a `create` delta without changing factual belief. Validated evidence may produce `support`, `refute`, or `split`; failures produce `unknown`; non-decisive or duplicate observations produce `zero`. Novel source families and repeated observations are tracked explicitly.

A reverse-image candidate with a reference image can create a full-image comparison question. Web evidence for a visually discriminative claim can create a regional question. Every `VisualQuestion` is linked to exactly one `source_evidence_id` or `source_discovery_id` and carries a normalized `target_bbox`, an `expected_property`, and recommended real visual tools. A follow-up call that supplies `visual_question_id` must match that pending question's source link, expected property, target box, and allowed ReInspect tool. Success or failure then records the regional observation and resolves or fails the question.

While such a visual question is pending, its claim is forced back to `open`; web or reverse-search output cannot stand in for the required image observation. Stopping assessments expose unresolved decisive claims, pending visual questions, source-family novelty, repeated observations, and remaining high-value actions. They are traceable state, while the deterministic Coverage Audit remains the iteration gate.

### 4. Coverage Audit And Replanning Loop

After each ReAct iteration, `_audit_plan_coverage()` deterministically maps claim-ledger status back to every plan question and marks it `unanswered`, `in_progress`, `resolved`, or `exhausted`. It records successful calls, distinct tools, grounded evidence counts, claim statuses, unresolved/exhausted priority IDs, and typed unverifiable reasons.

If the audit is incomplete and `MAX_VERIFICATION_ITERATIONS` allows another pass, Gemini revises the plan from the current gaps and retained evidence. Replanning preserves resolved questions, refines unresolved work, increments the revision, and starts another native ReAct iteration.

`complete` means all priority-1 claim slots are supported or refuted. `investigation_complete` means either that condition is met or the bounded iteration budget has ended. At the final budget, unresolved claim slots proceed to Ledger Judgment as `unverifiable`; this is a factual insufficiency result, not an engineering fallback. The default maximum is two verification iterations. Every-tool-failed runs, missing accepted structured output, and other engineering failures still raise before Judgment.

### 5. Judgment

Judgment runs only after the investigation reaches a valid bounded stopping state. The no-tool Gemini stage returns `LedgerJudgment`, and the validator treats the claim and evidence ledgers as the decisive factual contract.

Minimum deterministic rules are:

- there must be exactly one decision for every decisive `claim_id`;
- every cited `evidence_id` must exist, belong to that claim, have non-neutral matching stance, and appear in `selected_evidence_ids` exactly once as a set;
- a supported claim requires `support`, a refuted claim requires `refute`, and every other claim is `unresolved` with no decisive evidence IDs;
- any refuted decisive claim yields `fake`; all decisive claims supported yields `real`; otherwise the verdict is `unverifiable`;
- `unverifiable` must carry exactly the deterministic typed reasons, such as source conflict, single-source-family dependency, access limitation, absent decisive evidence, and budget exhaustion.

The active `policy_rule_id` is `reinspect-v1`. Final reasoning, key evidence, and assessment are recompiled from validated claim/evidence IDs, so free-form model text cannot introduce new facts. Invalid judgment output after bounded correction raises. Protocol, provider, malformed-output, and tool-system failures are engineering errors and are never relabeled as `unverifiable`.

## Failure Contract

All active Gemini LLM and vision requests use the Gemini Interactions API. The runtime never switches wire protocols or model providers, converts native calls to prompt tags, or synthesizes fallback plans, evidence, or judgments after failure.

Retries resend the same Interactions request only for transport errors and HTTP `429`, `500`, `502`, `503`, or `504`. Other HTTP errors fail immediately. Retry exhaustion, malformed success payloads, missing interaction IDs, invalid required actions, and invalid final structured output are hard failures.

The explicitly selected upload, visual-search, and browse-fetch providers also do not fall through after failure. `run_single()` propagates failures. `run_batch()` converts an individual exception to a `verdict: "error"` record so unrelated items can continue; this is batch isolation, not a factual verdict fallback.

## Configuration

Credentials are environment-only and may be loaded from an untracked `.env`. Never put secrets in source, committed configuration, prompts, traces, tests, or documentation.

```dotenv
GEMINI_API_KEY=...
SERPER_API_KEY=...

GEMINI_WIRE_API=interactions
# Optional compatible endpoint override:
GEMINI_INTERACTIONS_URL=https://generativelanguage.googleapis.com/v1beta/interactions
```

`GOOGLE_API_KEY` is accepted instead of `GEMINI_API_KEY`. Gemini credentials are sent as `x-goog-api-key`. `AGENT_LLM_WIRE_API`, `VISION_LLM_WIRE_API`, and `GEMINI_VISION_WIRE_API` must be unset or `interactions` for a Gemini path.

Select all external routing providers explicitly in deployed environments:

```dotenv
IMAGE_UPLOAD_PROVIDER=oss
VISUAL_SEARCH_PROVIDER=serper_lens
BROWSE_FETCH_PROVIDER=jina
```

Accepted values and implementation defaults are:

| Variable | Values | Default |
| --- | --- | --- |
| `IMAGE_UPLOAD_PROVIDER` | `oss`, `custom`, `temp` | `oss` |
| `VISUAL_SEARCH_PROVIDER` | `serper_lens`, `zhipu_image_search` | `serper_lens` |
| `BROWSE_FETCH_PROVIDER` | `jina`, `direct` | `jina` |

OSS upload requires `OSS_ACCESS_KEY_ID`, `OSS_ACCESS_KEY_SECRET`, `OSS_ENDPOINT`, and `OSS_BUCKET_NAME`; custom upload requires `IMAGE_UPLOAD_API_URL`; Zhipu search requires `ZHIPU_API_KEY`. Jina access may use `JINA_API_KEY`. `HTTPS_PROXY` / `HTTP_PROXY` configure network routing.

### Tool Cache

The disk-backed tool cache is opt-in and disabled by default:

```dotenv
TOOL_CACHE_ENABLED=0
TOOL_CACHE_DIR=.cache/tool_results
TOOL_CACHE_TTL_SECONDS=3600
# Optional override; normally leave unset.
TOOL_CACHE_NAMESPACE=
```

When enabled, only successful eligible results are cached. TTL limits reuse, and namespace isolates incompatible tool contracts, models, and provider selections. The automatic namespace includes those values unless `TOOL_CACHE_NAMESPACE` overrides it. Cache arguments and results are sanitized before disk persistence.

`MAX_VERIFICATION_ITERATIONS` defaults to `2`. Browse concurrency and extraction controls use the `BROWSE_*` variables defined by the integrations.

`GEMINI_VISION_MIN_OUTPUT_TOKENS` defaults to `8192`. Schema-bound Gemini image observations routed through `OpenAIVisionClient.create_image_json()` set `generation_config.thinking_level` from `GEMINI_VISION_THINKING_LEVEL`, whose default is `minimal`. These calls perform bounded visual extraction rather than planning, so the default preserves output budget for the required JSON. This setting is not a global agent thinking policy: Planning, native ReAct, Replanning, Ledger Judgment, browse passage selection, and direct Interactions tools such as reference comparison keep their own generation configuration. Invalid values or failed visual calls propagate; the runtime does not switch thinking modes, models, providers, or protocols as a fallback.

`BROWSE_EXTRACT_MAX_OUTPUT_TOKENS` defaults to `4096`, allowing the passage selector to finish its schema-bound output. `GEMINI_VISION_TIMEOUT_SECONDS` defaults to `240` and controls each Gemini image-observation request. All of these controls are environment-overridable without changing failure semantics.

## Trace Persistence

With tracing enabled, `VerificationWorkflow` writes sibling files under `outputs/traces/` or `--output-dir`:

- `<image-id>.json`: canonical machine-readable result and full state;
- `<image-id>.html`: required standalone diagnostic rendering.

Before any JSON, HTML, or cache write, persistence data is sanitized to redact credential fields and authentication parameters in signed URLs. The JSON trace is the source of truth, but HTML generation is not best-effort: rendering or write errors propagate from trace export.

Re-render saved traces without rerunning verification:

```powershell
python -m src.render_trace_html outputs\traces\example.json
python -m src.render_trace_html outputs\traces --output-dir outputs\trace_html
```

Do not commit generated traces or cache files.

When `IFV_DATA_ROOT` is set, default trace, evaluation, cache, benchmark-workbench,
and source-material paths are rooted there instead of inside the checkout. The gpu-13
wrapper sets it to `/gsdata/home/wza/image-factual-verifier-v2-data`; see
`docs/operations/gpu13.md` for the directory layout.

Probe the complete public Gemini Interactions contract without running an image case:

```powershell
python scripts/probe_gemini_interactions.py --model gemini-3-flash-preview --repeat 2
```

The probe uses the same environment-only credential path and verifies native structured
output, `function_call`, `function_result`, and `previous_interaction_id` continuation.

## Running And Tests

```powershell
python -m pip install -e ".[dev]"
python -m src path\to\image.jpg

python -m pytest -q test_unit.py test_failure_contracts.py test_native_interactions.py test_gemini_interactions_contract.py test_gemini_vlm_interactions.py test_trace_viewer.py
python -m pytest -q test_full_native_agent_trace.py
python test_workflow_smoke.py
```

The focused tests cover native call chaining, tool-result status validation, case/ledger judgment, discovery and exact-span grounding, passage-ID validation, incremental ReInspect state, provider non-fallback, cache TTL and namespace, coverage-driven replanning and typed insufficiency, Gemini-only Interactions, `thinking_level=minimal` for schema-bound image observations, persistence redaction, and required JSON/HTML export. `test_full_native_agent_trace.py` runs a controlled fixture through the complete production orchestrator and exports an auditable reference trace with two verification iterations, two coverage audits, replanning, Interaction parent/call IDs, ledger evidence, and ReInspect resolution. It is explicitly not a real-world factual result.

## Repository Map

```text
src/workflow.py                         workflow and required trace export
src/orchestrator/pipeline.py            stages, audit/replanning loop, validation
src/orchestrator/stage_runner.py        bounded native ReAct loop
src/orchestrator/state.py               state contracts and persistence view
src/orchestrator/ledger.py              case hashing and deterministic ledgers
src/orchestrator/investigation_state.py per-observation belief/ReInspect reducer
src/orchestrator/source_provenance.py   URL, source-family, and risk normalization
src/orchestrator/tool_registry.py       active stage tool allowlists
src/orchestrator/tool_result.py         success/error result contract
src/orchestrator/tool_cache.py          opt-in TTL/namespace cache
src/integrations/gemini/interactions.py Gemini Interactions client
src/integrations/browse/jina_reader.py  fetch, passage selection, and span provenance
src/redaction.py                        persistence sanitization
src/trace_viewer.py                     standalone HTML renderer
src/render_trace_html.py                JSON-to-HTML CLI
```

The active tool set comes from `src/orchestrator/tool_registry.py`, not from every module under `src/tools/`.

Dataset image generation also uses `GeminiInteractionsClient`. Image requests set
`response_format.type=image`, `aspect_ratio`, and `image_size`, and deliberately omit
`delivery`: the current image endpoint rejects both `inline` and `uri` delivery modes
in the tested configuration and returns image content inline when the interaction
completes. `scripts/generate_gpt_image_samples.py` defaults to the Gemini provider;
another provider is used only when explicitly selected and never as a failure fallback.
Video generation keeps its separate URI delivery contract.

## Server Operations

The verified gpu-13 deployment topology, Git-only source workflow, mandatory
`OMP_NUM_THREADS=1` guard, proxy configuration, isolated Conda bootstrap, and test
commands are documented in `docs/operations/gpu13.md`. Server checkouts are runtime
artifacts only: source changes are made locally, committed, pushed to GitHub, and then
fast-forwarded on the server.

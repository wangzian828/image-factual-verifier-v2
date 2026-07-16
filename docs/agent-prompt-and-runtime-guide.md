# Agent Prompt and Runtime Guide

This guide identifies which v3 decisions are model-authored and which are enforced by
deterministic code.

## Boundary map

| Stage | Decision owner | Input | Output |
| --- | --- | --- | --- |
| Release/case validation | Deterministic | v0.3 manifest and public row | Hash-verified `ImageOnlyRuntimeCase` |
| Scene perception | Gemini tool prompt | Image | Literal entities, boxes, scene, image type |
| OCR | PP-OCR service when configured, otherwise EasyOCR | Image or crop | Accepted positioned text plus isolated low-confidence candidates |
| Bootstrap | Deterministic | Case + perception/OCR | Brief, entities, facts, anchors, initial tasks |
| Target Planning | Gemini structured output + deterministic validator | Visible facts, OCR, anchors, tasks | One core fact and candidate evidence routes |
| ReAct | Gemini native function calling | Compact current investigation | One tool action, then segment output |
| Observation reduction | Deterministic | Tool result + task/fact state | Discovery/Evidence/Finding/Failure updates |
| Evidence Decision | Gemini structured output + deterministic validator | Active fact + qualified Evidence + pixel/OCR anchors | Semantic assessment, binding requirement, optional one-slot refinement |
| Reflection | Gemini structured output + deterministic validator | Global state every four actions | Bounded task-order and route delta |
| Coverage | Deterministic | Facts, Findings, Evidence, tasks, budget | Stop state and fact coverage |
| Verdict basis | Deterministic | One core fact and its evidence gaps | Allowed verdict and exact basis IDs |
| Judgment | Gemini structured output + deterministic validator | Allowed basis only | Matching `ImageOnlyJudgment` |
| Scoring/export | Deterministic, post-rollout | Trace + private references | Metrics, teacher score, policy examples |

## Perception

`perceive_scene` is a Gemini Interactions image request. It is instructed to report
literal visible content and normalized geometry, not infer evaluator gold.

`ocr_with_position` runs immediately afterward. It uses a configured
PP-OCR/PP-Structure-compatible service when available and falls back to a cached
EasyOCR reader. The runtime merges accepted regions with perception while retaining
low-confidence candidates separately. OCR strings remain observations; bootstrap
decides whether accepted text becomes a fact or retrieval anchor.

Failure of either required perception tool is an engineering error.

## Bootstrap

Bootstrap is not an LLM Planning stage. It deterministically:

- creates an inquiry-oriented immutable brief;
- converts visible entities and text into stable records;
- preserves pixel/OCR provenance;
- creates low-commitment candidate facts;
- chooses at most four initial tasks.

Target Planning then selects one image-grounded, evidence-routable
`CoreVerdictFact`. There is no mandatory first reverse-image search: Planning and
ReAct choose the first route from the open evidence gap. Reverse-image results, when
requested, remain Discovery.

The broad bootstrap `appears_to_depict` sentence is perception context only. It is
never promoted as a fallback core. Target Planning must establish exactly one
decisive atomic external-world or source-record proposition; failure after the
bounded correction round is recorded as an engineering error rather than silently
changing the evaluation target.

Because bootstrap is deterministic, it is not exported as a model-authored policy
action. Actual Target Planning is retained in the trace.

## ReAct prompt

The ReAct prompt tells Gemini to:

- choose one active `ResearchTask`;
- make exactly one native tool call;
- use the task ID as `question_id`;
- prefer unresolved priority-1 work and recommended tasks;
- treat snippets and reverse matches as Discovery;
- use fetched pages or successful visual observations for Evidence;
- compare a Discovery's `reference_image_url` directly before treating it as a visual
  match, pass the surrounding page as a fallback, and fetch its direct caption or
  context separately;
- propose Findings only with existing Evidence IDs;
- avoid duplicate routes;
- never output a verdict.

Planning's `suggested_tools` are first-hop advice, not a closed permission list.
A task-owned search Discovery deterministically enables `visit` for its candidate
page and `compare_with_reference` for its validated reference image. The next
segment exposes only the concrete uninspected route and binds its URL in the native
tool schema.

Source class is a soft provenance/risk signal, not a retrieval gate. Each new
retrieval batch exposes at most four concrete candidates, including unknown or UGC
pages, and Gemini selects one using the active question, title, snippet, and source
metadata. Inspecting one candidate consumes that batch; a later search batch may
still contribute a better page without requiring every SERP result to be visited.

The model receives compact active tasks, related fact statements, recent Discoveries,
eligible Evidence, Findings, Failures, action count, and next Reflection boundary.

## Native Interactions protocol

For a tool-bearing segment:

1. Create an Interactions request with system prompt, compact context, native function
   schemas, and `store=true`.
2. Read `function_call`.
3. Validate call ID, tool name, recursive argument schema, task ID, duplicate route,
   per-tool budget, source policy, and one-call-per-turn constraint.
4. Execute the selected tool.
5. Serialize `status=success|error`.
6. Reduce the observation into runtime state.
7. Send `function_result` with `previous_interaction_id`.
8. Run deterministic observation reduction.
9. At a material boundary, run a sparse Evidence Decision over accumulated qualified
   Evidence; otherwise retain the prior semantic decision.
10. Run Coverage immediately. Continue only if the core fact remains unresolved and
    an executable route exists.

Corrections remain in the same interaction chain. The runtime never changes provider,
model, protocol, or thinking policy to hide a failure.

## Observation reducer

The reducer, not Gemini prose, creates canonical records.

Discovery can be created by:

- `reverse_image_search`;
- `text_search`.

Evidence can be created only from:

- fetched exact web spans;
- validated positioned OCR or focused image-region observations tied to an open gap;
- successful reference comparisons.

General consistency/anomaly output is diagnostic. It cannot create verdict Evidence.
Search snippets remain Discovery. Fetched spans and visual comparisons become
provenance-preserving Evidence, but their relation to the active proposition is not
finalized by query-relative extractor labels.

A Finding proposal is accepted only if:

- its task exists;
- every fact belongs to that task;
- every Evidence ID exists and belongs to the same task;
- its stance agrees with owned Evidence;
- source-family provenance is retained.

The reducer validates source quality, directness, risk, provenance, and visual
comparison structure. Evidence Decision interprets those qualified records relative
to the active proposition. Same-subject evidence from a different capture stays
neutral for exact image binding, and generic identity pages do not prove that an
entity is visible in the input.

If support and refute both qualify, Evidence Decision compares their semantics,
scope, binding, directness, source risk, and temporal alignment. The runtime validates
the cited IDs and provenance. A tie remains `conflicted` while the Agent seeks
discriminating evidence. If bounded search cannot break the tie, the final gap is
insufficient evidence, not “conflict means unverifiable.”

A task remains active after an `insufficient|conflicted` decision. It resolves only
after a terminal Evidence Decision creates task-owned Findings for the active fact.

## Evidence Decision prompt

Evidence Decision is not called for every new Evidence row. It runs when inspected
Evidence may decide the case, before Reflection, or before an unresolved terminal
outcome. Gemini evaluates the active proposition rather than the wording of the query
that found the source and returns:

```text
supported | refuted | conflicted | insufficient
```

It also decides whether binding is `text_sufficient`,
`same_capture_helpful`, or `same_capture_required`. Reliable text alone can refute
claims such as “monarch butterflies naturally occur in Antarctica.” Missing a
reference image does not keep that investigation alive. Same-capture is mandatory
only when the source assertion must be tied to this exact input image.

For an unresolved broad visual relation, Gemini may propose one narrower visible
slot. Deterministic validation requires pixel/OCR anchors, newly reviewed selected
Evidence, the same salient subject, and preservation of the original non-target
relation. Thus “orange-and-black butterflies in Antarctica” may become “monarch
butterflies in Antarctica,” but not “monarch butterflies in Mexico,” and never “photo
by Jane Example.”

Accepted terminal decisions create auditable task-owned Findings without changing
Evidence text, URL, span, hash, or function-call provenance. Coverage then stops
before Reflection or another search.

## Reflection prompt

At actions 4, 8, 12, 16, 20, and 24, Reflection receives the full bounded state. It may
return:

- task priority updates;
- up to three new grounded tasks;
- recommended next tasks;
- remaining gaps and readiness signal.

Deterministic validation rejects unknown IDs, duplicate tasks, unsupported task-state
mutation, and new tasks that do not serve an unresolved core evidence gap. Task status
and core ownership are reducer-owned; Reflection cannot mark a task resolved, blocked,
or exhausted, and cannot replace the `CoreVerdictFact`.

## Coverage and stop

Coverage is deterministic and runs after every accepted tool action. It consumes the
latest Evidence Decision for the active fact and compares its selected Evidence and
three bounded gaps with the previous audit.

```text
verdict_determined
information_saturated
hard_budget_exhausted
continue
```

`verdict_determined` stops immediately when qualified support or refutation closes all
required core gaps. `information_saturated` occurs when no executable core route
remains or when two consecutive accepted-action checkpoints make no qualified core
progress. Discovery-only progress, optional metadata, new IDs, and task churn do not
reset the counter.

## Judgment prompt

The runtime first compiles the smallest sufficient winning chain:

- exact verdict;
- exact fact/Finding/Evidence ID sets;
- fake mechanism where applicable;
- fact-specific unresolved gaps.

Gemini receives only those allowed objects and must reproduce the deterministic result.
Any mismatch is rejected. Judgment is explanatory synthesis, not a second decision
maker.

## Tool-internal model prompts

Several tools make their own Gemini Interactions calls:

| Tool/path | Purpose |
| --- | --- |
| `perceive_scene` | Literal image inventory |
| `crop_and_inspect` | Answer a focused crop question |
| `count_objects` | Standalone diagnostic visible-object count; not exposed in the Agent loop |
| `check_consistency` | Physical/visual consistency |
| `analyze_visual_anomalies` | Structured forensic anomalies |
| `compare_with_reference` | Query/reference comparison |
| reverse-image semantic branch | Semantic retrieval query |
| browse extraction | Select one immutable page passage and stance |

Tool-internal token/call metrics are removed from model-visible JSON and added to trace
accounting.

## Engineering failures versus unverifiable

`unverifiable` requires a valid investigation with at least one successful
investigation tool call and bounded unresolved facts.

The following are engineering errors instead:

- missing credentials/tools;
- image hash mismatch;
- perception/OCR failure;
- provider/transport/protocol failure;
- invalid structured output after correction;
- all investigation tools failing;
- invalid state transition;
- timeout.

They produce no classification prediction.

## Policy trajectory export

Actual model-visible `policy_input` and `policy_action` snapshots are captured on
Target Planning, ReAct, Evidence Decision, Reflection, and Judgment steps.

Exporter behavior:

- excludes evaluator-private fields;
- preserves runtime observation references and provider IDs;
- emits tokenizer-aligned action IDs and masks;
- masks invalid/fatal-boundary actions with zero;
- records runtime commit, release ID, and protocol versions;
- emits no fake Planning examples.

Dataset export excludes episodes that fail correctness, visual binding, basis
minimality, semantic-duplicate, post-determination, low-value-action, Finding
validity, or unresolved-conflict gates. Excluded episode IDs and reasons remain in a
separate metadata artifact.

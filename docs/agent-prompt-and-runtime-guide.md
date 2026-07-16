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
8. Run deterministic reduction and Coverage immediately after the accepted action.
9. Continue only if Coverage leaves the core fact unresolved and an executable route
   remains.

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

A Finding proposal is accepted only if:

- its task exists;
- every fact belongs to that task;
- every Evidence ID exists and belongs to the same task;
- its stance agrees with owned Evidence;
- source-family provenance is retained.

Direct official evidence can refute an exact event/place/identity slot. An exact
same-capture/near-duplicate hosted by an original official source can bind a full
scene proposition. A non-original or unknown image host additionally requires a
fetched direct source assertion. The same subject in a different capture is neutral,
and a generic official identity page does not prove visible presence in the input.

If support and refute both qualify, deterministic adjudication compares visual
binding, source originality, directness, independence, source risk, and temporal
alignment. A tie remains `conflicted` only while the Agent seeks discriminating
evidence. If bounded search cannot break the tie, the final gap is insufficient
evidence, not “conflict means unverifiable.”

A task may own Findings while remaining active. A core-owning task becomes resolved
only when its Findings materially close the owned core evidence gap; a weak Finding
must not remove the task from future scheduling.

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

Coverage is deterministic and runs after every accepted tool action. It compares the
one core fact, its winning qualified Evidence, and its three bounded gaps with the
previous audit.

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
Target Planning, Attribution Planning, ReAct, Reflection, and Judgment steps.

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

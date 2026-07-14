# Agent Prompt and Runtime Guide

This guide identifies which v3 decisions are model-authored and which are enforced by
deterministic code.

## Boundary map

| Stage | Decision owner | Input | Output |
| --- | --- | --- | --- |
| Release/case validation | Deterministic | v0.3 manifest and public row | Hash-verified `ImageOnlyRuntimeCase` |
| Scene perception | Gemini tool prompt | Image | Literal entities, boxes, scene, image type |
| OCR | EasyOCR | Image | Positioned text regions |
| Bootstrap | Deterministic | Case + perception/OCR | Brief, entities, facts, anchors, initial tasks |
| Initial reverse search | Deterministic schedule, real tool | Image | Discovery only |
| ReAct | Gemini native function calling | Compact current investigation | One tool action, then segment output |
| Observation reduction | Deterministic | Tool result + task/fact state | Discovery/Evidence/Finding/Failure updates |
| Reflection | Gemini structured output + deterministic validator | Global state every four actions | Bounded task/fact delta |
| Coverage | Deterministic | Facts, Findings, Evidence, tasks, budget | Stop state and fact coverage |
| Verdict basis | Deterministic | Final decisive-fact state | Allowed verdict and exact basis IDs |
| Judgment | Gemini structured output + deterministic validator | Allowed basis only | Matching `ImageOnlyJudgment` |
| Scoring/export | Deterministic, post-rollout | Trace + private references | Metrics, teacher score, policy examples |

## Perception

`perceive_scene` is a Gemini Interactions image request. It is instructed to report
literal visible content and normalized geometry, not infer evaluator gold.

`ocr_with_position` is EasyOCR and runs immediately afterward. The runtime merges both
results. OCR strings remain observations; bootstrap decides whether they become text
facts, relation facts, or retrieval anchors.

Failure of either required perception tool is an engineering error.

## Bootstrap

Bootstrap is not an LLM Planning stage. It deterministically:

- creates an inquiry-oriented immutable brief;
- converts visible entities and text into stable records;
- preserves pixel/OCR provenance;
- creates low-commitment candidate facts;
- chooses at most four initial tasks;
- activates up to three central, routed decisive facts.

The initial reverse-image search always runs once and records only Discovery.

Because bootstrap is deterministic, the trajectory exporter does not fabricate a
Planning target.

## ReAct prompt

The ReAct prompt tells Gemini to:

- choose one active `ResearchTask`;
- make exactly one native tool call;
- use the task ID as `question_id`;
- prefer unresolved priority-1 work and recommended tasks;
- treat snippets and reverse matches as Discovery;
- use fetched pages or successful visual observations for Evidence;
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
8. Continue until the four-action boundary or a valid segment output.

Corrections remain in the same interaction chain. The runtime never changes provider,
model, protocol, or thinking policy to hide a failure.

## Observation reducer

The reducer, not Gemini prose, creates canonical records.

Discovery can be created by:

- `reverse_image_search`;
- `text_search`;
- `crop_and_search`.

Evidence can be created only from:

- fetched exact web spans;
- successful image-region observations;
- successful reference comparisons.

A Finding proposal is accepted only if:

- its task exists;
- every fact belongs to that task;
- every Evidence ID exists and belongs to the same task;
- its stance agrees with owned Evidence;
- source-family provenance is retained.

Direct official or visual evidence can resolve a fact. Otherwise, two independent
qualified source families are required.

## Reflection prompt

At actions 4, 8, 12, 16, 20, and 24, Reflection receives the full bounded state. It may
return:

- task priority/status updates;
- up to three new grounded tasks;
- up to two decisive-fact proposals;
- recommended next tasks;
- remaining gaps and readiness signal.

Deterministic validation rejects unknown IDs, duplicate tasks, unresolved tasks marked
resolved without Findings, blocked tasks without Failures, and decisive facts with no
executable route.

## Coverage and stop

Coverage is deterministic and runs after each Reflection. It compares current fact
status and Evidence IDs with the previous audit.

```text
coverage_complete
information_saturated
hard_budget_exhausted
continue
```

`information_saturated` requires two consecutive low-gain Reflection intervals and no
unattempted priority-1 task. Discovery-only progress does not reset the counter.

## Judgment prompt

The runtime first compiles:

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
| `count_objects` | Visible object count |
| `check_consistency` | Physical/visual consistency |
| `analyze_visual_anomalies` | Structured forensic anomalies |
| `compare_with_reference` | Query/reference comparison |
| reverse/crop search query generation | Semantic retrieval query |
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
ReAct, Reflection, and Judgment steps.

Exporter behavior:

- excludes evaluator-private fields;
- preserves runtime observation references and provider IDs;
- emits tokenizer-aligned action IDs and masks;
- masks invalid/fatal-boundary actions with zero;
- records runtime commit, release ID, and protocol versions;
- emits no fake Planning examples.

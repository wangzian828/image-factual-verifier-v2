# Gemini Interaction and Prompt Sequence

**Runtime:** v4 discrepancy-first
**Date:** 2026-07-17
**Status:** living document; update with every prompt or stage-order change

The active short prompts are quoted below. Tool-specific prompt text remains in the
referenced Python constants. This document records exactly what Gemini sees, why a
call happens, its schema, and its Interaction lifetime.

## 1. End-to-end order

```text
1. Perceive Scene                         independent multimodal call
2. Image Account Planning                standalone, controlled image view
3. Investigation ReAct                   fresh tool-roundtrip root
4. Tool execution                        deterministic runtime
5. Function result                       returned inside the same short roundtrip
6. Discrepancy Decision                  standalone sparse checkpoint
7. Continue from a Search Hypothesis     repeat 3-6 while bounded
8. Final Judgment                        standalone request
```

Auxiliary Gemini calls used inside tools do not replace or fork semantic state.
They return observations to canonical state as tool results. No hidden Interaction
history crosses an action or stage boundary.

## 2. Ordered main chain

### 1. Perceive Scene

- Source: `src/tools/perceive_scene.py::PERCEIVE_SCENE_PROMPT`
- Interaction: independent tool-internal call
- Sees original image: yes
- Input: original image only
- Purpose:
  - literal scene account;
  - visible entities and normalized regions;
  - image type;
  - no web search or verdict.
- Output: `PerceptionReport`
- Next: deterministic OCR and bootstrap facts.

Core instruction:

> Describe only visible content and return literal entities with normalized
> bounding boxes. Do not perform fact checking.

### 2. Image Account Planning

- v4 source: `src/orchestrator/image_only_prompts.py`
- Replaces: `TARGET_PLANNING_SYSTEM_PROMPT`
- Interaction: `standalone_request`
- Sees original image: yes, controlled view supplied explicitly
- Thinking: Gemini keeps its configured semantic-stage level. Qwen3.5 uses a
  1,024-token Planning reasoning wall; reasoning is archived as excluded diagnostic
  material and never enters later model context or Evidence.
- Input:
  - a compact, deduplicated projection of the PerceptionReport;
  - positioned OCR observations;
  - all non-mechanical pixel/OCR VisualFact anchors;
  - deduplicated retrieval clues.
  - public case/media identity only; final Judgment fields such as
    `required_output` and `stop_policy` are not included in Planning.
  - deterministic bootstrap ResearchTasks are retained in canonical state but are
    not model-visible because Planning replaces them with semantic hypotheses.
- Purpose:
  - emit one to three high-salience ImageClaims;
  - separately emit at least one bounded SearchHypothesis without Claim-key binding;
  - plan beyond the image's vocabulary to establish underlying facts independently;
  - use model knowledge only to propose unverified retrieval leads;
  - avoid selecting a verdict owner;
  - keep external identities, dates, sources, creators, and platforms tentative.
- Output: `ImageAccountPlanningOutput`
- Next: deterministic Claim, hypothesis, and ResearchTask creation. Initial route
  attachment to the image account is bookkeeping, not a semantic conclusion.

Exact instruction:

> You are the Image Account Planning root. Plan an open fact-check of the real-world
> account communicated by the image. Return one high-salience ImageClaim preserving
> the central subject, event, and relation. Add at most two more only for independent
> verdict-changing assertions; do not inventory visible details.
> SearchHypotheses ask what actually happened, not merely whether the image's proposed
> value or an identical image can be found. Image clues do not limit the search. Prior
> knowledge supplies unverified leads; only tool Evidence establishes facts.
> Hypotheses do not own the verdict.
>
> Output fields: account_summary;
> image_claims[{claim_key, statement, kind, predicate, anchor_fact_ids, salience}];
> search_hypotheses[{hypothesis_key, statement, queries, expected_information,
> suggested_tools, priority}].

For the local Qwen Chat Completions profile, the same canonical output model is
used. The wire schema omits only string `minLength`, `maxLength`, `pattern`, and
`format`, which LMDeploy 0.13 declares unsupported and which can stall nested
guided decoding. Object structure, required fields, enums, list bounds, and numeric
bounds remain server-constrained; the complete Pydantic model is validated after
decoding. A bounded correction turn receives the actual validation error. The only
local syntax repair allowed is restoring a missing top-level opening `{` when the
remaining response is already one complete parseable object; no field or value is
inferred.

Qwen3.5 semantic stages use its official non-greedy sampling profile. Planning has
`thinking_token_budget=1024`; Discrepancy Decision and Judgment use 2,048, and
Reflection uses 1,536. The hard reasoning budget is separate from `max_tokens`.
The formal Qwen3.5 service does not enable MTP speculative decoding because current
vLLM releases have open reasoning-boundary and structured-output defects for that
combination.

### 3. Investigation ReAct

- Source: `src/orchestrator/image_only_prompts.py::DISCREPANCY_REACT_SYSTEM_PROMPT`
- Interaction: new `tool_roundtrip` for each action
- Sees original image: no; current recorded image understanding is explicit, and
  visual tools can inspect the saved image
- Input:
  - active ImageClaims;
  - open SearchHypotheses;
  - Discoveries, Evidence, Findings, failures;
  - attempted routes and remaining action budget;
  - deterministic tool constraints.
- Purpose:
  - choose exactly one permitted tool action;
  - investigate one unresolved claim through one hypothesis;
  - change the query angle when a different lead better tests the same claim;
  - inspect existing candidates before expanding search.
- Output: one native function call or bounded segment output
- Next: deterministic tool execution.

Exact instruction:

> Choose exactly one runtime-authorized tool action that most reduces uncertainty
> about an unresolved ImageClaim. Its attached SearchHypothesis supplies context and
> ownership, not a boundary on the investigation. Frame retrieval around what actually
> happened, not merely whether the image's proposed value or an identical image can be
> found. Prior knowledge may supply leads, but only tool Evidence establishes a fact.
> Do not change the ImageClaim.
>
> Inspect a promising page or reference image before repeating retrieval for that
> route. Search titles, snippets, and reverse-image matches are Discovery only.
> For page inspection, select one owned ImageClaim and state the passage sought.
> Qualified Evidence requires a fetched exact span or a successful visual
> observation with recorded provenance. Use only supplied observations, do not decide
> a verdict, and do not introduce external identities or metadata as new
> ImageClaims. The runtime owns IDs, claim/hypothesis ownership, route duplication,
> budgets, Evidence eligibility, state transitions, and stopping.

### 4. Tool Function Result

- Prompt: none; deterministic protocol message
- Interaction: returned only to the function-calling Interaction
- Sees original image: no hidden inherited image; visual tool results are explicit
- Input:
  - exact serialized tool result;
  - claim and hypothesis ownership IDs;
  - deterministic state delta;
  - remaining control state.
- Purpose:
  - preserve native Gemini function-call protocol;
  - expose only recorded observations.
- Output: next Gemini action or stage boundary.

### 5. Discrepancy Decision

- v4 source: `src/orchestrator/image_only_prompts.py`
- Replaces: `EVIDENCE_DECISION_SYSTEM_PROMPT`
- Interaction: `standalone_request`
- Sees original image: no; receives explicit image understanding, anchors, and any
  focused visual observations
- Trigger:
  - new qualified direct Evidence;
  - same-capture/reference comparison;
  - scheduled boundary with material new Evidence;
  - immediately before unresolved termination.
- Input:
  - original image context;
  - ImageClaims and SearchHypotheses;
  - exact new and prior Evidence;
  - visual anchors;
  - attempted routes and remaining budget.
- Purpose:
  - assess affected ImageClaims;
  - determine whether Evidence establishes a material discrepancy;
  - preserve recorded Evidence direction: support for supported, refute for
    refuted, and both for conflicted;
  - keep neutral or different-capture/no-edit comparisons non-terminal;
  - add or retire bounded SearchHypotheses;
  - propose `continue | fake | real`;
  - optionally request focused visual reinspection.
- Output: `DiscrepancyDecisionOutput`
- Next:
  - stop when deterministic verdict preconditions accept the proposal;
  - otherwise return to Investigation ReAct.

Exact instruction:

> You are the sparse multimodal Discrepancy Decision checkpoint. Compare the
> reviewed Evidence with the current image account. Update only affected
> Claim assessments; establish a MaterialDiscrepancy only when cited Evidence and
> visible anchors support it. You may retire or add a bounded, non-duplicate
> hypothesis or request one Evidence-motivated image reinspection. Omit Claims that
> have no reviewed owned Evidence. Use Evidence only in its recorded
> admissible_stances; neutral Evidence cannot support or refute a Claim. Task
> ownership permits review but does not establish semantic coverage; update only
> the Claims the Evidence actually addresses and use their allowed visual anchors.
> Treat qualified refutation of a high-salience Claim as decisive; unresolved other
> Claims do not weaken it. Propose fake for a decisive high-salience
> discrepancy, real when all high-salience claims are supported and meaningful
> routes are closed, otherwise continue. Use only supplied IDs and return the
> required JSON schema.

### 6. Final Judgment

- Source: `src/orchestrator/image_only_prompts.py::DISCREPANCY_JUDGMENT_SYSTEM_PROMPT`
- Interaction: `standalone_request`
- Sees original image: no; receives the compiled basis and recorded image understanding
- Input:
  - compiled verdict when Evidence already determined one;
  - accepted `DiscrepancyVerdictBasis`;
  - selected claims, discrepancies, Findings, and Evidence;
  - unresolved gaps.
- Purpose:
  - explain the already compiled verdict;
  - cite exactly the compiled basis;
  - add no new facts or searches.
- Output: `DiscrepancyJudgment`
- Next: trace persistence, scoring, and export.

Exact instruction:

> You are the constrained final synthesizer for discrepancy-first-v4. Give the final
> binary fact-check verdict, real or fake, from the supplied complete investigation
> basis. When compiled_verdict is non-empty, reproduce it. Otherwise weigh the
> recorded Evidence, image understanding, conflicts, failed routes and unresolved
> gaps and choose the better-supported binary conclusion. Copy every compiled-basis
> ID list and unresolved_gaps exactly. The assessment may summarize only supplied
> material; do not add historical facts or reopen search.

## 3. Auxiliary Gemini calls

These calls are independent of the semantic-stage Interactions. Their outputs are
observations, not state transitions.

| Order when invoked | Prompt source | Sees image | Role |
|---|---|---:|---|
| semantic reverse search | `src/tools/reverse_image_search.py::IMAGE_QUERY_PROMPT` | yes | derive retrieval query from the image |
| crop query | `src/tools/crop_and_search.py::CROP_QUERY_PROMPT` | crop | describe a local visual retrieval anchor |
| focused visual inspection | `src/tools/focused_visual_inspection.py::FOCUSED_VISUAL_INSPECTION_PROMPT` | original/crop | answer one Evidence-motivated visual question |
| reference comparison | `src/tools/compare_reference.py::COMPARE_PROMPT` | original + reference | record same-capture and material-difference observations |
| visual anomaly scan | `src/tools/visual_anomaly.py` prompts | original | diagnostic pixel observations only |
| webpage extraction | `src/integrations/browse/jina_reader.py::EXTRACT_PROMPT` | no | use `retrieval_goal` to select exact passages; judge stance only against `image_claim` |

The webpage extractor is an independent request with two trusted fields:

- `image_claim`: one model-selected, task-owned ImageClaim bound by the runtime;
  the extractor's `support | refute | unclear` stance is always relative to this
  atomic field;
- `retrieval_goal`: the passage sought by the current action. It ranks and selects
  passages but cannot determine stance or change the ImageClaim.

Stance follows whether the exact passage makes `image_claim` true or false. A page
reporting that someone made the claim does not support its truth; an explicit denial
is refuting Evidence. A passage that names the actual value of the disputed relation
remains useful even when it never repeats the image's proposed value; the extractor
still assigns direction only from the exact selected text.

The model selects only a Claim ID already owned by the scheduled ResearchTask; the
runtime injects its exact text and records the ID in provenance. One ReAct action
exposes one task-scoped route family, so URL, reference and Claim choices cannot be
combined across tasks. The webpage body is untrusted data. Passage selection reads
the cleaned full document up to the 60,000-character budget without a fixed
passage-count cap, and returned Evidence records retain both trusted fields for audit.

## 4. Prompt change checklist

Every prompt or interaction-order change must update:

1. the Python prompt constant;
2. its structured output schema;
3. this ordered document;
4. prompt-boundary tests;
5. one canonical trace assertion showing image visibility and parent Interaction;
6. training export stage labels when the semantic stage changes.

## 5. Migration ledger

| Stage | v3 executable | v4 target | Status |
|---|---|---|---|
| Perception | `PERCEIVE_SCENE_PROMPT` | retained | unchanged |
| Planning | `TargetPlanningOutput` | `ImageAccountPlanningOutput` | implemented and default |
| ReAct | core-fact task loop | claim/hypothesis task loop | implemented |
| Evidence checkpoint | `EvidenceDecisionOutput` | `DiscrepancyDecisionOutput` | implemented |
| Reflection | separate strategy call | absorbed into sparse discrepancy decision | removed from v4 path |
| Query replan | separate auxiliary calls | bounded hypothesis update | removed from v4 path |
| Judgment | core-fact basis | claim/discrepancy basis | implemented |

## 6. Current acceptance status

The local deterministic and mocked-Interactions gates pass for the complete v4
chain, including standalone-stage lifecycle checks, short native tool roundtrips,
claim/hypothesis ownership, sparse decisions, atomic reducers, terminal Coverage,
strict audit, and policy export. The 2026-07-20 Queen canary produced a correct,
evidence-determined `fake` with no protocol rejection; corrected strict audit and
heterogeneous live canaries remain pending.

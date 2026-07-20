# Gemini Interaction and Prompt Sequence

**Runtime:** v4 discrepancy-first
**Date:** 2026-07-17
**Status:** living document; update with every prompt or stage-order change

Executable prompt text remains in the referenced Python constants. This document
is the ordered index: what Gemini sees, why the call happens, what schema it must
return, and which interaction owns the next call.

## 1. End-to-end order

```text
1. Perceive Scene                         independent multimodal call
2. Image Account Planning                main Interaction root, original image
3. Investigation ReAct                   main Interaction continuation
4. Tool execution                        deterministic runtime
5. Function result                       returned to main Interaction
6. Discrepancy Decision                  sparse main Interaction continuation
7. Continue from a Search Hypothesis     repeat 3-6 while bounded
8. Final Judgment                        main Interaction continuation
```

Auxiliary Gemini calls used inside tools do not replace or fork semantic state.
They return observations to the main chain as tool results.

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
- Interaction: creates the stored main Interaction root
- Sees original image: yes, attached once
- Thinking: `high` by default; thought tokens are recorded, never Evidence
- Input:
  - PerceptionReport;
  - OCR observations;
  - pixel/OCR VisualFacts;
  - retrieval anchors;
  - bootstrap tasks.
- Purpose:
  - emit one to three high-salience ImageClaims;
  - emit bounded SearchHypotheses;
  - plan beyond the image's vocabulary to establish underlying facts independently;
  - use model knowledge only to propose unverified retrieval leads;
  - avoid selecting a verdict owner;
  - keep external identities, dates, sources, creators, and platforms tentative.
- Output: `ImageAccountPlanningOutput`
- Next: deterministic claim, hypothesis, and ResearchTask creation.

Core instruction:

> State the factual account the image asks us to check, then plan the investigation
> freely. The image supplies claims and clues, not the boundary of the search.
> Establish underlying facts independently; hypotheses remain unverified leads.

### 3. Investigation ReAct

- Source: `src/orchestrator/image_only_prompts.py::REACT_SYSTEM_PROMPT`
- Interaction: continues the main Interaction
- Sees original image: inherited through `previous_interaction_id`
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

Core instruction:

> Select one runtime-authorized action that most reduces uncertainty about an
> unresolved ImageClaim. Claim/hypothesis ownership records lineage but does not
> restrict the query angle. Model knowledge is a lead until tool Evidence verifies
> it.

### 4. Tool Function Result

- Prompt: none; deterministic protocol message
- Interaction: returned to the same main Interaction
- Sees original image: inherited
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
- Interaction: sparse continuation of the main Interaction
- Sees original image: inherited through `previous_interaction_id`
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
  - propose `continue | fake | real | unverifiable`;
  - optionally request focused visual reinspection.
- Output: `DiscrepancyDecisionOutput`
- Next:
  - stop when deterministic verdict preconditions accept the proposal;
  - otherwise return to Investigation ReAct.

Core instruction:

> Compare the supplied Evidence with the original image account. Identify a
> material factual discrepancy only when it is tied to visible anchors and cited
> Evidence. Failure to find a discrepancy is not proof that the image is real.

### 6. Final Judgment

- Source: `src/orchestrator/image_only_prompts.py::JUDGMENT_SYSTEM_PROMPT`
- Interaction: final continuation of the main Interaction
- Sees original image: inherited
- Input:
  - deterministically compiled verdict;
  - accepted VerdictBasis;
  - selected claims, discrepancies, Findings, and Evidence;
  - unresolved gaps.
- Purpose:
  - explain the already compiled verdict;
  - cite exactly the compiled basis;
  - add no new facts or searches.
- Output: `ImageOnlyJudgment`
- Next: trace persistence, scoring, and export.

Core instruction:

> Return the compiled verdict and explain only the selected basis. Do not reopen
> investigation or introduce uncited factual claims.

## 3. Auxiliary Gemini calls

These calls are independent of the stored main Interaction. Their outputs are
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

- `image_claim`: the positive account communicated by the image; the extractor's
  `support | refute | unclear` stance is always relative to this field;
- `retrieval_goal`: the fact or relation sought on the selected page; it ranks and
  selects passages but cannot determine stance.

The runtime binds both fields from the owning ResearchTask. They are not writable
tool arguments exposed to the investigation model. The webpage body is untrusted
data. Passage selection reads the cleaned full document up to the 60,000-character
budget without a fixed passage-count cap, and the returned Evidence records retain
both trusted fields for audit.

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
chain, including original-image root inheritance, claim/hypothesis ownership,
sparse decision checkpoints, atomic reducers, terminal Coverage, strict audit, and
policy export. The frozen four-trace replay and real Gemini canary gates remain
pending; unit tests are not production acceptance.

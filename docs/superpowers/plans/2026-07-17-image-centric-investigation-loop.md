# Image-Centric Investigation Loop

**Date:** 2026-07-17

**Status:** persistent-image main chain implemented; bounded strategy checkpoint in progress

## 1. Problem

The current runtime looks at the original image during initial perception, then
compresses it into text records. Target Planning, ReAct, Evidence Decision,
Reflection, Query Replan, and Judgment normally receive only those records.

This creates a structural failure mode:

```text
rich image meaning
  -> lossy scene summary
  -> one frozen core proposition
  -> competent search for the wrong proposition
```

Later stages cannot recover an omitted event, relation, location cue, or visual
contradiction because they neither see the image nor own a general target revision.
Adding sample-specific prompt rules or stop thresholds cannot repair that boundary.

## 2. Goal

Make the original image an immutable first-class observation throughout the
investigation:

```text
image + OCR
  -> multimodal target hypothesis
  -> multimodal tool choice
  -> web/reference observation
  -> sparse multimodal investigation checkpoint
       keep or revise target
       assess evidence
       choose the remaining gap and next route
       propose stop
  -> validated continuation or final judgment
```

Gemini owns semantic interpretation. Deterministic code owns IDs, provenance,
tool execution, duplicate prevention, budgets, schema validity, and the rule that
conclusions may cite only real observations.

## 3. Runtime changes

### 3.1 Image visibility

The original image is attached once to Target Planning, which creates the stored
Gemini main investigation chain. ReAct, Evidence Decision, Reflection, and Judgment
inherit it through `previous_interaction_id`; they do not upload it again.

Query Concept Extraction, Query Replan, OCR, webpage extraction, and tool-internal
visual calls remain independent. Tool-internal visual calls are still used for
focused crops, semantic image retrieval, and reference comparison.

When a one-action ReAct segment ends after a function call, the pending
`function_result` is submitted with the next main-chain `user_input`. This preserves
the native tool protocol across deterministic state boundaries.

### 3.2 Replace the frozen target with a revision history

Target Planning creates an initial hypothesis, not an irreversible evaluation
target. A later checkpoint may keep it or create a new revision.

Every accepted revision must:

- preserve a reference to the prior target;
- cite existing image/OCR anchor IDs;
- cite the new Evidence that motivated the revision;
- remain a claim about what the image visibly communicates or depicts;
- retire the prior target and supersede its open tasks;
- preserve the complete revision history in the trace.

Code validates references and lineage. It must not implement person-, domain-,
sample-, keyword-, or slot-specific semantic rules.

### 3.3 One multimodal investigation checkpoint

Replace the overlapping semantic duties of Evidence Decision, Reflection, and
Query Replan with one sparse checkpoint. It receives:

- the original image;
- current and prior target revisions;
- pixel/OCR observations;
- exact qualified Evidence;
- recent Discoveries and failures;
- attempted routes and remaining budget.

It returns one structured decision:

- evidence assessment: `supported | refuted | conflicted | insufficient`;
- selected Evidence IDs;
- target action: `keep | revise`;
- optional grounded target revision;
- bounded task/query delta when continuing;
- remaining gap;
- `ready_to_stop`;
- rationale.

The checkpoint runs:

- after a material Evidence batch;
- at the scheduled four-action boundary if no material checkpoint ran;
- before any unresolved terminal outcome;
- before accepting a final verdict.

It is not called after every search result.

### 3.4 Stop ownership

The checkpoint owns the semantic proposal to stop. Deterministic code accepts it
only when:

- `supported` or `refuted` cites existing qualified Evidence and no requested
  target revision remains unapplied; or
- bounded search is genuinely saturated/exhausted and the assessment remains
  `insufficient` or `conflicted`.

The runtime must not continue after an accepted terminal checkpoint. It also must
not stop merely because a numeric score, source class, missing same-capture image,
or fixed prompt heuristic says so.

### 3.5 Bounded continue / replan / stop control

The existing low-gain and route-exhaustion signals are inputs to semantic strategy,
not verdicts by themselves. At an interval boundary, or once immediately before an
unresolved terminal outcome, Reflection chooses:

- `continue`: name the concrete remaining route and expected information;
- `replan`: replace the current search direction with one genuinely different
  query and state what decisive information it should recover;
- `stop_unresolved`: declare that bounded search has no worthwhile new direction.

Termination remains provable:

- total accepted tool actions never exceed 24;
- each task has one immutable semantic replan allowance shared by evidence-led and
  stagnation-led replanning;
- at most one non-interval saturation Reflection may run per case;
- an accepted replan does not reset actions, attempts, route history, Evidence, or
  low-gain history;
- the replacement query must pass runtime novelty validation;
- stale candidates from the replaced direction are abandoned;
- a rejected or unavailable final replan falls through to
  `information_saturated`.

### 3.6 Keep reliable deterministic boundaries

Retain:

- release and image-hash validation;
- Discovery/Evidence/Finding/Failure separation;
- exact webpage-span recovery;
- evidence provenance and source-family metadata;
- native tool schema validation;
- duplicate-route and action-budget enforcement;
- engineering-error separation from `unverifiable`;
- strict trace auditing.

Remove or retire:

- one-time slot-specific core refinement;
- no-image Planning/Reflection/Judgment;
- a separate query-concept/query-replan policy path;
- deterministic semantic target-preservation heuristics;
- post-determination Reflection or search.

## 4. Experiment protocol

### H1: image visibility prevents target-information loss

Prediction: attaching the original image at semantic and action-selection
boundaries preserves event/relation qualifiers without sample-specific rules.

Implemented first change:

- no changes to search providers, Evidence provenance, source policy, or action
  budget;
- attach the original image once at Planning and keep the main policy chain stateful;
- keep Query Replan and other auxiliary interactions independent;
- compare target statements and route choices against the current baseline.

The ablation is diagnostic, not the final architecture. If information still
cannot be repaired because target ownership is frozen, proceed to H2.

### H2: a unified checkpoint repairs targets without search expansion

Prediction: one image-grounded checkpoint can revise a wrong or underspecified
target and close the resulting evidence chain with fewer low-value actions than
the current separate Evidence Decision, Reflection, and Query Replan stages.

## 5. Acceptance

Deterministic:

- all existing provenance, release, tool, and engineering-failure tests pass;
- policy snapshots prove one image-bearing Planning root and a continuous parent-ID
  chain across later main policy stages;
- policy snapshots replace image base64 with a `runtime_image` reference;
- every target revision cites valid image anchors and Evidence IDs;
- no duplicate, unknown, or post-stop tool route is accepted;
- no sample-specific string or domain rule is added.

Real trajectories:

- use heterogeneous cases from the v4 20-case development set;
- include at least one ordinary real photo, one refuted compositional/claim case,
  and one screenshot or source-record case;
- include the Queen bus case as a known target-loss diagnostic, not as the sole
  optimization target;
- target statements preserve or correctly revise salient image relations;
- correct verdict on at least three selected cases;
- no actions after decisive evidence;
- trace audit passes;
- model-call count is explained by visible policy/tool/checkpoint boundaries.

Failure of one case triggers architectural error analysis. It does not authorize
another case-specific prompt clause.

## 6. Delivery order

1. Record the baseline target/image-visibility map.
2. Implement and run H1.
3. Review heterogeneous real traces.
4. Implement H2 only after confirming the remaining failure is target ownership,
   not provider/tool breakage.
5. Remove superseded stages and validators.
6. Update architecture, prompt/runtime, operations, and trajectory documents.
7. Run deterministic suite and real acceptance set.

## 7. Dynamic adjustment: visually anchored external hypotheses

The Pillars of Creation canary exposed an over-constrained Planning boundary. The
policy correctly recognized the scene and proposed JWST, NIRCam, and release-year
hypotheses, but deterministic validation required every named value to already
appear in pixels or OCR and then required a two-entity relation for the fallback.

Planning now distinguishes visual grounding from prior confirmation:

- at least one salient subject or scene must be anchored in image/OCR state;
- an external identity, instrument, place, event, date, or source may be proposed
  as a tentative hypothesis and must later be supported or refuted by Evidence;
- source-record hypotheses without visible text may own the core only when they
  name a specific candidate rather than request generic provenance discovery;
- a one-subject scene relation is represented as `identified_as`, while
  `depicts_relation` remains reserved for two visible entities;
- disconnected hypotheses and queries without any visible anchor still fail.

This removes lexical model-memory policing from Planning without weakening
Evidence ownership, source binding, action budgets, or final verdict validation.

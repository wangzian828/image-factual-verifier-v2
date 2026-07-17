# Image-Centric Investigation Loop

**Date:** 2026-07-17

**Status:** active implementation plan

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

The original image must be attached to:

- Target Planning;
- every ReAct action selection;
- every investigation checkpoint;
- final Judgment.

Tool-internal visual calls remain available for focused crops, semantic image
retrieval, and reference comparison. They complement rather than replace the
shared original-image context.

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

### 3.5 Keep reliable deterministic boundaries

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

First ablation:

- no changes to search providers, Evidence provenance, source policy, or action
  budget;
- make Planning, ReAct, existing semantic checkpoints, and Judgment multimodal;
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
- policy snapshots prove image attachment at all required stages;
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


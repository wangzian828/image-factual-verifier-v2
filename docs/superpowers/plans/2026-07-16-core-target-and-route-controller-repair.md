# Core Target and Route Controller Repair

**Date:** 2026-07-16

**Status:** implemented locally; gpu-13 Monarch acceptance pending

## Why

The successful retry on the clean v3 checkout established a valid runtime trace:

```text
initial Planning correction
  -> correct real-world ecology core target
  -> text search discovers a relevant migration page
  -> discovery incorrectly triggers Attribution
  -> the ecology task's visit goal becomes image attribution
  -> relevant migration text is discarded as "does not identify this image"
```

The retry therefore ended `unverifiable` after 5 actions and strict audit rejected
one avoidable protocol rejection. The earlier 74-call Monarch trace came from a
dirty legacy server checkout and is not used as performance evidence.

## Design goal

Keep one stable core fact and preserve its evidence goal from retrieval through
page extraction. Discovery is a lead; it must not convert a world-fact verification
task into image attribution.

```text
core world relation
  -> search lead
  -> visit lead using the original relation-specific goal
  -> Evidence
  -> sparse semantic Evidence Decision
  -> Finding
  -> Coverage
```

The retry's eventual core was already suitable:

```text
The input image depicts a real-world ecological event where monarch butterflies
migrated to Antarctica and clustered on a pine tree near penguins.
```

Evidence that normal monarch migration is confined to North America/Mexico can
directly refute this core. Pixel anomaly checks remain diagnostic and cannot
independently own a verdict.

## Scope

### 1. Preserve the core evidence goal across Discovery

`_image_only_task_evidence_goals` currently changes any task with a Discovery into:

```text
Does this candidate public source identify or directly describe the same input image?
```

That goal is appropriate only for source-binding/provenance tasks. For an active
world-fact core, retain `task.question` after search results arrive. The page reader
must receive the concrete ecological/event/place relation it is supposed to support
or refute.

### 2. Remove runtime Attribution target expansion

The runtime Attribution stage is removed. A generic SERP/Lens Discovery cannot create
creator/title/platform facts or mutate the active world-fact goal.

Qualified Evidence is reviewed at a sparse Evidence Decision checkpoint. Gemini may:

- support, refute, conflict, or leave the active proposition insufficient;
- decide whether text is sufficient or same-capture binding is required;
- once, narrow an unknown visible subject/place/event slot while preserving the
  original relation.

The runtime validates Evidence IDs, task/fact ownership, pixel/OCR anchors, one-time
budget, and relation preservation. It does not use a score threshold or prompt rule to
decide truth.

### 3. Dynamically expose only executable tools

Before every ReAct segment, derive the allowed tool names from the reducer's existing
`remaining_material_routes` inventory:

```text
pending candidate page -> visit only
untried text-search slot -> text_search available
untried Lens/semantic branch -> reverse_image_search available
```

If no tool owns a remaining route, Coverage receives `information_saturated` before
StageRunner requests a function. This prevents the model from seeing `text_search`
while a page visit is mandatory, and prevents duplicate/exhausted calls from becoming
protocol-correction loops.

The later full `route_id` action catalog remains optional. Start with dynamic tool
exposure because it uses the reducer inventory already implemented and directly
addresses the observed trace.

Semantic duplicate detection remains as a guardrail, not the normal control path.

### 4. Stop before Reflection or another search

After a material Evidence Decision, Coverage runs immediately. A supported/refuted
core with all required gaps closed sets `verdict_determined`; the loop cannot run
Reflection or another tool action afterward. Before an unresolved terminal outcome,
all pending core Evidence receives one mandatory semantic review.

## Acceptance tests

1. The Monarch ecology task keeps its migration/coexistence goal when visiting the
   Florida Museum/NOAA-style search lead, and the page can create refuting Evidence.
2. A Discovery alone does not invoke semantic adjudication or create
   creator/title/product tasks.
3. When a candidate page is uninspected, only `visit` is exposed for that task; no
   strict-audit protocol rejection occurs.
4. When every route is exhausted or duplicate, the segment terminates
   deterministically as bounded unresolved; it never raises
   `protocol_correction_budget_exhausted`.
5. A real Monarch retry, then Berlin Wall supported and Queen bus refuted, must
   pass strict trace audit before any broader run.
6. Unknown orange-and-black butterflies may refine once to monarch butterflies while
   retaining Antarctica; a refinement that substitutes Mexico or photographer
   metadata is rejected.
7. Reliable ecology text can refute the Antarctica relation without a reference
   image, and the trace contains no action/Reflection after that decision.

## Non-goals

- No modification to the data-pipeline repository or frozen v4 release.
- No change to the existing real/fake mapping for a supported/refuted ordinary
  world-fact core.
- No unrestricted target refresh or target expansion.
- No replacement of Gemini while it remains the teacher runtime.

## Local implementation result

Implemented:

- removed the active Attribution stage and deterministic fallback target invention;
- required Target Planning to establish exactly one atomic external-world or
  source-record core, with invalid Planning failing explicitly instead of promoting
  bootstrap `appears_to_depict` prose;
- normalized only ungrounded parenthetical scientific binomials before the normal
  Planning grounding checks, without allowing other remembered metadata;
- made Planning tools first-hop suggestions while task-owned Discoveries
  deterministically authorize and URL-bind their concrete `visit` or
  `compare_with_reference` follow-up;
- replaced source-class route gating with one bounded model-selected inspection per
  retrieval batch; source class remains soft provenance/risk metadata;
- added `image_only_evidence_decision` and trajectory export;
- made semantic checkpoints sparse rather than per-Evidence;
- allowed text-only closure when semantically sufficient;
- made same-capture conditional on the proposition;
- added one relation-preserving visual-slot refinement;
- made Coverage consume the latest semantic decision;
- added strict trace auditing for Evidence ownership, refinement grounding/budget,
  and post-determination actions.

Local deterministic validation is complete; real acceptance remains the Monarch,
Berlin Wall, and Queen bus sequence on the clean gpu-13 worktree.

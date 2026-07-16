# Core Target and Route Controller Repair

**Date:** 2026-07-16

**Status:** planned; waiting for the clean Monarch retry before implementation

## Why

The first valid `Monarch` run on the clean v3 checkout reproduced two controller
problems before a transient Gemini HTTP 500 stopped the case:

```text
visible ecological contradiction
  -> model proposes "fictional/composite" target
  -> target validator rejects its direction
  -> runtime falls back to broad scene provenance
  -> Lens discovery triggers optional attribution work
```

The earlier 74-call Monarch trace came from a dirty legacy server checkout and is
not used as performance evidence. The clean run still confirms the target-direction
failure: its two initial Planning revisions rejected the model's ecological target
instead of retaining a focused factual hypothesis.

## Design goal

Keep one stable core fact, but stop encoding the benchmark verdict direction through
English phrasing such as “positive proposition.” A core target may naturally be:

```text
"The pictured ecological juxtaposition is fictional rather than a documented event."
```

or:

```text
"Monarch butterflies naturally overwinter in the Antarctic landscape shown."
```

Both are valid hypotheses. What matters is explicit, deterministic verdict mapping:

```text
evidence supports core target -> real or fake, as declared by target orientation
evidence refutes core target  -> the opposite class
insufficient/conflicted evidence -> unverifiable
```

Pixel anomaly checks remain diagnostic. They cannot independently own a verdict.

## Scope

### 1. Add explicit core-target orientation

Add a `support_verdict` field to target proposals and persisted VisualFacts:

```text
support_verdict = real | fake
```

The runtime, not the final Judgment prompt, maps a resolved fact status through this
field. Existing facts default to `real`, preserving ordinary source/event/location
targets.

Validation continues to require:

- image/OCR grounding;
- one atomic factual relation;
- no remembered names, dates, or source metadata;
- an external evidence route for a `fake`-oriented target;
- no direct verdict from VLM anomaly output alone.

Remove only the linguistic rule that a target must be phrased as a positive
authenticity/world claim. Do not add a list of wording exceptions.

### 2. Keep attribution optional by construction

Attribution planning may run only when it can close a current core gap:

```text
source/image binding is required
or
a discovered same-capture/source record can directly adjudicate the core
```

A generic Lens hit, title, artist, platform, product page, or creator detail cannot
create a verification task merely because it shares the image subject.

### 3. Make executable routes a runtime-owned inventory

Before every ReAct segment, the reducer computes a small `ActionOption` inventory:

```text
route_id
task_id
tool_name
fixed candidate URL/reference, when applicable
open core gap served
```

The model chooses among this inventory. It may supply a bounded query only for an
explicit `text_search` option. The reducer rejects routes outside the inventory
without counting a fake successful action.

If the inventory is empty, Coverage receives `information_saturated`; StageRunner
must return the deterministic segment boundary rather than spending protocol
corrections forcing an impossible tool call.

This replaces the current failure mode:

```text
force one tool call
  + duplicate / exhausted / disallowed tool
  -> protocol correction
  -> another invalid call
  -> engineering error
```

It does not remove semantic duplicate detection. It makes duplicate prevention a
precondition for exposed actions.

## Acceptance tests

1. The actual Monarch Planning proposal “fictional or surreal ecological
   juxtaposition” is accepted as an externally-verifiable, `fake`-oriented core
   target; an unsupported VLM anomaly alone still cannot finish the case.
2. A direct ecology/source Evidence that supports a fake-oriented core compiles
   `fake`; refuting it compiles `real`.
3. Existing real-oriented core facts still compile supported -> `real` and
   refuted -> `fake`.
4. A Lens discovery for an ecology core does not invoke attribution planning or
   create creator/title/product tasks.
5. When every route is exhausted or duplicate, the segment terminates
   deterministically as bounded unresolved; it never raises
   `protocol_correction_budget_exhausted`.
6. A real Monarch retry, then Berlin Wall supported and Queen bus refuted, must
   pass strict trace audit before any broader run.

## Non-goals

- No modification to the data-pipeline repository or frozen v4 release.
- No LLM judge for exact evidence-chain recovery.
- No unrestricted target refresh or target expansion.
- No replacement of Gemini while it remains the teacher runtime.

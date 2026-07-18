# Visual Fact Discrepancy Agent v4 Architecture

## Supported runtime

The default input is the immutable data-pipeline v0.3 `image_only` contract with
`decision_policy_version=reinspect-v2`:

```text
case_id + image_path + image_sha256
```

The release field identifies the data protocol only. `run_eval` does not pass it
through as an Agent selector. The v4 Agent and every canonical runtime trace use
`decision_policy_version=discrepancy-first-v4`; run manifests record the release
policy under `benchmark` and the effective Agent policy under `agent`.

The tagged v3 implementation is frozen at `runtime-v3-final-20260717`. Legacy
schemas and reducers remain temporarily for deterministic historical replay; they
are not the v4 semantic path.

## End-to-end control flow

```text
hash-verified original image
  -> Gemini literal scene perception
  -> positioned OCR
  -> deterministic visual facts and retrieval anchors
  -> Image Account Planning (main Interaction root; original image attached once)
       1-3 ImageClaims
       bounded SearchHypotheses
  -> claim/hypothesis-owned ReAct action
  -> deterministic Discovery / Evidence / Failure reduction
  -> sparse multimodal Discrepancy Decision
       ClaimAssessment
       optional MaterialDiscrepancy
       bounded hypothesis updates
       optional one focused visual reinspection
       continue | fake | real | unverifiable proposal
  -> deterministic discrepancy Coverage and minimal verdict basis
  -> constrained v4 Judgment
```

All later main-chain stages inherit the original image through
`previous_interaction_id`. Tool-internal model calls are independent observations and
cannot mutate semantic state.

## State ownership

### ImageClaim

A positive factual statement communicated by visible pixels or reliable embedded
text. It cites pixel/OCR `VisualFact` anchors, has `high|medium` salience, and is
assessed as `open|supported|refuted|conflicted|unresolved`.

### SearchHypothesis

A bounded retrieval direction attached to one or more ImageClaims. External names,
places, dates, events, sources, creators, platforms, instruments, and species
identities remain hypotheses until qualified Evidence supports a semantic decision.
A hypothesis never owns a verdict.

### MaterialDiscrepancy

An image-aware conclusion proposed by Gemini and accepted only when it cites affected
ImageClaims, their visible anchors, and task-owned qualified Evidence. It is
`decisive|supporting` and `established|conflicted`.

## Deterministic boundaries

Code owns:

- input/hash validation and evaluator-private isolation;
- stable IDs, references, task ownership, and atomic state replacement;
- native tool schemas and one-call-per-action execution;
- Discovery/Evidence separation and successful-call provenance;
- duplicate-route, action, hypothesis, decision, and reinspection budgets;
- sparse checkpoint scheduling, Coverage, verdict preconditions, and strict audit.

Gemini owns:

- the image account and salient ImageClaims;
- retrieval hypotheses within deterministic bounds;
- Evidence-to-claim semantic assessment;
- whether a visually anchored material discrepancy exists;
- the terminal proposal that deterministic Coverage may accept or reject.

Search titles, snippets, and reverse-image matches are Discovery only. Web Evidence
requires a fetched exact span, offsets, canonical source, artifact hash, retrieval
time, directness, stance, and successful function-call provenance. Visual Evidence
requires a successful focused observation or reference comparison with image hashes
and provenance. General anomaly opinions are diagnostic only.

## Verdict rules

- `fake`: at least one established decisive MaterialDiscrepancy affects a
  high-salience ImageClaim and cites qualified Evidence plus visible anchors.
- `real`: every high-salience ImageClaim is supported, no decisive discrepancy
  remains, meaningful high-salience routes are closed, and Gemini proposes real.
- `unverifiable`: a high-salience ImageClaim remains insufficient or conflicted, no
  decisive discrepancy is established, its meaningful routes are closed, and Gemini
  proposes unverifiable.

Failure to find a discrepancy is not evidence of reality. Provider, protocol,
runtime, required-tool, and all-tools-failed conditions are engineering errors and
end before Judgment.

## Stop and budgets

The hard action cap is 24. One accepted native tool call is one action. The v4 loop
stops immediately after terminal Coverage, before any further search. One focused
visual reinspection may be requested by Discrepancy Decision. New hypotheses are
bounded globally and per decision; semantically duplicate routes are rejected.

## Audit and training

The strict auditor verifies claim/hypothesis/task ownership, successful Evidence
calls, ClaimAssessment Evidence scope and direction, reference-comparison stance
coherence, discrepancy-to-claim anchors, qualified refuting discrepancy Evidence,
the complete `VisualFact -> Finding -> Evidence -> successful call` verdict chain,
terminal Coverage, basis/Judgment equality, interaction ancestry, action-count parity,
and absence of post-verdict actions.

`ifv-policy-v2` exports Image Account Planning, v4 ReAct, Discrepancy Decision, and
v4 Judgment. Training eligibility requires classification correctness, complete
directionally consistent Evidence chains, discrepancy alignment, stop quality, and
no protocol rejection or legacy core ownership. The pure policy exporter repeats
these gates and rejects a trace even if upstream score metadata is wrong.

## Acceptance status

Local deterministic reducers, mocked native Interactions, the complete default
workflow, strict audit, scoring, and export tests pass. Three frozen historical
fixtures retain admissible conclusions; the prior Andreea fixture is now a required
safety rejection because its different-capture Evidence was neutral. The first real
canary exposed engineering ownership gaps and the second exposed the neutral-Evidence
semantic upgrade now covered by regression gates. Production acceptance still
requires one new real Gemini canary with manual trace inspection, followed by three
to four heterogeneous canaries. Unit tests alone do not mark v4 complete.

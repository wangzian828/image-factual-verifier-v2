# Discrepancy-First Runtime v4

**Date:** 2026-07-17

**Status:** active implementation plan

## 1. Objective

Runtime v4 investigates an image account rather than freezing one speculative
claim before search:

```text
original image
  -> image account
  -> bounded retrieval hypotheses
  -> web and visual Evidence
  -> image-aware discrepancy decision
  -> fake | real | unverifiable
```

The runtime looks for a material factual discrepancy tied to visible anchors. A
discrepancy may be pixel-level, relational, textual, contextual, geographic,
temporal, ecological, or source-related. Synthetic pixels are not themselves a
fake verdict when the material image account is factually supported.

## 2. State contract

### ImageClaim

One positive factual statement communicated by the pixels or embedded text.

- `claim_id`
- `statement`
- `anchor_fact_ids`
- `salience: high | medium`
- `status: open | supported | refuted | conflicted | unresolved`

Planning emits one to three high-value claims. It does not select a verdict owner.

### SearchHypothesis

A bounded retrieval direction attached to one or more ImageClaims.

- `hypothesis_id`
- `claim_ids`
- `statement`
- `queries`
- `expected_information`
- `status: open | active | exhausted | retired`

External identities, dates, photographers, platforms, instruments, events, and
source candidates are allowed. A SearchHypothesis cannot directly own a verdict.

### MaterialDiscrepancy

An image-aware semantic conclusion grounded in qualified Evidence.

- `discrepancy_id`
- `statement`
- `affected_claim_ids`
- `visual_anchor_fact_ids`
- `evidence_ids`
- `materiality: decisive | supporting`
- `status: established | conflicted`

Only Gemini may propose discrepancy semantics. Deterministic code validates IDs,
Evidence ownership, budgets, lineage, and cardinality.

### ClaimAssessment

The latest Evidence-based state of one ImageClaim.

- `claim_id`
- `assessment: supported | refuted | conflicted | insufficient`
- `evidence_ids`
- `remaining_gap`

## 3. Multimodal decision ownership

The existing Gemini main Interaction keeps the original image visible from
Planning through investigation. A sparse `DiscrepancyDecision` runs:

- after qualified direct Evidence or same-capture comparison;
- at a scheduled action boundary when material new Evidence exists;
- before unresolved termination.

It receives the original image, ImageClaims, SearchHypotheses, exact Evidence,
visual anchors, attempted routes, and remaining budget. It may:

- assess claims;
- establish or reject a material discrepancy;
- add or retire bounded SearchHypotheses;
- request one focused visual reinspection;
- propose `continue | fake | real | unverifiable`.

It does not run after every search result.

## 4. Verdict rules

### fake

At least one decisive MaterialDiscrepancy:

- cites existing qualified Evidence;
- identifies affected ImageClaims;
- cites image/OCR anchor facts;
- is accepted by the image-aware decision checkpoint.

One decisive discrepancy is terminal.

### real

- every high-salience ImageClaim is supported;
- no established or unresolved decisive discrepancy remains;
- the image-aware checkpoint proposes real;
- bounded route state permits stopping.

Failure to find a discrepancy is never sufficient by itself.

### unverifiable

- at least one high-salience claim remains insufficient or conflicted;
- no decisive discrepancy is established;
- meaningful routes are saturated or exhausted;
- the image-aware checkpoint proposes unverifiable.

## 5. Deterministic responsibilities

Code owns:

- release and image-hash validation;
- IDs and object references;
- tool schemas and execution;
- Evidence provenance;
- duplicate-route prevention;
- action, hypothesis, reinspection, and decision budgets;
- state transitions;
- verdict preconditions;
- strict trace audit.

Code does not own:

- whether a date, photographer, platform, or source is semantically central;
- whether Evidence reveals a material discrepancy;
- whether a source hypothesis should change the image account;
- factual conclusions not present in recorded Evidence.

## 6. Migration

1. Add v4 state models and deterministic reducer tests.
2. Replace Target Planning with Image Account Planning.
3. Convert Planning hypotheses into claim-owned ResearchTasks.
4. Replace Evidence Decision with Discrepancy Decision.
5. Replace core-only Coverage and verdict compilation.
6. Migrate prompt rendering and route selection.
7. Migrate strict audit, process scoring, reference-chain scoring, and export.
8. Remove `core_verdict_fact_id`, one-core refinement, and hypothesis promotion.
9. Replay frozen Andreea, Queen, Monarch, and Pillars fixtures.
10. Run one real Gemini canary only after deterministic replay passes.

## 7. Acceptance

- no external hypothesis can own a verdict before Evidence;
- Andreea product insertion establishes a decisive discrepancy and fake;
- Queen event/transport contradiction establishes a decisive discrepancy and fake;
- Monarch ecological contradiction establishes a decisive discrepancy and fake;
- Pillars official identity support closes real without requiring pixel originality;
- no post-decision action;
- no duplicate or protocol-rejected route;
- strict audit passes;
- training export excludes any trajectory that lacks discrepancy/claim Evidence
  alignment.

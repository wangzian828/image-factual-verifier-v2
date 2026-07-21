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
  -> standalone Image Account Planning (controlled original-image view)
       1-3 ImageClaims
       independently planned, bounded SearchHypotheses
  -> case/hypothesis-owned ReAct action
  -> deterministic Discovery / Evidence / Failure reduction
  -> sparse multimodal Discrepancy Decision
       ClaimAssessment
       optional MaterialDiscrepancy
       bounded hypothesis updates
       optional one focused visual reinspection
       continue | fake | real proposal
  -> deterministic discrepancy Coverage and minimal verdict basis
  -> constrained v4 Judgment
```

Planning, every Discrepancy Decision, and Judgment are independent requests built
from an explicit, versioned workspace handoff. One ReAct action may use
`previous_interaction_id` only to complete its native
`function_call -> function_result -> output` protocol. No hidden Interaction history
crosses an action or stage boundary. Tool-internal model calls are independent
observations and cannot mutate semantic state.

The local Qwen Chat Completions transport has no provider Interaction ID, so the
runtime records equivalent request ancestry itself. Context manifests distinguish
`standalone_request`, `tool_roundtrip`, and `protocol_correction`; only a correction
may set `parent_request_id` to the rejected request. The strict auditor follows this
chain transitively and requires it to end in an accepted output before classifying
the rejection as recovered.

## State ownership

### ImageClaim

A positive real-world proposition communicated by visible pixels or reliable
embedded text. It records what the image asks the viewer to believe, not merely the
meta-fact that a caption, post, or advertisement contains the assertion. It cites
pixel/OCR `VisualFact` anchors, has `high|medium` salience, and is assessed as
`open|supported|refuted|conflicted|unresolved`. Planning produces exactly one high
Claim containing the complete central relation; up to two additional independent
Claims must be medium.

### SearchHypothesis

A bounded retrieval direction for the image account. Its Planning schema contains
no Claim key: the route may ask for the actual underlying value instead of repeating
the depicted value. After Planning, the reducer attaches the route to the current
account Claims only as broad bookkeeping for provenance, budgets, Evidence review,
and stopping. That attachment does not establish support or refutation. The
Discrepancy Decision must still select the affected ImageClaim and prove the
directional Finding/Evidence chain. A hypothesis never owns a verdict.
Planning must provide at least one route with an executable first hop; the runtime
does not constrain the route's factual angle. Any explicit planned web query must
include `text_search` in that Hypothesis so the query is executable rather than dead
context.

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
time, directness, stance, and successful function-call provenance. Page retrieval
uses an explicit two-field contract: `retrieval_goal` selects relevant passages,
while stance is judged only against one model-selected, task-owned `image_claim`
whose exact text the runtime binds from its Claim ID. Each ReAct action exposes one
task-scoped route family, preventing invalid cross-task URL/Claim combinations.
Both fields remain in the canonical tool result and Evidence ledger. Visual Evidence
requires a successful focused observation or reference comparison with image hashes
and provenance. General anomaly opinions are diagnostic only.

Archive recall is optional support for an otherwise executable task, not an
unbounded route family. It becomes available only after that task has produced
archived investigation material, is limited to two recall/read cycles per task, and
`read_evidence` is exposed only for IDs returned by the pending recall.

Final discrepancy Judgment is split into a model-owned output and a runtime-owned
canonical record. The model returns only `verdict`, `confidence`, and
`overall_assessment`; deterministic code injects the compiled Claim, discrepancy,
Finding, Evidence, and unresolved-gap IDs. This prevents a final synthesis call
from inventing or dropping basis identifiers.

## Verdict rules

- `fake`: at least one established decisive MaterialDiscrepancy affects a
  high-salience ImageClaim and cites qualified Evidence plus visible anchors.
  A qualified refutation of such a Claim cannot be downgraded because another
  Claim remains unresolved.
- `real`: every high-salience ImageClaim is supported, no decisive discrepancy
  remains, meaningful high-salience routes are closed, and Gemini proposes real.
- unresolved or conflicted internal state is retained in the verdict basis. After
  meaningful routes close or the 24-action cap is reached, bounded Judgment chooses
  the better-supported binary verdict and reports the unresolved gaps.
  Its compiled basis always includes every high-salience Claim plus unresolved
  Claims of any salience, together with their recorded anchors, Evidence, Findings,
  and an explicit gap explaining why the evidence-determined exit did not fire.

Failure to find a discrepancy is not evidence of reality. Provider, protocol,
runtime, required-tool, and all-tools-failed conditions are engineering errors and
end before Judgment.

## Stop and budgets

The hard action cap is 24. One accepted native tool call is one action. The v4 loop
stops immediately after terminal Coverage, before any further search. Normal stops
are `verdict_determined`, `meaningful_routes_exhausted`, and
`hard_budget_exhausted`; provider/protocol failures remain `engineering_error`.
The remaining-route inventory includes only tasks that still own at least one
`open`, `conflicted`, or `unresolved` ImageClaim. A stale active task attached only
to supported/refuted Claims cannot keep the investigation alive or trigger a
no-executable-task error.
No-gain streaks are diagnostic only. One focused visual reinspection may be
requested by Discrepancy Decision. New hypotheses are bounded globally and per
decision; semantically duplicate routes are rejected.

## Audit and training

The strict auditor verifies claim/hypothesis/task ownership, successful Evidence
calls, ClaimAssessment Evidence scope and direction, reference-comparison stance
coherence, discrepancy-to-claim anchors, qualified refuting discrepancy Evidence,
the complete `VisualFact -> Finding -> Evidence -> successful call` verdict chain,
terminal Coverage, basis/Judgment equality, interaction ancestry, action-count parity,
and absence of post-verdict actions. For Qwen, protocol-correction ancestry uses
durable context request IDs rather than invented provider Interaction IDs, including
multi-hop retries.

Local student profiles also own their serving endpoint instead of inheriting the
shared legacy `QWEN_LOCAL_BASE_URL`. `student-qwen3-vl-local` defaults to port 8899;
`student-qwen3.5-local` defaults to port 8901 and may be overridden only by its
profile-scoped `QWEN35_LOCAL_BASE_URL`; its model override is likewise isolated as
`QWEN35_LOCAL_MODEL`. The same resolved endpoint and model are passed to the policy
backend and every visual tool, preventing split-brain text/vision routing.

`ifv-policy-v2` exports Image Account Planning, v4 ReAct, Discrepancy Decision, and
v4 Judgment. Training eligibility requires classification correctness, complete
directionally consistent Evidence chains, discrepancy alignment, stop quality, and
no protocol rejection or legacy core ownership. The pure policy exporter repeats
these gates and rejects a trace even if upstream score metadata is wrong.

## Acceptance status

Local deterministic reducers, mocked native Interactions, the complete default
workflow, strict audit, scoring, and export tests pass. Three frozen historical
fixtures retain admissible conclusions; the prior Andreea fixture is now a required
safety rejection because its different-capture Evidence was neutral. The 2026-07-20
Queen canary produced an evidence-determined `fake`, complete
claim/discrepancy/Evidence alignment, immediate stopping, and no protocol rejection.
Its first acceptance command exposed nested runtime artifacts being mistaken for
canonical traces; trace discovery is now limited to direct `traces/*.json` files.
Production acceptance still requires the corrected strict audit plus three to four
heterogeneous canaries. Unit tests alone do not mark v4 complete.

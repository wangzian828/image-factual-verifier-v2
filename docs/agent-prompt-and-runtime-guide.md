# v4 Prompt and Runtime Guide

## Boundary map

| Stage | Semantic owner | Deterministic enforcement |
| --- | --- | --- |
| Case validation | none | exact three-field row, path boundary, SHA-256 |
| Perception/OCR | Gemini/tool provider | required-tool success and literal schemas |
| Bootstrap | none | stable visual facts and retrieval anchors |
| Image Account Planning | Gemini | exactly one central high Claim, up to two medium Claims, pixel/OCR anchors, bounded hypotheses, atomic apply |
| ReAct | Gemini | one claim/hypothesis-owned native tool action, route and provider bounds |
| Observation reduction | none | Discovery/Evidence/Failure separation and immutable provenance |
| Discrepancy Decision | Gemini | reviewed Evidence scope, ownership, anchors, budgets, atomic apply |
| Coverage/basis | none | exact evidence-determined or bounded-binary boundary and smallest allowed basis |
| Judgment | Gemini | exact basis/ID equality; select binary verdict only when Evidence did not already determine it |

## Image Account Planning

Planning is a standalone request and receives a controlled original-image view.
It also receives a compact observation packet containing deduplicated perception,
positioned OCR, all non-mechanical pixel/OCR VisualFact anchors, and retrieval
clues. Deterministic bootstrap tasks and lifecycle handoff metadata stay in the
canonical archive but are not presented as Planning output examples. It emits
positive ImageClaims and tentative SearchHypotheses; it does not create a core
verdict fact or verdict. The Planning schema exposes no Claim key or per-Claim
verification question on a hypothesis.
It must still emit at least one executable route; this is a structural requirement,
not a rule about which fact the route should investigate. A Hypothesis with explicit
web queries must include `text_search`; otherwise those queries would never execute.

Each ImageClaim states the underlying real-world proposition conveyed to the
viewer. It does not replace that proposition with the easier meta-claim that visible
text, a post, or an advertisement merely makes the assertion.
Exactly one Claim is high salience and preserves the complete central relation;
optional independent Claims are medium rather than fragments of that relation.

The image defines the account to fact-check and supplies initial clues; it does not
bound the investigation's facts, sources, relations, or query vocabulary. Planning
should independently establish the underlying real-world facts rather than merely
look for the value proposed by the image. Prior knowledge may contribute tentative
SearchHypotheses, but only tool Evidence can establish them.

Image Account Planning uses Gemini `thinking_level=high` by default because it must
separate the depicted value from the underlying fact to investigate. Its thought
tokens are recorded in the trace. Investigation, extraction, visual tools, Decision,
and Judgment remain concise and do not treat hidden reasoning as Evidence.

## ReAct and native protocol

Each action starts a fresh native tool round trip and selects exactly one active
claim/hypothesis task. `previous_interaction_id` is used only between that action's
`function_call` and `function_result`; it is never passed to the next action or
semantic stage. The runtime dynamically exposes
only untried bounded routes. A retrieval batch may enable one concrete `visit` or
`compare_with_reference` action. Pending candidates are inspected before repeating
retrieval. Claim/hypothesis ownership preserves lineage; it is not a semantic cage.
Gemini may choose any useful query angle, including an independent question about
the underlying real-world fact. Such a query does not change the ImageClaim or
create Evidence.

Archive recall is a bounded aid within an executable task. It is hidden before the
task creates archived investigation material, permits at most two recall/read cycles
per task, and exposes exact reads only for pending recalled IDs. Recall candidates
remain memory and cannot become Evidence by being retrieved again.

One action exposes one task-scoped route family. For webpage inspection Gemini
selects one Claim ID already owned by that task and writes the passage it wants;
the runtime binds the exact Claim text for stance extraction. This prevents URL,
Task and Claim choices from being recombined across unrelated routes.

After Planning, the reducer broadly registers each initial route against the current
image account so Evidence, budgets, and stopping remain auditable. This internal
attachment is not shown as a Planning choice and cannot establish a ClaimAssessment
or MaterialDiscrepancy without a qualified directional Finding/Evidence chain.

After tool execution, the runtime reduces the canonical result, stores the native
`function_result`, and ends the action segment. The next semantic checkpoint submits
that pending result followed by one `user_input` containing the newly compiled state.
No provider, model, or wire-protocol switch may hide an error.

## Discrepancy Decision

The standalone checkpoint runs sparsely after qualified direct Evidence, same-capture/reference
comparison, a material Evidence boundary, or immediately before unresolved
termination. It sees ImageClaims, hypotheses, exact Evidence, visible anchors,
attempted routes, and remaining budgets through the explicit workspace handoff.

It may assess claims, establish or conflict a MaterialDiscrepancy, add/retire bounded
hypotheses, request one Evidence-motivated visual reinspection, and propose a verdict.
It may not cite search snippets, invent IDs, expand the image account, or turn
provider failure into a factual verdict.

A qualified refutation of one high-salience Claim is already decisive for `fake`.
Another unresolved Claim does not lower that contradiction to supporting or require
the investigation to reconstruct every other aspect of the image first.

## Observation semantics

Search and reverse-image output create Discovery. A fetched exact web passage or
successful visual observation may create Evidence. Extractor stance is
query-relative; it cannot change an ImageClaim. Only an accepted Discrepancy Decision
updates ImageClaim semantics.

Web stance follows whether the exact passage makes the bound Claim true or false.
Merely reporting that the Claim was asserted is not support for its truth, while a
direct denial is refutation. The action's retrieval goal never reverses this logic.

Evidence and Findings preserve task ownership. A discrepancy affecting multiple
claims needs owned qualified Evidence for every affected claim and visible anchors
that overlap each claim.

## Judgment and export

The deterministic compiler selects the exact claims, discrepancies, visible anchors,
Findings, Evidence, and unresolved gaps. Evidence-determined Judgment reproduces the
compiled binary verdict; unresolved terminal cases use bounded binary Judgment over
the same complete basis. Neither mode adds facts. The strict auditor and
`ifv-policy-v2` exporter reject unknown IDs,
misalignment, post-verdict actions, protocol rejection, private evaluator data, or
active legacy core ownership.

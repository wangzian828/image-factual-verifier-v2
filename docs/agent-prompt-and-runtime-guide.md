# v4 Prompt and Runtime Guide

## Boundary map

| Stage | Semantic owner | Deterministic enforcement |
| --- | --- | --- |
| Case validation | none | exact three-field row, path boundary, SHA-256 |
| Perception/OCR | Gemini/tool provider | required-tool success and literal schemas |
| Bootstrap | none | stable visual facts and retrieval anchors |
| Image Account Planning | Gemini | 1-3 claims, pixel/OCR anchors, high salience, bounded hypotheses, atomic apply |
| ReAct | Gemini | one claim/hypothesis-owned native tool action, route and provider bounds |
| Observation reduction | none | Discovery/Evidence/Failure separation and immutable provenance |
| Discrepancy Decision | Gemini | reviewed Evidence scope, ownership, anchors, budgets, atomic apply |
| Coverage/basis | none | exact fake/real/unverifiable preconditions and smallest allowed basis |
| Judgment | Gemini explanation only | exact compiled verdict and ID equality |

## Image Account Planning

Planning is the stored main Interaction root and receives the original image once.
It also receives perception, positioned OCR, pixel/OCR VisualFacts, retrieval
anchors, and bootstrap tasks. It emits positive ImageClaims and tentative
SearchHypotheses; it does not create a core verdict fact or verdict.

External remembered metadata is not an ImageClaim merely because it could identify
the image. It belongs in a SearchHypothesis until Evidence supports its relevance.

## ReAct and native protocol

Each action continues the main Interaction through `previous_interaction_id` and
selects exactly one active claim/hypothesis task. The runtime dynamically exposes
only untried bounded routes. A retrieval batch may enable one concrete `visit` or
`compare_with_reference` action. Pending candidates are inspected before repeating
retrieval.

After tool execution, the runtime reduces the canonical result, stores the native
`function_result`, and ends the action segment. The next semantic checkpoint submits
that pending result followed by one `user_input` containing the newly compiled state.
No provider, model, or wire-protocol switch may hide an error.

## Discrepancy Decision

The checkpoint runs sparsely after qualified direct Evidence, same-capture/reference
comparison, a material Evidence boundary, or immediately before unresolved
termination. It sees ImageClaims, hypotheses, exact Evidence, visible anchors,
attempted routes, and remaining budgets in the inherited original-image chain.

It may assess claims, establish or conflict a MaterialDiscrepancy, add/retire bounded
hypotheses, request one Evidence-motivated visual reinspection, and propose a verdict.
It may not cite search snippets, invent IDs, expand the image account, or turn
provider failure into unverifiable.

## Observation semantics

Search and reverse-image output create Discovery. A fetched exact web passage or
successful visual observation may create Evidence. Extractor stance is
query-relative; it cannot change an ImageClaim. Only an accepted Discrepancy Decision
updates ImageClaim semantics.

Evidence and Findings preserve task ownership. A discrepancy affecting multiple
claims needs owned qualified Evidence for every affected claim and visible anchors
that overlap each claim.

## Judgment and export

The deterministic compiler selects the exact claims, discrepancies, visible anchors,
Findings, Evidence, and unresolved gaps. Judgment reproduces that basis and adds no
new facts. The strict auditor and `ifv-policy-v2` exporter reject unknown IDs,
misalignment, post-verdict actions, protocol rejection, private evaluator data, or
active legacy core ownership.

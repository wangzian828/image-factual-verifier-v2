# v4 Prompt and Runtime Guide

中文备份见 [agent-prompt-and-runtime-guide-zh.md](agent-prompt-and-runtime-guide-zh.md)。
代码中的 prompt 常量与 schema 是运行时唯一准则。
本文件是运行时说明，不是实际 system prompt 原文。

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
`target_facts` rows (represented internally as `ImageClaim` objects) and
tentative SearchHypotheses; it does not create a core
verdict fact or verdict. The Planning schema exposes no Claim key or per-Claim
verification question on a hypothesis.
It must still emit at least one executable route; this is a structural requirement,
not a rule about which fact the route should investigate. Non-empty Hypothesis
`queries` already declare a text-search route, so the reducer derives `text_search`
without editing query content. `suggested_tools` carries additional capabilities.
Before the reducer commits any Claim, Hypothesis, or Task, the complete Planning
object is checked against the active SourceAccessPolicy. A blocked query causes a
bounded `planning_revision`; the runtime neither rewrites the query nor commits the
safe-looking siblings from that rejected object. The same active policy filters
provider result rows before their titles, snippets, aggregates, or URLs become
Discovery or provider-visible context. Known fact-check domains are excluded at
this retrieval boundary for active evaluation policies; the filter does not rewrite
queries or decide Evidence semantics.

Each ImageClaim states the underlying real-world proposition conveyed to the
viewer. It does not replace that proposition with the easier meta-claim that visible
text, a post, or an advertisement merely makes the assertion.
It is written positively as what the image asks the viewer to accept, never as a
suspicion, contradiction, authenticity judgment, or verdict. Search queries seek
the underlying facts and sources rather than a ready-made fact-check answer.
Exactly one Claim is high salience and preserves the complete central relation;
optional independent Claims are medium rather than fragments of that relation.

The image defines the account to fact-check and supplies initial clues; it does not
bound the investigation's facts, sources, relations, or query vocabulary. Planning
should independently establish the underlying real-world facts rather than merely
look for the value proposed by the image. Prior knowledge may contribute tentative
SearchHypotheses, but only tool Evidence can establish them.

Image Account Planning uses Gemini `thinking_level=high` by default and requests
the provider's `thinking_summaries=auto`. The trace may record the returned thought
summary and thought-token count; this is not the hidden chain of thought.
Investigation, extraction, visual tools, Decision, and Judgment remain concise and
do not treat thought summaries as Evidence.

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
The ReAct policy is not asked to emit a segment-summary JSON object: after one
accepted tool call, the runtime closes the action segment and compiles the next
handoff. Reflection, Replan, Decision, and Judgment are separate checkpoints and
are not required before every tool action.

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

If v4 ReAct repeatedly selects rejected duplicate routes until its correction budget
ends, Runtime sends no extra model request and invents no search. It appends a
deterministic boundary tied to the rejected request IDs, records the current Task as
`blocked` without increasing the action count, and starts a standalone Discrepancy
Decision. The episode remains visible to audit and is ineligible for policy export.
Transport, schema, lifecycle, and non-v4 correction exhaustion still fail closed.
The v4 Discrepancy Decision may use the same deterministic boundary as an empty
`continue` checkpoint after its final rejected update; it never invents a Claim,
Evidence, discrepancy, or binary verdict.

## Discrepancy Decision

The standalone checkpoint runs sparsely after qualified direct Evidence,
same-capture/reference comparison, a material Evidence boundary, bounded route
selection exhaustion, or immediately before unresolved termination. It sees
ImageClaims, hypotheses, exact Evidence, visible anchors, attempted routes, failures,
and remaining budgets through the explicit workspace handoff.

It may assess claims, establish or conflict a MaterialDiscrepancy, add/retire bounded
hypotheses, request one Evidence-motivated visual reinspection, and propose a verdict.
It may not cite search snippets, invent IDs, expand the image account, or turn
provider failure into a factual verdict.
`supported` means the exact ImageClaim is true and `refuted` means it is false.
Before a correction turn, the runtime reports all independent ID, ownership,
direction, and Finding-chain contract errors it can establish from the same output;
it also reports incompatible verdict/route preconditions instead of serially hiding
later errors behind the first failure. A final parseable JSON object rejected by
either schema or semantic validation remains an `output_rejected` trace step with
its exact fields and reason; it is not mislabeled as an empty format error.
Coupled atomic consequences are reported together: refuting a high-salience Claim
requires its decisive established discrepancy and `fake` proposal in the same JSON.
An already-recorded discrepancy is canonical state, not a template for the next
update. Repeating the same affected Claim, Evidence, materiality, and status is
rejected as a structured duplicate without demanding another ClaimAssessment.

A qualified refutation of one high-salience Claim is already decisive for `fake`.
Another unresolved Claim does not lower that contradiction to supporting or require
the investigation to reconstruct every other aspect of the image first.

## Observation semantics

Search and reverse-image output create Discovery. A fetched exact web passage or
successful visual observation may create Evidence. The extractor labels whether the
passage covers the same complete Claim relation, a partial relation, a different
instance, or an unclear scope; it separately labels support, contradiction,
background, or uncertainty. Only a same complete relation with support or
contradiction is directional. Other exact spans remain reviewable neutral context.
The action's retrieval goal selects text but never determines these labels or changes
an ImageClaim. Only an accepted Discrepancy Decision updates Claim semantics.

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
Rejected Planning revisions remain visible for audit but are not SFT targets.

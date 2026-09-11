# PSD completion and acceptance checklist

This checklist supersedes any earlier claim that PSD was fully accepted.
Implementation, synthetic tests, live-provider tests and a real training round
are separate milestones. A component test is not an end-to-end acceptance.

## Scope and ordering

Do not modify the Agent runtime, consume evaluation cases as training data, or
interrupt the baseline → judge → SFT → post-SFT evaluation → judge experiment.
Edit/test/commit locally, push GitHub, and only fast-forward pull on the server.
Data, API artifacts, archives and checkpoints stay under `/volume/ybo/wza`.
Retain the 131072-token upper bound; do not silently truncate a PSD target.

## Acceptance matrix

| Stage | Required acceptance | Initial audit |
| --- | --- | --- |
| Fresh rollouts | Frozen training split, current checkpoint, numeric token capture, raw request archive | Provenance gates exist; no real PSD training rollout collected on H20 |
| Localization | Earliest evidenced recoverable policy decision, exact source-index/hash binding | Validator exists; automatic semantic verifier missing |
| Hint | Procedural, no answer/query/exact action leakage; private context isolated | Proposer/audits exist; no real repaired training case accepted |
| Replay | Original prefix/media unchanged; frozen current policy supplies replacement and complete suffix | Runtime adapter exists; full real-case acceptance missing |
| PSD judge | Direct image/evidence review, selected-step repair, complete grounded episode; reject fake repairs | Only external artifact validation; automatic judge missing |
| Admission | Source failed, local verifier passes, correct complete episode, strict audit, exact sampled token hashes | Existing gates require integration, persistence and tamper tests |
| Targets | Repair decision only; passing-turn preservation; unhinted student; immutable ordered media | Numeric/media component tests pass; real repair assembly not demonstrated |
| Loss/update | Frozen same-checkpoint top-20, upstream per-target weights, LoRA rank32, SP4 | Real 9B short synthetic CPU update/reload passes; not a real PSD round |
| Resume | Preserve successful calls, retry transport failures only; optimizer/scheduler/RNG resume; next round uses new weights and fresh rollouts | Top-k/offline finalizer exist; live judge resumption missing |
| Outcome | Independent held-out before/after evaluation and judge, repair/preservation breakdown | Not measured; loss reduction is not capability improvement |

## Work sequence

1. Complete the automatic semantic localizer and multimodal PSD judge; bind
   requests, images, full episode, prompt version, model and token hashes.
2. Wire them into repair generation and offline finalization, saving each
   expensive result before later stages can fail. No successful trajectory
   resampling just because a verifier timed out; no retry-until-pass judging.
3. Add positive/negative/tamper/resume tests and live blinded synthetic controls
   (including label-only, unavailable-tool, leaked-answer and prompt-injection).
   Report these as synthetic diagnostics, not real repair quality.
4. Locate training public cases plus independent private reference material;
   collect fresh bounded current-Qwen rollouts and validate real repairs. The
   transferred SFT messages alone are not raw on-policy archives/private gold.
5. Assemble verified repairs plus preservation, collect frozen top-20,
   materialize and validate datums, execute an optimizer/resume acceptance when
   GPU scheduling permits, then validate next-round checkpoint provenance.
6. Update this report with exact commands/results/commits and unresolved gates.
   Never substitute fabricated artifacts or evaluation data for missing inputs.

Reference: [PSD paper](https://www.canvas.inc/research/privileged-self-distillation)
and upstream revision `778be78bdac582b51a975ff819046583aad383e0`.

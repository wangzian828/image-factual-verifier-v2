# PSD upstream parity audit

Audit date: 2026-09-11

Upstream reference:

- Repository: <https://github.com/essamsleiman/psd>
- Audited commit: `778be78bdac582b51a975ff819046583aad383e0`
- Method article: <https://www.canvas.inc/research/privileged-self-distillation>

The reference repository is checked out separately on the Spark source host at
`/home/liuruiqi/wza/psd-upstream-reference`. It is not vendored into this
repository. One upstream implementation note contains an obsolete historical
abbreviation; the current public method and this repository use PSD.

## Contract comparison

| Official PSD contract | IFV implementation | Status |
|---|---|---|
| Collect failures from the current policy's own rollout | `psd_candidates.py` accepts explicit train cases and retains the exact student-reached raw-history step | Aligned |
| Locate a recoverable model-reached decision | strict/runtime locations plus named verifier-guided semantic localization in `psd_repair.py` | Aligned |
| Hint constructor may use privileged training information | separate `hint_constructor` role and audited L1-L3 procedural hints | Aligned |
| Frozen self-teacher is the round-start policy | teacher/student provider, model, checkpoint and checkpoint-manifest hash must match | Aligned and fail-closed |
| Teacher sees the hint; student receives the same state without it | separate teacher/student native histories; role flags are mandatory | Aligned |
| A tool call is not proof of repair | local admission requires a named task-verifier artifact with checks and observed evidence | Aligned |
| Initial no-hint rollout failed | private-gold scorer must explicitly return `result_correct=false` | Aligned |
| Hinted continuation passes locally and as a complete task | local verifier, strict trace audit, terminal report and private-gold result must all pass | Aligned |
| Unhinted retry is diagnostic, not an admission gate | `unhinted_student_affects_acceptance=false` | Aligned |
| Distil the hinted teacher distribution, not the hint model | only `frozen_self_teacher` may supply the distribution | Aligned |
| Exact token IDs; no text decode/re-encode | raw prompt/completion token capture and hashes are required | Aligned |
| Verified episode must contain the exact distilled teacher action | full hinted trace and local verifier bind the teacher prompt/completion hashes | Aligned and fail-closed |
| Sparse top-20 normalized teacher distribution | exact length, uniqueness, finite probability and normalized-mass checks | Aligned |
| Loss only on completion next-token positions | prompt/observation positions receive zero weight | Aligned |
| Preservation comes from verified current-policy passes | full-task pass plus strict trace audit are mandatory | Aligned |
| Published target weighting is per-target, without aggregate source rebalancing | every repair and every retained preservation step keeps its configured row weight; both kinds remain mandatory | Aligned |
| Row weights affect gradient magnitude | weighted token losses are summed, then averaged across datums | Aligned; fixed from weight-mass normalization |
| Resumable, provenance-bound target materialization | target IDs, token hashes, teacher identity, checkpoint and manifest hash are checked | Aligned for imported/online caches |
| Training refuses incomplete target packages | launcher verifies manifest schema/status, file hash, top-K, context, source kinds and effective mass | Aligned |
| Finite forward/backward and recoverable checkpoints | plugin rejects non-finite weights/loss; launcher records resources and supports checkpoint resume | Aligned at framework-smoke level |
| Every new PSD round recollects from the previous round output | rollout gate binds the served round-start checkpoint; round completion binds the new checkpoint; later rounds require that exact prior output and a new run ID | Aligned and fail-closed |

## Intentional backend differences

The official releases use Tinker for Qwen3.5-9B and River for the larger
Qwen3.6 experiment. IFV uses ms-swift and a local Qwen-compatible Chat
Completions runtime. This is an infrastructure substitution, not a method
change:

- top-20 probabilities are captured during the actual frozen-policy
  continuation when the server exposes numeric token IDs and top logprobs;
- the same sparse `[T, K]` target representation is passed to a custom
  ms-swift cross-entropy loss;
- imported caches are accepted only when their teacher deployment and exact
  token hashes match the target package.

The IFV implementation deliberately uses stricter full-episode and trace
audits than the local BFCL turn checker because an image fact-check result can
be structurally valid yet unsupported.

## Changes made during this audit

- `5ab2ab9`: enforced role semantics, strict preservation admission, both
  source kinds, 1:1 effective mass, cache teacher provenance and upstream loss
  scaling.
- `788ddcb`: bound local/full verification to the exact teacher prompt and
  completion token hashes.
- `99beb22`: added PSD datum preflight, plugin preflight, environment gate,
  checkpoint-space gate and resume support.
- `9d7b9a2`: bound repair collection to a serving profile and immutable
  checkpoint-manifest digest.
- Current completion pass: corrected the earlier aggregate 1:1 interpretation
  to the published Qwen3.5 recipe: 506 repair targets and 815 preservation
  step-targets, each at weight 1.0 before token-length effects. Datum manifests
  now state `weighting_policy=per_target` and reject aggregate rebalancing.

## Validation boundary

The following are not yet empirical claims:

1. No real IFV Qwen repair/preservation package exists yet, so an actual
   top-20 teacher collection followed by an optimizer step has not been run.
2. The repair driver consumes a separately produced, hash-bound complete
   hinted-episode artifact. It does not yet reconstruct and finish the entire
   IFV Orchestrator episode by itself from an arbitrary historical step.
3. The current PSD training profile is a LoRA, one-step, sequence-parallel-size
   1 engineering smoke. The SFT environment supports 131072 tokens, but that
   does not prove a 131072-token PSD target fits or that PSD can use SP8.
4. Multi-round chaining is now enforced by immutable rollout and completion
   gates, but it has not yet been exercised on a real two-round IFV run.

Formal PSD training therefore remains gated on real training data, a real
five-case repair smoke, complete verifier-bound episodes, top-20 capture, and
one real optimizer-step canary. Those checks must pass before a full run is
started.

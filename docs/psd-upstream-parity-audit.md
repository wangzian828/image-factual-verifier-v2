# PSD upstream parity audit

Audit date: 2026-09-11

## H20 follow-up: multimodal implementation and validation

The gap described below has now been implemented in `psd_media.py`,
`psd_hf_teacher.py`, target/repair assembly and the native Qwen3.5 Swift
template. Image bytes, order (including repeats), processor settings, tensor
hashes, grids and cache identity are bound. Missing/changed media and mismatched
placeholder grids fail before scoring or training. Teacher scoring keeps exact
token IDs and uses the original round-start checkpoint. The text-only vLLM
Completions fallback is never used for image targets.

Validation on the H20 host, without interrupting GPU evaluation:

- Real deployed Qwen3.5-9B weights on CPU, two repeated images, hinted teacher
  top-20, unhinted student LoRA rank32 optimizer step: passed.
- Loss 8.3161335; visual adapter gradient L1 15815.1431; changing image pixels
  changes logits (maximum difference 18.0).
- Adapter save/reload reproduces logits exactly (maximum difference 0.0);
  optimizer state and RNG save/reload passed.
- Four-rank native Swift SP loss/gradient smoke passed; reference and SP loss
  both 5.741323, without gathering full-vocabulary logits.
- Reports: `/volume/ybo/wza/runs/psd-multimodal-9b-cpu-smoke/result.json` and
  `/volume/ybo/wza/runs/psd-sp4-cpu-smoke.json`.

These checks validate the implemented data/model path, not repair quality on
IFV tasks or 128K GPU capacity. Real training-only verifier-approved repairs,
a full H20 optimizer probe on the actual target lengths and any multi-round
quality claims still require their respective run artifacts. The current
baseline/SFT/post-training evaluation order is unchanged.

### Historical finding that motivated this change

The alignment table below describes the text-token implementation, not a
validated image-Agent training path. The H20 follow-up found a material gap:
target builders, forced-token Completions scoring and the pre-tokenized Swift
template carry no image pixels, image grids or multimodal position IDs.
Exact prompt token IDs alone do not preserve visual conditioning. Therefore
the earlier "aligned" entries must not be read as multimodal readiness.

Visual targets now fail explicitly before target export, offline scoring,
datum materialization and template encoding. The Qwen3.5 special IDs are
verified against the deployed 9B config (vision start/end 248053/248054,
image/video 248056/248057). This is a correctness guard, **not completion of
multimodal PSD**. Existing text-only unit/sparse-loss tests remain useful but
cannot establish image-Agent readiness.

The profile gate now accepts four ranks with SP4 as well as eight with SP8.
Both preserve 32 unique targets per optimizer step through accumulation 32;
SP ranks cooperate on one datum and must not be counted as independent data.
This does not certify H20 capacity or replace a real optimizer canary.
The shared launcher root was also corrected from `training/` to repository
root, preventing duplicated `training/training/` script and plugin paths.

Remaining acceptance work, using training cases only:

1. Preserve immutable media references, byte hashes, processor settings and
   visual ordering from the actual archived teacher/student requests.
2. Implement exact-completion teacher scoring with those media inputs. The
   existing token-only Completions fallback is unsuitable for visual targets.
3. Feed processor pixel tensors, grids and independently computed teacher and
   student multimodal positions through the datum/template/SP paths without
   re-tokenizing the banked completion or exposing the hint to the student.
4. Verify multi-image and repeated-image cases, missing/changed media rejection,
   frozen-teacher provenance, causal loss positions and finite gradients on
   real H20 inputs. Then run repair verification and an optimizer canary.

Official evaluation cases, including the frozen 1527 and ongoing 1682 runs,
must never be used to construct PSD training targets. The current baseline,
SFT and post-training evaluation sequence remains separate from this work.

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
| Resumable, provenance-bound target collection/materialization | forced token-ID scoring through vLLM; target IDs, returned IDs, token hashes, serving profile, teacher identity, checkpoint and manifest hash are checked; successful rows are fsynced and skipped on retry | Aligned |
| Training refuses incomplete target packages | launcher verifies manifest schema/status, file hash, top-K, context, source kinds and effective mass | Aligned |
| Finite forward/backward and recoverable checkpoints | plugin rejects non-finite weights/loss; launcher records resources and supports checkpoint resume | Aligned at framework-smoke level |
| Repair verification can finish after asynchronous local audit without resampling | driver persists the complete teacher episode; in-place offline finalization makes zero provider calls, keeps a pre-finalize backup and updates attempts atomically | Aligned |
| Native long-context sparse loss | top-20 targets follow ms-swift's SP/RP split; only scalar position losses are gathered and fp32 CE is chunk-recomputed | Aligned backend substitution |
| Published optimizer recipe | LoRA rank 32, `4e-5`, 32 unique targets/step, five epochs, clip 1.0, seed 0 | Aligned in the production profile |
| Every new PSD round recollects from the previous round output | rollout gate binds the served round-start checkpoint; LoRA serving records base/adapter separately; round completion binds the new adapter; later rounds require that exact prior output and a new run ID | Aligned and fail-closed |

## Intentional backend differences

The official releases use Tinker for Qwen3.5-9B and River for the larger
Qwen3.6 experiment. IFV uses ms-swift and a local Qwen-compatible Chat
Completions runtime. This is an infrastructure substitution, not a method
change:

- top-20 probabilities are captured during the actual frozen-policy
  continuation when the server exposes numeric token IDs and top logprobs;
- incomplete online captures are scored exactly once more by forcing the
  original `teacher_prompt_ids + completion_ids` through the same attested
  frozen vLLM checkpoint and selecting the completion-position prompt
  logprobs, mirroring upstream `river_topk.py`;
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
- Current completion pass: added the native SP8 sparse-loss path, fixed the
  production profile to the published Qwen3.5 recipe, wired Gemini only as the
  hint constructor, and added an attested resumable vLLM forced-token top-20
  collector.

## Validation boundary

The following are not yet empirical claims:

1. No real IFV Qwen repair/preservation package exists yet, so an actual
   top-20 teacher collection followed by an optimizer step has not been run.
2. The SP8 target ordering, global loss and gradient scaling have a distributed
   CPU smoke, and the checked-in profiles are fixed to eight-GPU 128K SP8. This
   does not prove a 131072-token Qwen target fits on the eight A100s; that claim
   requires the real one-step memory probe while all eight GPUs are idle.
3. Multi-round chaining is now enforced by immutable rollout and completion
   gates, but it has not yet been exercised on a real two-round IFV run.

The frozen GPU-13 vLLM environment (`0.23.1rc1.dev1348+g47f1b47a7`) was
inspected directly: both Chat Completions and Completions expose
`prompt_logprobs`, `return_token_ids`, `return_tokens_as_token_ids`, and the
response protocol carries `prompt_token_ids` plus integer-keyed per-position
logprobs. A real Gemini native Interactions structured-output smoke also
completed with `thinking_level=low`; that validates hint-constructor transport,
not Qwen repair quality. The remaining empirical boundaries above are kept
explicit.

Formal PSD training therefore remains gated on real training data, a real
five-case repair smoke, complete verifier-bound episodes, top-20 capture, and
one real optimizer-step canary. Those checks must pass before a full run is
started.

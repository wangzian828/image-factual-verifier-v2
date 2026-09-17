# PSD checker-feedback correction (2026-09-17)

## Latest user scope: 100-case paired evaluation, not full-test inference

After the existing 400 × 8 PSD round and its five training epochs, run a small
paired comparison of the original three-epoch SFT checkpoint and the new PSD
checkpoint. **Each model gets the same 100 cases** (200 complete Agent runs in
total), following smoke validation; do not launch 1,527-case inference or enlarge
this subset without a new user decision. The training scope is unchanged.

The list is frozen at
`/volume/ybo/wza/runs/psd-paired-eval100-20260917-v1/case-list.txt`.
Its `selection.json` binds the existing 1,526-case public runtime manifest and a
deterministic outcome-independent selection, using the existing
`scripts/prepare_psd_paired_eval.py`, count 100, salt
`psd-sft3-vs-psd-fixed100-v1`. This is a subset of the registered 1,527-case test
set, selected from its available-image runtime release; it does not change the
full-test denominator or include the historically absent image.

The sample has **26 supported / 74 refuted**, with all 100 image paths present
at selection time. Case-list SHA-256:
`95c9e25b1fdab72e12031e73c61edfccd240a08a6671d9dd8c29d2245764d6b0`.
No model outcomes were used for selection, no images were copied, and no new
inference/provider calls were launched by this preparation. The list must not
be redrawn because of model failures, scores, or inconvenient cases.

Use one frozen Agent/tool/prompt/serving/sampling/budget protocol for both arms;
fresh paired runs avoid silently comparing different historical serving recipes.
Resume already completed outputs of these exact paired runs rather than repeat
successful cases. Preserve the SFT3 original checkpoint. Both arms use the same
established **evaluation** judge/model/material coverage, not the PSD repair
reviewer's identity. Report BAcc, Macro-Precision, Macro-F1, supported/refuted
recall, SESR, completion/failure rate, tool/thinking behavior and cost/latency.
For this comparison the denominator is **100**, with failed cases retained;
class recall denominators remain 26 and 74. Do not label these as 1,527-case
results or overwrite the existing full-test main table. Treat the pilot as a
small-sample trend check, not proof of a significant/general improvement.

## Defect and upstream boundary

The private source checker could fail a report for fabricated support or an
unsupported assertion, while an independent localizer declared the episode
correct because its binary verdict was correct. `run_psd_repair_driver.py`
then returned `no_recoverable_site`, before trying any hint or policy rerun.
The slate proposer also lacked the source checker's cited observations.
This was our extra admission veto, not upstream PSD behavior.

Pinned upstream:

- [BFCL slate search](https://github.com/essamsleiman/psd/blob/778be78bdac582b51a975ff819046583aad383e0/experiments/bfcl/repair/slate.py)
  feeds checker results into search; an empty slate executes a plain retry.
- [Agentic instructions](https://github.com/essamsleiman/psd/blob/778be78bdac582b51a975ff819046583aad383e0/experiments/bfcl/agentic/system_prompt.md)
  use observed errors/checker feedback without reference answers. They permit
  stopping when further hints cannot help; they do not require our independent
  localizer veto. Upstream slate and agentic drivers are distinct paths.

## Correction

- Slate mode retains the verified-source-failure gate but skips independent
  semantic localization. Its first policy decision is explicitly a mechanical
  **replay origin**, not a claim that this decision caused the failure.
- The proposer receives failed-checker status, acceptance criteria and literal
  checker citations with source indices. Every copied quote must occur in the
  observed trace (or the actual previous injected hint for a hint audit).
  No private explanation, gold or corrected answer is copied. This is a
  deterministic, task-specific projection, not another LLM arbitration gate.
- Previously verified hints must still be retained. A genuinely empty slate
  runs a complete plain retry within the same six-rerun budget. A passing
  unhinted retry creates **zero repair targets**, never a fabricated hint target.
- `no_recoverable_site` cases re-enter the existing search. Old manifests and
  semantic-localization artifacts are retained. Legacy `no_further_grounded_hint`
  resumes the already accepted empty proposal without resetting its budget.
- Completed proposals, continuations, verifier decisions and successful targets
  are reused. Original source traces and SFT3 checkpoint are not changed.
- Slate round hint keys are now strings before hashing. Historical integer-key
  digests are readable only if reconstructing those exact integer keys recovers
  the original digest. No checksum bypass or arbitrary cache repair is allowed.

The acceptance criteria, exact token/pixel checks, raw teacher scoring, native
tool workflow and final capability evaluation are not relaxed.

## Scope and verification

The user reaffirmed **400 images × 8**, finish this round before expanding the
sample pool. Preserve all 3,200 collected trajectories. Final accounting must
separate source rollouts, unique images, repair attempts and accepted training
targets; do not describe all non-target trajectories as deleted.

Local and H20 targeted tests: **55 passed** (slate runtime, empty-slate full
rerun, cached resume, private-feedback exclusion, observed quotes, binding
integrity and incremental model routing).

Deployment: `/volume/ybo/wza/training-artifacts/psd-checker-feedback-20260917-v58`.
Search stays `psd-production-round1-20260917-v1/search-gemini37-flash-high`;
uncached external calls remain routed to **Gemini 3.6 Flash / high**, 16 in
flight. Historical directory/model identity is not the wire model.

Real canary **completed** for main-06188 and main-06217, both previously vetoed
by the localizer. Each used one proposal and one complete Qwen rerun, then passed
the full-task checker and strict target assembly. They produced respectively
**2 and 1 accepted local targets**, in 241.42 and 192.24 seconds. No rejected
proposal. These two diagnostic successes do not estimate the overall repair
rate. Another 55 local repair/source-verifier regression tests passed (110
local targeted tests total; the first 55 were also run on H20).

Canary outputs live under the existing search's
`checker-feedback-validation-v58`; these bounded diagnostic reruns are separate
from formal training targets. The staged launcher requires two completed real
reruns before switching only the owned prepare-controller group. In-flight
interrupted attempts remain recorded and use the ordinary recovery budget.
GPU services, GPU guard and friend experiments are not restarted.

After source repair completes: assemble accepted repair/preservation targets,
frozen-teacher exact top-20 scoring, final four-GPU global-batch-32 short-step
acceptance, then the planned five-epoch PSD run in a new output directory.
This code correction alone is **not** completion of that training pipeline.

## Live handoff and follow-up boundary

v58 was committed/pushed locally as `a0826e6` and switched into the existing
formal search. Controller receipt is
`/volume/ybo/wza/runs/psd-formal-prepare-controller-20260917-checker-feedback-v1/process.json`
(launch PID 1412183 is a hint, not an identity check). The scoped controller
restart recorded 27 interrupted in-flight infrastructure attempts; all completed
results were retained and the interrupted attempts use the ordinary persisted
recovery budget. Thirteen formerly vetoed cases had reopened at the next check.

Live inspection exposed a boundary missing from the v58 feedback projection:
source admission also accepts deterministic verdict/structural failures when
the semantic reviewer said pass. The projection must use the **already verified
overall source failure**, not incorrectly demand semantic fail as well. v59
passes that proof to the projection without serializing expected verdicts or
private explanations. Local/H20 targeted tests: **57 passed**, plus the earlier
55 local repair/source tests. Three affected saved real sources also pass the
corrected gate, with **zero new provider calls**.

v59 deployment is `/volume/ybo/wza/training-artifacts/psd-source-gate-20260917-v59`.
`scripts/server/resume_psd_source_gate.py` waits for the current v58 prepare pass
to finish, verifies there is no remaining prepare child, and only then replaces
the controller with `psd-formal-prepare-controller-20260917-checker-feedback-v2`.
It does not interrupt running investigations or restart models. Read its
`handoff-state.json` before deciding which controller owns the run.

Remaining pre-existing paused cases are **not** all transient provider errors:
some cached slate reviews violate literal-quote/position contracts; some retry
ledgers have unresolved/nonretryable attempts. One strict-assembly rejection
(`30bdc9a8739a2eb7`, main-06268) is a semantic-review pass conflicting with the
deterministic private verdict check (`private_gold_verdict_mismatch`). It remains
rejected, not a training target. These need bounded corrective feedback/recovery
before the final bank can be complete; blindly replaying the same invalid cache
does not solve them. Do not erase them, weaken admission, or report that every
PSD issue/full training is finished. Continue while other valid cases progress.

## Bounded tail recovery (v60, staged 2026-09-18)

Live inspection of the resumed formal pass separated two additional tail
conditions without changing the PSD method or accepting weaker targets:

- Completed slate-review responses can have a valid semantic decision but
  nonliteral/misbound citations. A separately bound evidence-only correction
  now freezes `status`, `failed_position`, `passing_positions`, and
  `explanation`; it may only copy literal repaired-trace substrings and bind
  them to valid native positions/step indices. If that immutable decision
  cannot be supported, the case remains rejected. No fail/pass decision is
  resampled.
- Direct tokenizer/live-serving httpx transport exceptions could bypass the
  backend wrapper and be persisted as `nonretryable_error`. They now consume
  the same existing infrastructure-attempt budget as wrapped transport errors.
  At a completed-pass boundary only, exact legacy transport error types are
  reclassified after byte-for-byte backup and hashing. Attempts remain charged;
  budgets and case selection are not reset.

The v60 handoff waits behind v59 and then for v59's current prepare pass to
finish. It does not interrupt active investigations. Local and H20 targeted
tests: **98 passed**. The immutable deployment is
`/volume/ybo/wza/training-artifacts/psd-tail-recovery-20260918-v60`; delayed
handoff PID 1421031 is only a receipt hint. Read its `handoff-state.json` and
the upstream v59 state before identifying the live owner. Staging does not mean
the corrections are live until that completed-pass handoff succeeds.

Live update: v60 switched at v59's completed-pass boundary with zero interrupted
rollouts. It backed up and reclassified **69** exact legacy transport ledgers;
no attempt or budget was reset. In the first live v60 pass, 22 bound
evidence-only corrections had already passed local validation (15 immutable
fail decisions and 7 immutable pass decisions). The remaining latest
`nonretryable_error` ledgers were four `ValueError` cases, so they were not
misclassified as transport. Controller:
`/volume/ybo/wza/runs/psd-formal-prepare-controller-20260918-tail-recovery-v3`.

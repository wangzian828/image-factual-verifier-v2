# PSD checker-feedback correction (2026-09-17)

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

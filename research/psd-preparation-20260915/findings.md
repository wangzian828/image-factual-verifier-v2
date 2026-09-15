# PSD preparation findings

## Verified before new experiments

- The 4,000-input v3 release is prepared; see `../../docs/psd-candidate-pool-20260914.md`. No new policy rollouts from this pool have been claimed.
- Code inspection confirms preservation is currently gated by `classification_correct` plus strict audit only. `verify_source_rollout_failure` currently accepts only an explicitly wrong final label. Together these omit correct-label/unsupported-evidence failures from semantic repair.
- Existing repair review already checks causal error correction, grounded complete episodes, procedural hints, literal evidence and immutable bindings. Reuse these boundaries; do not replace them with a label-only shortcut.
- Historical real CPU optimizer/resume and H20 SP4 optimizer tests are recorded in `../../docs/psd-completion-checklist.md`. They do not establish acceptance of new source-admission changes or production PSD improvement.

## Completed in this turn

- Bound semantic source review is integrated into new round preparation, deterministic reward postprocess and candidate routing. Correct-label semantic/structural failures now have a causal source proof; original exact-token/hinted-complete-episode checks are retained.
- Real source canary: 8/8 completed, 2 pass / 6 fail, 0 unresolved or provider errors. Original labels were correct for 5/8; 3 of these were flagged by the semantic reviewer and reached the repair-source verifier. The source run was not changed.
- Offline routing audit passed. Real resume verified 16 immutable review/provider-cache files with **zero provider calls**. The first cache-check SSH connection closed before execution; the safe retry passed. No review decision was resampled.
- Server isolated CPU regression: **219 passed** (including PyTorch-dependent tests), JUnit at `.../training-artifacts/psd-preparation-20260915/psd-tests-final.xml`. Additional deterministic postprocess source-routing controls passed locally; final local full-suite count is recorded in the log.
- Frozen server plan: 4,000 training inputs, 32 balanced training canary IDs, 400 balanced formal-test observation IDs, formal denominator 1,527. Plan reuse passed without generating new selections. No copied images/checkpoints.
- Measured preparation footprint at the end of testing: source-review directory 313,856 bytes, plan 75,264 bytes, isolated code/test directory 16,265,728 bytes (before the small final cache-audit JSON). Whole-user-directory `du` did not complete before its SSH session closed; no quota conclusion was inferred.
- Last main check: SFT 2,665/3,084, finite loss .159, no OOM/NCCL/Errno28; all GPUs 100%, 75,424 MiB each. Gemini 1,018 unique successes. These are a timestamped-in-turn snapshot, not current live counters.

## Remaining acceptance, not claimed complete

- Source-review accuracy is not measured. In particular, one OCR-attribution rejection may be a materiality boundary case; do not equate all semantic fail decisions with independently adjudicated factual errors.
- New SFT snapshot export/load and fresh canary; GPUs remain occupied by the main experiment.
- Real worst-length throughput/capacity and full-state PSD GPU save/reload/next-step acceptance.
- Actual user quota or a concrete storage budget. Shared-volume free space is insufficient evidence.
- Full production round, next-round policy collection and capability evaluation.

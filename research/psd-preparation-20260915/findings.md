# PSD preparation findings

## Verified before new experiments

- The 4,000-input v3 release is prepared; see `../../docs/psd-candidate-pool-20260914.md`. No new policy rollouts from this pool have been claimed.
- Code inspection confirms preservation is currently gated by `classification_correct` plus strict audit only. `verify_source_rollout_failure` currently accepts only an explicitly wrong final label. Together these omit correct-label/unsupported-evidence failures from semantic repair.
- Existing repair review already checks causal error correction, grounded complete episodes, procedural hints, literal evidence and immutable bindings. Reuse these boundaries; do not replace them with a label-only shortcut.
- Historical real CPU optimizer/resume and H20 SP4 optimizer tests are recorded in `../../docs/psd-completion-checklist.md`. They do not establish acceptance of new source-admission changes or production PSD improvement.

## Unverified / pending

- New semantic source-admission integration, its tests and saved-real-trace canary.
- New SFT snapshot export/load and fresh PSD rollout; GPUs are occupied by the main experiment.
- Full production round, next-round policy collection and capability evaluation.

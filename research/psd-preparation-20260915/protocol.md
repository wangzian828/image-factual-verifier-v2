# PSD preparation protocol — 2026-09-15

Status: preparation only; no production PSD training authorized by this protocol.

## Frozen scope

- Input: `/volume/ybo/wza/data/psd-candidate-pool-4000-20260914-v3`.
- 4,000 training samples, 1,421 real / 2,579 fake, 3,982 unique images.
- Retain internal provenance; describe the input as one PSD training pool in paper-facing reports.
- No training-source development holdout. A deterministic formal-test subset may be used for development observation only, never gradients, repair targets or training hints. Official evaluation uses denominator 1,527.
- Main three-epoch SFT and Gemini evaluation continue unchanged. Do not launch PSD GPU work until their scheduled work releases the GPUs. Preserve old weights and immutable experiment artifacts.
- Server operations stay under `/volume/ybo/wza`; only local Windows git commits/pushes. The separate overlap investigation is outside this task.

## Questions and predeclared acceptance

1. **Source admission:** Is a correct label being mistaken for a verified task pass? Require an independently bound source semantic review in addition to deterministic correctness and strict audit for preservation. Grounding failure can become a repair seed even when the label is correct. Transport/parse failures remain pending, not negative labels. Do not silently resample a completed semantic rejection.
2. **Causal repair:** Can a correct-label/ungrounded source be verified as a failed source without weakening exact prefix, image, trace, private-reference and token bindings? Add tests for tampered/stale review artifacts, positive sources and missing evidence. A hint must remain procedural; a repaired complete episode still requires its existing independent judge and mechanical checks.
3. **Production readiness:** Do round preparation, checkpoint identity, training-only membership, sparse top-20 targets, resume and storage gates remain enforced? Exercise affected CPU tests and an isolated server CPU diagnostic on saved real traces. No repeated GPU acceptance during the active SFT.
4. **Launch specification:** Freeze an explicit preparation/rollout/repair/train/evaluate sequence for the new SFT snapshot and 4,000 inputs. Report unresolved gates instead of claiming full capability improvement.

## Experiment order (exploratory engineering validation)

- E0: inspect current code and primary upstream method/implementation; commit this protocol before running new tests.
- E1: bounded synthetic regression matrix: grounded correct / unsupported correct / wrong / unresolved judge / malformed or changed binding / non-training membership / token and image failure. Exact expected routing, no metric-based tuning.
- E2: affected PSD regression suite, compileall, git diff checks. Existing historical real-GPU and CPU-resume results remain historical evidence, not validation of new changes.
- E3: isolated read-only real-trace diagnosis and, when credentials/configuration are already explicitly available, a bounded semantic-review canary on saved training traces. Keep API outputs and images on the server. Distinguish diagnostic outcomes from unbiased repair-rate estimates.
- E4: verify the preparation/launch manifest, cached restart behavior and blocked GPU prerequisites; record evidence and remaining gates.

## Deferred confirmatory gates

After the main SFT/evaluation work releases the GPUs: export/load the selected new SFT checkpoint, bind its identity, collect fresh on-policy canaries with complete tool/image/token archives, source-review and repair, exact top-20 materialization, nonzero optimizer update + checkpoint reload, fresh next-round rollout, then fixed-subset development evaluation and formal full-test evaluation. These are required before declaring a complete production PSD round or capability improvement. No training-time use of test gold.

## Resource limits and reporting

- No full teacher trajectory duplication; immutable media/archives and content-addressed caches are reused.
- Check user-directory usage and real quota where available; shared filesystem free space is not proof of user quota.
- Save progress per case/stage; preserve completed reviews on retry. Never turn API errors into training labels.
- Primary quality reports: BAcc and judge-based SESR, plus success/repair acceptance, tool contract errors, thinking/output length stops, wall time, token counts and storage. Distinguish all-input denominators from successful-only diagnostics.
- Log interventions and test results in `research-log.md`; summarize validated claims and limitations in `findings.md` and `research-state.yaml`.

# v3 to v4 Safety-Invariant Parity Audit

Date: 2026-07-18

Baseline: `runtime-v3-final-20260717`

This audit covers safety invariants, not the v3 core-fact control structure. v4 keeps
its discrepancy-first semantic state while restoring the validation boundaries that
must survive any orchestration refactor.

| Invariant | v3 baseline | v4 enforcement | Regression gate |
|---|---|---|---|
| Discovery is not Evidence | Search and reverse-search rows remain Discovery until an eligible inspection | v4 reducer keeps separate records; strict audit rejects in-place promotion and Discovery IDs in a verdict basis | `test_strict_audit_rejects_discovery_as_verdict_evidence`; v4 audit graph tests |
| Evidence comes from a successful call | Evidence stores `function_call_id`; audit and scoring recover the successful tool step | reducer records only successful results; v4 audit, scoring, and exporter independently require successful provenance | `test_strict_audit_accepts_discrepancy_first_v4_trace`; v4 scorer/export tests |
| Directional Evidence is qualified | v3 adjudication used stance, directness, quality, risk, and binding metadata | `evidence_semantics.py` requires direct moderate/strong Evidence, no blocking risk, and coherent stance metadata | `test_discrepancy_decision_rejects_assessment_without_required_direction` |
| Reference-comparison direction is coherent | a different unedited capture was a binding/discovery lead, not standalone refutation | support requires an unedited same capture; refute requires edit evidence on the same capture; a likely different capture cannot carry either terminal direction | `test_discrepancy_decision_rejects_forged_reference_stance` |
| ClaimAssessment matches Evidence | v3 fact resolution retained support/refute direction and conflict semantics | supported requires support, refuted requires refute, conflicted requires both; all Evidence must be owned by a claim task | reducer, strict-audit, scoring, and exporter semantic-upgrade tests |
| Established decisive discrepancy is refuting | v3 fake closure required a refuted decisive fact | every affected v4 claim requires an owned qualified refute chain before an established decisive discrepancy is accepted | reducer fake test and v4 discrepancy-alignment audit/scoring tests |
| Finding links task, fact, and Evidence | v3 audit checked task/fact/Evidence ownership | v4 reducer requires a directional Finding for terminal semantics; v4 audit rechecks the complete graph and matching stance | `test_discrepancy_decision_requires_finding_evidence_chain` |
| Verdict basis has a complete chain | v3 required `VisualFact -> Finding -> Evidence -> successful call` | v4 basis compiler selects a qualified directional chain per selected claim; strict audit, scoring, and exporter reject missing links | `test_strict_audit_rejects_v4_verdict_without_finding_chain` |
| Terminal verdict matches semantic state | v3 Coverage compiled from the resolved core fact | fake requires decisive refutation; real requires supported high-salience claims and closed routes; unverifiable requires unresolved high-salience claims and closed routes | `test_v4_reducers.py` Coverage tests |
| Route/action termination is bounded | v3 audited the 24-action cap and post-verdict actions | v4 strict audit checks action-count parity, cap, terminal Coverage count, and no post-verdict action; protocol/route-control rejections remain strict failures | v4 strict-audit and historical replay tests |
| Bad traces cannot become training data | v3 scoring/export required valid chains and protocol state | v4 scoring adds decision-Evidence consistency and directional discrepancy alignment; policy export repeats the same fail-closed graph checks | `test_v4_process_scorer_excludes_neutral_evidence_semantic_upgrade`; `test_v4_policy_export_rejects_neutral_evidence_semantic_upgrade` |

## Historical replay review

- Queen, Pillars, and Monarch retain qualified directional Evidence and still replay
  deterministically to `fake`, `real`, and `fake`.
- The prior Andreea replay had manually labeled a different-capture comparison as
  `refute`. That label was not producible by the conservative visual Evidence reducer.
  The fixture now preserves `stance=neutral` and must be rejected as a fake verdict
  basis. A sanitized copy of the real-canary failure state lives in
  `test_fixtures/v4_semantic_safety_regressions.json`.

The Andreea case may return `fake` only after a future rollout records qualified
refuting Evidence with a complete Finding chain. Search Discovery, blocked pages,
model rationale, and different-capture comparisons cannot substitute for that chain.

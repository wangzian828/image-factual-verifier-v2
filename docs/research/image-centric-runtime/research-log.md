# Image-Centric Runtime Research Log

## 2026-07-17 — bootstrap

- Baseline commit: `5f4ac7e`.
- Structural diagnosis: original-image information is compressed before Target
  Planning and is unavailable to later policy stages.
- H1 is locked before implementation: attach the original image to every semantic
  and action-selection boundary without changing retrieval or evidence policy.
- H2 remains conditional: unify target revision, evidence assessment, replanning,
  and stopping only if H1 confirms that frozen target ownership is the remaining
  failure.


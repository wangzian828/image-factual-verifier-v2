# Image Factual Verifier v3

Runtime and evaluation system for auditable image factual verification. The active
development direction is the VisualFact-driven, image-only search agent described in:

- `docs/superpowers/specs/2026-07-14-visual-fact-search-agent-design.md`
- `docs/superpowers/plans/2026-07-14-visual-fact-search-agent.md`

The benchmark runtime now targets only the v0.3 image-only release contract. Manifest
parsing, three-field runtime cases, image hashing, private-gold isolation, and
classification-compatible predictions are active. VisualFact bootstrap, dynamic
tasks, Reflection checkpoints, and `reinspect-v2` Judgment are being introduced phase
by phase; image-only execution fails explicitly until those stages are active.

## Repository boundary

This project owns:

- Gemini Interactions agent runtime and tools;
- perception, investigation, evidence ledgers, coverage, and Judgment;
- canonical traces and trace auditing;
- benchmark release consumption and evaluator-private post-rollout joins;
- trajectory export after the runtime is stable.

Benchmark construction, acquisition, review, image generation, release finalization,
licenses, and source snapshots live in the separate
`image-factual-verifier-data-pipeline` project.

The repositories do not import each other. Their only integration boundary is the
immutable release layout documented in
`docs/runtime-release-contract.md`.

## Environment

Python 3.11 is required.

```powershell
python -m pip install -e ".[dev]"
$env:GEMINI_API_KEY = "..."
$env:SERPER_API_KEY = "..."
python -m src path\to\image.jpg
```

## Tests

Credential-free runtime contract:

```powershell
python -m pytest -q
```

Focused orchestration and release-consumer checks:

```powershell
python -m pytest -q `
  test_release_adapter.py `
  test_eval_artifacts.py `
  test_unit.py `
  test_evidence_grounding.py `
  test_failure_contracts.py `
  test_native_interactions.py `
  test_image_only_v2_trajectory.py
```

These are local contract and scripted-state tests, not proof of real system
availability. Real Gemini, search, browse, visual-tool, and trace acceptance uses:

```powershell
python scripts/run_real_canary.py `
  --benchmark path\to\v0.3-image-only-release\runtime_input\cases.jsonl `
  --output-dir path\to\new-canary-output
```

The canary refuses missing provider credentials, fake/scripted model names, engineering
errors, failed strict trace audits, and runs that do not exercise search, page visit,
and visual-observation tool classes. gpu-13 setup follows `docs/operations/gpu13.md`.

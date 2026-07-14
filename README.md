# Image Factual Verifier v3

Runtime and evaluation system for auditable image factual verification. The active
development direction is the VisualFact-driven, image-only search agent described in:

- `docs/superpowers/specs/2026-07-14-visual-fact-search-agent-design.md`
- `docs/superpowers/plans/2026-07-14-visual-fact-search-agent.md`

The runtime currently preserves the production `reinspect-v1` path while the
VisualFact data layer, dynamic tasks, Reflection checkpoints, and `reinspect-v2`
Judgment contract are introduced phase by phase.

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
  test_full_native_agent_trace.py
```

Real Gemini probes and gpu-13 validation follow `docs/operations/gpu13.md`.

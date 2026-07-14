# Image Factual Verifier v3

VisualFact-driven, image-only factual investigation runtime.

The supported flow is:

```text
ImageOnlyRuntimeCase
  -> Gemini perception + EasyOCR
  -> deterministic VisualFact/task bootstrap
  -> native Gemini Interactions ReAct
  -> Reflection every four real tool actions
  -> deterministic decisive-fact Coverage
  -> reinspect-v2 Judgment
  -> real | fake | unverifiable
```

`reinspect-v2` is the current v3 verdict-policy identifier. It is not support for an
older project version. Claim-mode inputs and `reinspect-v1` are unsupported.

## Repository boundary

This repository owns the Agent runtime, tools, canonical traces, strict trace audit,
post-rollout process scoring, policy-trajectory export, and dataset audit.

Benchmark acquisition, construction, review, release packaging, classification gold,
classification scoring, licenses, and source snapshots belong to the separate
`image-factual-verifier-data-pipeline` repository. The repositories communicate only
through the immutable v0.3 release contract in
`docs/runtime-release-contract.md`.

## Public input

Each runtime row contains exactly:

```json
{
  "case_id": "case_...",
  "image_path": "assets/sha256/...",
  "image_sha256": "..."
}
```

The runtime verifies the image hash before perception. Evaluator-private gold is not
loaded until all rollouts finish.

## Local setup

Python 3.11 is required.

```powershell
python -m pip install -e ".[dev]"
$env:GEMINI_API_KEY = "..."
$env:SERPER_API_KEY = "..."
$env:IMAGE_UPLOAD_PROVIDER = "oss"
$env:VISUAL_SEARCH_PROVIDER = "serper_lens"
$env:BROWSE_FETCH_PROVIDER = "jina"
python -m src path\to\image.jpg
```

Gemini perception receives the image through the Interactions API.
`ocr_with_position` is a separate EasyOCR observation; Codex does not manually inspect
benchmark images during runtime.

## Evaluation

```powershell
python -m src.eval.run_eval `
  --benchmark path\to\release\runtime_input\cases.jsonl `
  --output-dir path\to\new-run
```

The run writes:

- `predictions.jsonl`: successful classification rows only, exactly
  `case_id + verdict`;
- `run_results.jsonl`: diagnostics, costs, trace paths, and engineering errors;
- `process_metrics.jsonl`: deterministic per-case process metrics;
- `trajectory_scores.jsonl`: componentized teacher scores and diagnostics;
- `policy_trajectories.jsonl`: model-visible request/action examples;
- `summary.json`, `run_manifest.json`, and canonical `traces/*.json`.

An engineering failure never becomes factual `unverifiable`. It produces no
classification prediction, so the data-owned scorer counts that case as missing and
wrong.

## Validation

```powershell
python -m pytest -q
python scripts/audit_real_trace.py path\to\traces --json --strict-scheduler
```

The active suite contains 156 contract and scripted-state tests. They validate code
boundaries, not live provider availability.

No-mock acceptance requires:

```powershell
python scripts/run_real_canary.py `
  --benchmark path\to\release\runtime_input\cases.jsonl `
  --output-dir path\to\new-canary-output `
  --limit 2
```

The local image canary is currently blocked by the local proxy exit returning Gemini
HTTP 400: `This API is not available in your current location.` The required live
acceptance must be completed on gpu-13 using `docs/operations/gpu13.md`.

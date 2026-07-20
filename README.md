# Image Factual Verifier v4

Discrepancy-first, image-only factual investigation runtime.

```text
ImageOnlyRuntimeCase
  -> Gemini perception + positioned OCR
  -> deterministic visual facts and retrieval anchors
  -> Image Account Planning (1-3 ImageClaims + SearchHypotheses)
  -> account/hypothesis-owned native Gemini ReAct
  -> sparse multimodal Discrepancy Decision
  -> deterministic Coverage and claim/discrepancy/Evidence basis
  -> constrained discrepancy-first-v4 Judgment
  -> real | fake | unverifiable
```

The default decision policy is `discrepancy-first-v4`. The v3 runtime is frozen at
tag `runtime-v3-final-20260717`; legacy schemas may remain temporarily for read-only
trace replay but do not enter the v4 default path.

## Repository boundary

This repository owns the runtime, tools, canonical traces, strict audit,
post-rollout process scoring, and policy export. Benchmark construction, private gold,
classification scoring, licenses, and source snapshots belong to the separate data
pipeline repository and enter only through the immutable release contract.

Each public row contains exactly `case_id`, `image_path`, and `image_sha256`.
Evaluator-private gold is loaded only after every rollout.

## Local setup

Python 3.11 is required.

```powershell
python -m pip install -e ".[dev]"
$env:GEMINI_API_KEY = "..."
$env:SERPER_API_KEY = "..."
python -m src path\to\image.jpg
```

Gemini sees the original image in perception and once at the Image Account Planning
main-chain root. Later ReAct, Discrepancy Decision, and Judgment calls inherit the
visual context through `previous_interaction_id`.

Search titles, snippets, and reverse-image matches are Discovery only. Verdict
Evidence must preserve exact fetched text or a successful visual observation,
provenance, artifact hashes, and successful function-call ownership. Provider or
protocol failure is an engineering error, never `unverifiable`.

## Evaluation outputs

```powershell
python -m src.eval.run_eval `
  --benchmark path\to\release\runtime_input\cases.jsonl `
  --output-dir path\to\new-run
```

Outputs include predictions, run diagnostics, canonical traces, process metrics,
reference-chain metrics, v4 teacher scores, and `ifv-policy-v2` trajectories. The v4
training gate focuses on Evidence-chain recovery, discrepancy alignment, stop quality,
protocol validity, and absence of post-verdict actions.

## Validation and acceptance

```powershell
python -m pytest -q
python -m compileall -q src scripts
git diff --check
python scripts/audit_real_trace.py path\to\traces --json --strict-scheduler
```

Local tests are deterministic gates, not production acceptance. v4 completion also
requires replaying the four frozen historical traces, then one real Gemini canary
with manual trace review, followed by three to four heterogeneous canaries. Do not
run a full 20-case batch or Qwen trajectory production before those gates pass.

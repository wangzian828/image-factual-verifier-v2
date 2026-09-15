# PSD preparation log

## 2026-09-15 — E0 / code audit

User requested completion of PSD experiment preparation. Applied the autoresearch preparation/protocol workflow using the existing hourly heartbeat; no extra automation created. Inspected candidate routing, deterministic postprocessing, repair-source failure verification and semantic repair reviewer. Found the label-only source admission gap. No new tests or production PSD jobs have run before this protocol commit. Main SFT/Gemini jobs and immutable historical data remain untouched.

## E1 / source admission and causal regression

Implemented a separate private source semantic reviewer, literal-evidence and immutable request/artifact binding, pending-vs-fail separation, cached source review driver, candidate routing, source-audit canonical binding, and correct-label failure propagation through complete causal verification and offline finalization. Added pre-outcome plan/ID freezing and an offline real-source review auditor. Agent/prompt/tool implementation was not changed.

First test invocation lacked `PYTHONPATH=.:training`; collection failed and was rerun with the correct import roots. Initial focused suite: 51 pass, 1 expected-count assertion failed because the new pending queue was absent from the test expectation; corrected that assertion. Broader local collection lacked torch; ran dependency-free PSD tests locally and the complete PSD suite on the server CPU. No package or active training environment was upgraded.

## E2 / regression

Local dependency-free PSD suite progressed 202 -> 205 -> **208 passed** (final run 3.21 seconds). Server isolated code progressed 216 -> **219 passed** in 10.55 seconds, CUDA hidden and OMP threads limited to 2. JUnit saved on server. Three further deterministic postprocess integration controls were added and passed in the final local suite. `compileall` and `git diff --check` passed before commit.

## E3 / bounded real data

Reviewed all eight frozen real training canary traces from `psd-real-training-canary-20260912-r2` with `gemini-3.1-pro-preview`, concurrency 2, cached native images. Eight completed decisions: 2 pass / 6 fail; labels alone had 5 passes, with 3 of these now semantic repair candidates. No source trace, original result, image or old weight was modified. The offline audit passed. A no-network-client resume revalidated 16 cache/review artifacts, made zero provider calls, and preserved both negative and positive decisions. This is exploratory verifier/routing validation, not capability or judge-accuracy estimation.

## E4 / launch preparation and resource boundary

Created and revalidated the server plan and fixed 32 training-canary / 400 formal-test observation ID lists. No training-source holdout. Retained 128K, top20, batch32, existing SP4 fallback / DP4 throughput candidate, with real worst-length/save-load gates before selecting a production profile. Actual steps/duration and checkpoint disk size remain to be measured after target admission. Roughly 16 MB of isolated code/tests plus <0.4 MB of plan/reviews were added; no image/checkpoint copies. A whole-directory usage check lost its SSH connection; subsequent normal checks succeeded, so quota remains unknown, not assumed full or unlimited.

Existing hourly heartbeat now treats preparation as completed and retains only deferred launch-acceptance reminders alongside the original main pipelines. No new recurring task, GPU PSD job, production round or evaluation restart was created.

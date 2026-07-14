# VisualFact Search Agent v3 Implementation Plan

**Date:** 2026-07-14

**Updated:** 2026-07-15

**Status:** Runtime implementation complete; real gpu-13 acceptance pending

**Repository:** `D:\image-factual-verifier-v2-worktrees\visual-fact-search-agent`

**Branch:** `codex/image-factual-verifier-v3`

**Data repository:** `D:\image-factual-verifier-data-pipeline` (read-only from v3)

## 1. Goal

Build an image-only, single-agent factual investigation runtime:

```text
ImageOnlyRuntimeCase
  -> Gemini perception + EasyOCR
  -> deterministic VisualFact/task bootstrap
  -> native Interactions ReAct
  -> Reflection every four real actions
  -> decisive-fact Coverage
  -> reinspect-v2 Judgment
  -> real | fake | unverifiable
```

Required invariants:

- Discovery, Evidence, Finding, Failure, and verdict state remain separate.
- Every factual conclusion has exact source/tool-call provenance.
- Gold is inaccessible until rollout completes.
- Engineering failures never become factual `unverifiable`.
- Real-provider acceptance is required in addition to tests.

`reinspect-v2` is the current v3 verdict-policy identifier. It is not support for a v2
project or legacy runtime.

## 2. Repository boundary

### Data pipeline owns

- benchmark acquisition, construction, review, and release packaging;
- content-addressed images, manifest, public cases, private gold;
- acceptable evidence and process-reference protocol;
- classification protocol and classification scorer;
- licenses, checksums, source snapshots, and construction QA.

### v3 owns

- v0.3 release consumption and image hash validation;
- Gemini/EasyOCR perception;
- VisualFacts, tasks, Findings, Reflection, Coverage, and verdict basis;
- canonical traces and strict trace auditing;
- post-rollout process scoring and teacher diagnostics;
- policy trajectory export and split-safe dataset audit.

The repositories communicate only through the immutable release directory. They do
not import each other's Python modules.

## 3. Phase A — v0.3 release consumer

- [x] Require manifest-driven
  `ifv-image-only-benchmark-release-v0.3`.
- [x] Require `ifv-image-only-runtime-v1`,
  `input_mode=image_only`, and `decision_policy_version=reinspect-v2`.
- [x] Parse exact three-field `ImageOnlyRuntimeCase`.
- [x] Reject extra/missing fields and path escape.
- [x] Verify image SHA-256 before execution.
- [x] Resolve all companion artifacts from the manifest.
- [x] Support optional inactive/active source-access policy.
- [x] Keep private gold unread until all rollouts finish.
- [x] Record public artifact hashes before rollout and gold hash afterward.
- [x] Emit strict successful classification rows only.
- [x] Remove v0.2 and claim-mode fallback.

Acceptance: complete.

## 4. Phase B — VisualFact bootstrap

- [x] Add strict `InvestigationBrief`.
- [x] Add `VisualEntity`, `RetrievalAnchor`, and `VisualFact`.
- [x] Add bounded `ResearchTask`, `Finding`, Evidence, Discovery, and Failure schemas.
- [x] Run Gemini `perceive_scene` with the actual image.
- [x] Run EasyOCR `ocr_with_position`.
- [x] Preserve normalized image/OCR provenance.
- [x] Build stable deterministic IDs.
- [x] Create at most four initial evidence-routable tasks.
- [x] Activate up to three central routed decisive facts.
- [x] Run initial reverse-image search as Discovery only.
- [x] Persist bootstrap state before later provider failures.

Real perception was completed locally for both development images. Re-feeding those
outputs through the current bootstrap produced:

- Artemis: a decisive scene/event/location proposition rather than only “four people”;
- Bigelow: decisive vessel-identity facts tied to visible NOAA/name/registration
  markings.

Acceptance: implementation complete; final live end-to-end acceptance remains Phase E.

## 5. Phase C — Dynamic tasks and Reflection

- [x] Implement deterministic fact/task/Finding/Failure reducer.
- [x] Keep task/fact history append-only through stable IDs.
- [x] Resolve tasks only through qualified Findings.
- [x] Block/exhaust tasks only through recorded failures.
- [x] Run Reflection at cumulative actions 4, 8, 12, 16, 20, and 24.
- [x] Count successful, failed, blocked, and empty real tool calls as actions.
- [x] Exclude model output, Reflection, Coverage, Judgment, and rejected calls.
- [x] Allow bounded task updates, new tasks, decisive-fact proposals, and recommendations.
- [x] Reject Evidence/Finding/verdict creation by Reflection.
- [x] Enforce:

  ```text
  MAX_TOOL_ACTIONS = 24
  REFLECTION_INTERVAL = 4
  MAX_REFLECTIONS = 6
  INITIAL_TASKS_MAX = 4
  TOTAL_TASKS_MAX = 12
  NEW_TASKS_PER_REFLECTION_MAX = 3
  DECISIVE_FACTS_MAX = 6
  NEW_DECISIVE_FACTS_PER_REFLECTION_MAX = 2
  ```

- [x] Treat two consecutive invalid Reflections as engineering failure.
- [x] Remove the old fixed-question replanning runtime.

Acceptance: implementation complete.

## 6. Phase D — Coverage and Judgment

- [x] Bind every Finding to task, fact, Evidence, source family, and successful call.
- [x] Preserve exact page span, offsets, URL, artifact hash, retrieval time, stance,
  directness, risk, and access-policy checks.
- [x] Keep Discovery out of Findings and verdict basis.
- [x] Qualify official/visual evidence or independent source-family corroboration.
- [x] Audit decisive facts as supported, refuted, conflicted, blocked, exhausted, or
  unresolved.
- [x] Exclude Discovery/task churn from substantive gain.
- [x] Stop only on coverage, two low-gain Reflection intervals, or 24 actions.
- [x] Compile deterministic verdict and exact `VerdictBasis`.
- [x] Enforce:

  ```text
  any decisive refuted fact -> fake
  all decisive facts supported -> real
  otherwise -> unverifiable
  ```

- [x] Require
  `VisualFact -> Finding -> Evidence -> successful tool call`.
- [x] Require Gemini Judgment to match verdict, policy, ID sets, and gaps exactly.
- [x] Render VisualFacts, tasks, Findings/Evidence, Reflection, Coverage, and basis.
- [x] Strictly audit IDs, references, action budget, Reflection cadence, interaction
  chains, private leakage, Discovery misuse, and verdict state.
- [x] Fail as engineering error when all investigation tools fail.
- [x] Remove claim-driven canonical state and legacy trace auditing.

Acceptance: implementation complete.

## 7. Phase E — real group-001 acceptance

Required release:

```text
D:\image-factual-verifier-data-pipeline\data\benchmarks\releases\image_only\
image-only-benchmark-group-001
```

Required no-mock command:

```powershell
$release = "D:\image-factual-verifier-data-pipeline\data\benchmarks\releases\image_only\image-only-benchmark-group-001"
$run = "D:\image-factual-verifier-runs\group-001-v3-<new-id>"

python scripts/run_real_canary.py `
  --benchmark "$release\runtime_input\cases.jsonl" `
  --output-dir $run `
  --model gemini-3.5-flash `
  --limit 2
```

Then:

```powershell
python scripts/audit_real_trace.py "$run\traces" --json --strict-scheduler
```

And run the data-owned classification scorer against `predictions.jsonl`.

### Current real status

- [x] Real Gemini text Interactions probe completed:
  structured output, native function call, and function-result continuation.
- [x] Real Gemini perception was previously obtained for both images.
- [x] Local canary preserves provider failures as engineering errors.
- [x] Failed cases produce no classification prediction.
- [x] Data scorer counts missing predictions as wrong.
- [ ] Both current-code cases complete without engineering error.
- [ ] Both predictions are correct.
- [ ] Both strict trace audits pass.
- [ ] Required search, reverse-image/upload, visit, and visual tool classes succeed.
- [ ] Canonical traces are manually inspected.

Local failed runs:

```text
D:\image-factual-verifier-runs\group-001-v3-canary-20260715-01
D:\image-factual-verifier-runs\group-001-v3-canary-20260715-02
```

The second run failed during image perception with:

```text
HTTP 400: This API is not available in your current location.
```

This is a local proxy-exit restriction, not a factual Agent result. The trace correctly
records an engineering error, `predictions.jsonl` is empty, and classification scoring
returns zero rather than accepting fake `unverifiable` rows.

### gpu-13 completion command

After pushing the branch and securely authenticating the existing Jupyter endpoint:

```bash
cd /gs/home/wza/projects/image-factual-verifier-v2
bash scripts/server/update_gpu13_checkout.sh codex/image-factual-verifier-v3
bash scripts/server/bootstrap_gpu13.sh
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q
```

The release must exist under an approved gpu-13 data path. Then:

```bash
run_id="group-001-v3-$(date -u +%Y%m%dT%H%M%SZ)"
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python scripts/run_real_canary.py \
  --benchmark "<gpu13-release>/runtime_input/cases.jsonl" \
  --output-dir "$IFV_DATA_ROOT/runs/eval/$run_id" \
  --model gemini-3.5-flash \
  --limit 2

scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python scripts/audit_real_trace.py \
  "$IFV_DATA_ROOT/runs/eval/$run_id/traces" \
  --json --strict-scheduler
```

Phase E is the only runtime acceptance item still open.

## 8. Phase F — process evaluation

- [x] Implement decisive-fact matching and status alignment.
- [x] Implement acceptable-evidence hit rate.
- [x] Implement citation precision.
- [x] Implement evidence-to-vision bridge completion.
- [x] Implement verdict-basis alignment.
- [x] Report unmatched qualified Evidence for review.
- [x] Report valid-Finding precision, invalid tasks/activations, duplicate actions,
  premature finish, cost, latency, and first error.
- [x] Emit separate `process_metrics.jsonl` and `trajectory_scores.jsonl`.
- [x] Keep classification and process scoring separate.
- [x] Include process protocol and private gold SHA-256 in score metadata.
- [x] Include concrete fact/basis/evidence diagnostics in teacher scores.
- [x] Score missing/engineering-error traces explicitly.

Acceptance: implementation complete. Real group-001 metric acceptance follows Phase E.

## 9. Phase G — trajectories and training preparation

### G1. Versioned policy trajectories

- [x] Export actual ReAct, Reflection, and Judgment policy boundaries.
- [x] Preserve native function-call and function-result boundaries.
- [x] Persist model-visible `policy_input` and `policy_action` snapshots.
- [x] Record source run, runtime commit, release ID, and protocol versions.
- [x] Add tokenizer-aligned input/action IDs.
- [x] Add action loss masks.
- [x] Zero-mask invalid and fatal-boundary actions.
- [x] Exclude evaluator-private data and scorer annotations.
- [x] Keep model tokenizer injectable.
- [x] Use stable `utf8-byte-v1` only as the default adapter.
- [x] Do not fabricate a Planning target; bootstrap is deterministic.
- [x] Reserve `planning` in the schema for a future real policy stage.

### G2. Dataset export and audit

- [x] Export train/validation/test JSONL.
- [x] Keep an episode in one split.
- [x] Keep shared source families in one split.
- [x] Audit private leakage, runtime references, masks, duplicate step IDs, and split
  isolation.
- [x] Record componentized teacher score and episode metadata.
- [x] Add `configs/training/ifv_policy_v1.yaml`.
- [x] Keep `training_enabled: false`.
- [ ] Freeze a real teacher dataset from accepted live traces.
- [ ] Demonstrate stable real canaries beyond group-001.

The exporter/auditor implementation is complete. Dataset freeze remains blocked by
Phase E and is not replaced by scripted trajectories.

### G3. Student training

Student training is outside this runtime plan. Do not start it until real traces are
accepted and a separate training plan fixes model, tokenizer, hardware, trainer, and
evaluation baselines.

## 10. Validation policy

Validation layers are independent:

```text
contract tests
!= scripted trajectory
!= live provider protocol
!= factual end-to-end acceptance
```

Active deterministic checks:

```powershell
python -m pytest -q
python -m compileall -q src scripts
git diff --check
```

Current active suite:

```text
156 passed
```

This suite intentionally removed the old claim-ledger, fixed-replanning, and
claim-mode tests. It retains v3 release, perception, Gemini protocol, tool/provider
failure, source-policy, VisualFact state machine, scripted trajectory, strict trace,
process scoring, trajectory export, and dataset audit coverage.

## 11. Implemented commits

```text
7726f1f feat: consume image-only v0.3 releases
37c06de feat: run image-only VisualFact investigations
a8e86ea docs: document gpu13 Jupyter control client
acc4e4f feat: audit image-only VisualFact traces
5b60215 test: enforce image-only real canary
7e081b1 feat: export versioned policy trajectories
21e9313 feat: score investigation process trajectories
0c78763 feat: export and audit policy datasets
3fb775a chore: retire claim-driven scripted acceptance
2e64f50 feat: enrich process score provenance
36c6036 refactor: remove claim-driven runtime
```

## 12. Final completion checklist

- [x] v0.3 three-field input and private-gold isolation.
- [x] Gemini image perception and positioned OCR.
- [x] Image-grounded VisualFacts and bounded tasks.
- [x] One native tool call per action turn.
- [x] Reflection at cumulative actions 4, 8, 12, 16, 20, 24.
- [x] Discovery never directly becomes verdict Evidence.
- [x] Deterministic decisive-fact Coverage and verdict basis.
- [x] Engineering errors produce no classification prediction.
- [x] Process scoring, policy trajectories, and dataset audit.
- [x] Claim-driven runtime and redundant legacy tests removed.
- [ ] gpu-13 real two-case canary passes.
- [ ] Real teacher dataset is frozen from accepted traces.

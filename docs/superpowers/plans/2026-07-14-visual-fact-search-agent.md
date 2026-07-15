# VisualFact Search Agent v3 Implementation Plan

**Date:** 2026-07-14

**Updated:** 2026-07-15

**Status:** Runtime baseline and trajectory-quality remediation complete

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
- [x] Preserve both Lens and semantic-search reference-image URLs.
- [x] Expose untested official reference images to ReAct without turning them into
  Evidence.
- [x] Persist bootstrap state before later provider failures.

Real perception was completed locally for both development images. Re-feeding those
outputs through the current bootstrap produced:

- Artemis: a decisive scene/event/location proposition rather than only “four people”;
- Bigelow: decisive vessel-identity facts tied to visible NOAA/name/registration
  markings.

Acceptance: complete, including live end-to-end acceptance in Phase E.

## 5. Phase C — Dynamic tasks and Reflection

- [x] Implement deterministic fact/task/Finding/Failure reducer.
- [x] Keep task/fact history append-only through stable IDs.
- [x] Resolve tasks only through qualified Findings.
- [x] Keep a task active when its Finding is not yet strong enough to resolve an owned
  decisive fact.
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
- [x] Require same-capture/near-duplicate comparison before a reference image can
  support a scene-provenance fact; same subject in a different capture remains
  neutral for that fact.
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
- [x] Both committed-head cases complete without engineering error.
- [x] Both predictions are correct.
- [x] Both strict trace audits pass.
- [x] Required search, reverse-image/upload, visit, and visual tool classes succeed.
- [x] Canonical traces are manually inspected.

Accepted committed-head run:

```text
run:
D:\image-factual-verifier-runs\group-001-v3-canary-20260715-19

runtime commit:
fd305d712626a1c189433fe37a1e286f30238365

predictions:
case_509704a5a1034b2c -> fake
case_58720a90e3ef438a -> real

classification:
2/2 correct
Accuracy = 1.0
Macro-F1 over observed classes = 1.0
Fixed three-class Macro-F1 = 0.666667 because this development subset has no
unverifiable case

trace audit:
2 passed
0 scheduler rejections
0 protocol rejections
thought tokens = 0
```

The fake case reached `wrong_place` using direct NASA evidence that the ceremony was
at Johnson Space Center rather than Kennedy Space Center. The real case used actual
`compare_with_reference` calls and found a same-capture/near-duplicate reference for
NOAA Ship Henry B. Bigelow.

Accepted gpu-13 replication:

```text
checkout:
/gs/home/wza/projects/image-factual-verifier-v3

runtime commit:
abb7db553cd4d3e8046faed3c43dac3dce67e328

run:
/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/
group-001-v3-gpu13-20260715-01

predictions:
case_509704a5a1034b2c -> fake
case_58720a90e3ef438a -> real

server tests:
165 passed

trace audit:
2 passed
0 scheduler rejections
0 protocol rejections
thought tokens = 0
```

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

### gpu-13 accepted replication

After pushing the branch and securely authenticating the existing Jupyter endpoint:

```bash
cd /gs/home/wza/projects/image-factual-verifier-v3
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

Phase E is complete locally and on gpu-13. The server release was downloaded by
gpu-13 through temporary object storage rather than through the SSH control endpoint;
the transfer object was deleted after archive and release SHA-256 verification.

## 8. Phase F — process evaluation

- [x] Implement decisive-fact matching and status alignment.
- [x] Separate reference-chain recovery from general process scoring.
- [x] Implement evidence-to-vision bridge completion.
- [x] Implement verdict-basis alignment.
- [x] Keep unrelated Evidence neutral unless it enters the verdict basis.
- [x] Report valid-Finding precision, invalid tasks/activations, duplicate actions,
  premature finish, cost, latency, and first error.
- [x] Emit separate `process_metrics.jsonl` and `trajectory_scores.jsonl`.
- [x] Keep classification and process scoring separate.
- [x] Include process protocol and private gold SHA-256 in score metadata.
- [x] Include concrete fact/basis/evidence diagnostics in teacher scores.
- [x] Score missing/engineering-error traces explicitly.

Acceptance: complete. The accepted run emitted process metrics and componentized
teacher scores for both cases. Snapshot-identity metrics were subsequently removed
from runtime scoring; URL, span, artifact, and SHA identity remain data-pipeline audit
properties.

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
- [x] Freeze and strictly audit a real teacher dataset from accepted live traces.
- [ ] Demonstrate stable real canaries beyond group-001.

Accepted dataset:

```text
D:\image-factual-verifier-runs\group-001-v3-canary-20260715-19\policy_dataset

/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/
group-001-v3-gpu13-20260715-01/policy_dataset
```

It contains 48 examples from two real episodes:

```text
react = 38
reflection = 8
judgment = 2
invalid actions = 0
fatal boundaries = 0
strict audit errors = 0
local source runtime commit = fd305d712626a1c189433fe37a1e286f30238365
gpu-13 source runtime commit = abb7db553cd4d3e8046faed3c43dac3dce67e328
```

Because group-001 is one source-family group and only two cases, this artifact proves
the export/audit path; it is not yet a statistically useful Student training corpus.

### G3. Student training

Student training is outside this runtime plan. Do not start it until real traces are
accepted and a separate training plan fixes model, tokenizer, hardware, trainer, and
evaluation baselines.

The separate implementation plan is:

```text
docs/superpowers/plans/2026-07-15-qwen-vl-student-training-infrastructure.md
```

It freezes Gemini as the teacher/regression baseline, makes Qwen-VL the eventual
perception and policy student, isolates training from the runtime environment, and
defines the SFT/GRPO, checkpoint, serving, and dual-provider compatibility contracts.

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
175 passed
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
11a097a docs: finalize v3 runtime contract
fd305d7 fix: validate real image-only investigation routes
ca74e7e fix: improve image-only trajectory quality
52b8454 fix: finalize image-only segments deterministically
459a6f2 fix: make investigation boundaries deterministic
55db562 fix: advance image-only reflection boundaries
86d73c0 fix: trust exact official image provenance
c20948d fix: separate route control from protocol failures
```

## 12. Final completion checklist

- [x] v0.3 three-field input and private-gold isolation.
- [x] Gemini image perception and positioned OCR.
- [x] Image-grounded VisualFacts and bounded tasks.
- [x] One native tool call per action turn.
- [x] Reflection at cumulative actions 4, 8, 12, 16, 20, 24 unless a
  deterministically complete verdict terminates on that action.
- [x] Discovery never directly becomes verdict Evidence.
- [x] Deterministic decisive-fact Coverage and verdict basis.
- [x] Engineering errors produce no classification prediction.
- [x] Process scoring, policy trajectories, and dataset audit.
- [x] Claim-driven runtime and redundant legacy tests removed.
- [x] Committed-head real two-case canary passes.
- [x] Real teacher dataset is frozen from accepted traces.
- [x] Replicate the accepted commit on gpu-13 through the authenticated Jupyter
  control path.
- [x] Adjudicate qualified support/refute conflicts before final verdict.
- [x] Require exact visual binding for positive scene support.
- [x] Reject semantic duplicate routes across ReAct segments.
- [x] Compile minimal sufficient verdict bases.
- [x] Filter teacher episodes through explicit trajectory-quality gates.
- [x] Complete gpu-13 Phase H canary, classification, process scoring, and dataset
  audit.
- [x] Add evaluator-private reference-chain recovery without changing classification
  or training gates.

## 13. Phase H - trajectory-quality remediation

The two-case baseline proves that the runtime, provider protocol, classification
path, trace audit, and dataset export execute end to end. It does not prove that the
resulting trajectories are efficient, visually grounded, or suitable as teacher
data.

Manual review of the accepted local and gpu-13 traces found:

- a decisive scene refutation did not stop later low-value work;
- generic pages about a landmark, logo, or entity could support a fact about what is
  present in the input pixels;
- qualified support and refute evidence became a terminal `conflicted` status, which
  later collapsed to `unverifiable` without source-quality adjudication;
- duplicate detection compared exact tool arguments rather than semantic routes;
- failed reference-image downloads did not try a page/image fallback chain;
- verdict basis compilation retained weak and redundant evidence rather than a
  minimal sufficient subset;
- `evidence_to_vision_bridge_completion` measured structural linkage rather than an
  actual pixel/reference bridge;
- every successful episode was exportable as teacher data regardless of trajectory
  quality.

### H1. Research-derived design constraints

The remediation follows mechanisms shared by mature research agents and primary
literature:

- ReAct and WebGPT keep search actions and cited evidence explicit rather than
  treating model memory or snippets as proof.
- RARR and ALCE separate attribution correctness from answer correctness and require
  evidence to entail the exact supported statement.
- Self-RAG and CRAG use explicit critique/correction states when retrieved evidence is
  insufficient or unreliable.
- DeepResearcher trains against real web interaction rather than retrieval-only
  simulation.
- Deep Research Bench evaluates both the final answer and the research process,
  including citation correctness, completeness, source quality, and trace behavior.
- OpenAI deep research and Anthropic's research system both expose sources and retain
  a dedicated citation/verification pass instead of hiding evidence selection inside
  final prose.

These references motivate the implementation but do not replace real image-only
acceptance.

### H2. Correct evidence-conflict semantics

`conflicted` is an intermediate investigation state, not a verdict rule:

```text
qualified support + qualified refute
  -> conflict_resolution_required
  -> compare exact claim/scene binding, source originality, directness,
     independence, source risk, and temporal/event/place scope
  -> support wins -> supported
  -> refute wins -> refuted
  -> no discriminating evidence after bounded search -> evidence insufficient
  -> unverifiable
```

Evidence conflict itself must never be described as “unverifiable.” The final
`unverifiable` outcome means that the available qualified evidence is insufficient to
resolve the decisive proposition.

### H3. Implementation tasks

- [x] Add deterministic support/refute assessment with source and visual-binding
  strength.
- [x] Preserve conflict diagnostics and resolve a conflict when one direction has
  materially stronger, more direct, or better-bound evidence.
- [x] Require a real pixel/reference bridge before web evidence can support a
  proposition about what the input image contains.
- [x] Keep direct contradiction of an exact decisive event/place/identity slot
  eligible to refute that proposition.
- [x] Activate the smallest central decisive-fact set; do not make incidental logos,
  objects, or landmarks mandatory when a scene proposition already subsumes them.
- [x] Stop when the verdict is deterministically established after conflict
  adjudication; do not wait for unrelated supporting facts.
- [x] Block calls for tasks that became resolved during the current ReAct segment.
- [x] Canonicalize semantic routes for search queries, URLs, reference images, crops,
  and task targets; reject meaning-equivalent repeats.
- [x] Add browser-like image download headers, redirect validation, source-page image
  extraction, and URL-variant fallback for reference comparison.
- [x] Compile a minimal sufficient verdict basis from the strongest winning evidence.
- [x] Replace structural bridge scoring with actual same-capture or
  pixel-plus-source binding.
- [x] Add conflict-resolution, wasted-route, basis-minimality, and semantic-duplicate
  diagnostics.
- [x] Mark episodes training-eligible only when explicit trajectory-quality gates
  pass; preserve excluded episodes and reasons in dataset metadata.

### H4. Acceptance

Deterministic validation:

```powershell
python -m pytest -q
python -m compileall -q src scripts
git diff --check
```

Real acceptance must then rerun both group-001 cases and manually verify:

- correct `fake` and `real` classifications;
- no generic web page independently supports a pixel-presence fact;
- any support/refute conflict has an explicit adjudication reason;
- the fake case stops after the decisive refutation is established and adjudicated;
- no semantic duplicate route is executed;
- reference comparison either succeeds through a recorded fallback or records all
  attempted access paths without retrying the same route;
- every verdict basis is a minimal sufficient fact/Finding/Evidence chain;
- bridge and basis metrics agree with manual trace inspection;
- only quality-gate-passing episodes enter the policy dataset.

The same committed head must pass on gpu-13. Phase H is incomplete until both traces
are manually inspected after that run.

### H5. Accepted result

Accepted runtime commit:

```text
c20948d8dc4230c45e4c2f25707e0c52fc31bd80
```

Accepted gpu-13 run:

```text
/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/
group-001-v3-quality-20260715-07
```

Result:

```text
case_509704a5a1034b2c -> fake
case_58720a90e3ef438a -> real
Accuracy = 1.0
observed-class Macro-F1 = 1.0
fixed three-class Macro-F1 = 0.666667
engineering errors = 0
strict trace audit = 2/2
scheduler rejections = 0
protocol rejections = 0
route-control rejections = 0
```

Manual trajectory review:

- the fake case used 7 actions, stopped immediately after direct NASA evidence
  established Johnson Space Center, and compiled one fact, one Finding, and one
  official direct Evidence record as the basis;
- the real case used 3 actions, stopped on an exact same-capture NOAA official image,
  and compiled one fact, one Finding, and one reference-comparison Evidence record;
- neither case executed a semantic duplicate, continued after verdict determination,
  or used `current_time`;
- actual visual bridge completion, verdict-basis alignment, and basis minimality are
  all `1.0` for both cases;
- both cases pass the training-quality gate.

The v2 policy dataset export contains 2 episodes and 11 examples:

```text
react = 8
reflection = 1
judgment = 2
excluded episodes = 0
strict dataset audit errors = 0
```

Phase H acceptance: complete.

## 14. Phase I - evaluator-private reference-chain recovery

The original acceptable-evidence diagnostic required one runtime Evidence object to
match the frozen canonical URL, span, stance, source family, and artifact SHA-256.
Those fields are useful for data construction and release auditing, but they do not
measure whether the Agent recovered the intended factual evidence chain. Runtime
scoring therefore does not expose a separate snapshot-reproduction metric.

Phase I adds a separate evaluator artifact rather than changing classification,
teacher eligibility, or the data-pipeline contract:

```text
reference_chain_metrics.jsonl
```

It is deliberately limited to four metrics:

```text
fact_recovery_recall
chain_recovery_recall
evidence_recovery_recall
basis_reference_precision
```

The evaluated scope is only:

```text
visual_anchor
  -> decisive_fact
  -> expected_status
  -> acceptable_evidence
  -> source artifact
```

Semantic matching first applies conservative deterministic rules for a same-source
page/span version and an official same-capture image asset. An optional LLM judge may
inspect only qualified, direct, risk-free Evidence edges that remain unresolved. The
judge receives the already selected gold fact and acceptable references; it cannot
invent a new evaluation target. Unrelated evidence outside the final verdict basis is
neither rewarded nor penalized. Off-chain basis evidence lowers only
`basis_reference_precision`.

Implementation:

- [x] Emit deterministic `reference_chain_metrics.jsonl` from every new evaluation.
- [x] Add a standalone scorer for existing runs.
- [x] Remove URL/span/snapshot/SHA reproduction from runtime quality metrics.
- [x] Remove the old strict acceptable-evidence and citation fields from process
  scoring.
- [x] Add bounded optional LLM matching for unresolved qualified edges.
- [x] Keep the artifact out of classification overrides and training-admission gates.
- [x] Verify that off-chain evidence is neutral until selected into the verdict basis.
- [x] Replay both accepted group-001 traces on gpu-13.

Accepted implementation commits:

```text
b4f9641 feat: score frozen reference-chain recovery
eb5842a fix: recover reference spans from longer source text
65a5507 refactor: keep only reference-chain recovery metrics
```

Validation:

```text
local tests = 179 passed
gpu-13 tests = 179 passed
```

Accepted replay:

```text
/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/
group-001-v3-quality-20260715-07/reference_chain_metrics.jsonl
```

Both cases produced:

```text
fact_recovery_recall = 1.0
chain_recovery_recall = 1.0
evidence_recovery_recall = 1.0
basis_reference_precision = 1.0
```

The Artemis trace recovered the frozen NASA chain through a same-source official page
version with a longer body span. The NOAA trace recovered it through the official
same-capture image asset. Neither required an LLM call, which keeps obvious matches
deterministic; the LLM fallback remains available for genuinely unresolved qualified
alternatives.

Phase I acceptance: complete.

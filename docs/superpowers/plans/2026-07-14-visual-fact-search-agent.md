# VisualFact Search Agent v3 Implementation Plan

**Date:** 2026-07-14

**Status:** Active; Phase A consumer boundary complete, Phase B is next

**Runtime repository:** `D:\image-factual-verifier-v2-worktrees\visual-fact-search-agent`

**Runtime branch:** `codex/image-factual-verifier-v3`

**Data repository:** `D:\image-factual-verifier-data-pipeline`

## 1. Goal

Turn the current claim-driven verifier into an image-only, single-agent factual
investigation runtime while preserving:

- Gemini Interactions native function calling;
- strict separation of Discovery, Evidence, Finding, and verdict state;
- exact source and tool-call provenance;
- evidence-conditioned visual reinspection;
- deterministic coverage and judgment validation;
- canonical redacted traces;
- explicit engineering failure instead of disguising failures as `unverifiable`.

The target image-only flow is:

```text
ImageOnlyRuntimeCase
  -> InvestigationBrief
  -> bootstrap perception and retrieval anchors
  -> candidate VisualFacts
  -> bounded ResearchTasks
  -> ReAct tool use
  -> Finding and fact-state reduction
  -> Reflection every four real tool actions
  -> decisive-fact Coverage
  -> reinspect-v2 LedgerJudgment
  -> real | fake | unverifiable + verdict_basis
```

The approved architecture remains
`docs/superpowers/specs/2026-07-14-visual-fact-search-agent-design.md`.
This document is the current execution plan and supersedes the old phase ordering
that treated image-only as a dormant compatibility layer.

## 2. Current Baseline

The following work is complete on `codex/image-factual-verifier-v3`:

- [x] `8e084f8`: exhausted ReInspect dependencies remain unresolved.
- [x] `d5f8520`: raw discoveries do not count as substantive progress.
- [x] `a842972`: lead-only investigation iterations saturate.
- [x] `1b4b49c`: isolated benchmark release consumer boundary.
- [x] `8aac777`: benchmark construction code removed from the runtime repository.
- [x] `166c640`: empty private tool-runtime metrics no longer leak into tool output.
- [x] `88f2ad7`: contract/scripted/real validation split.
- [x] Fake smoke runners were removed.
- [x] The controlled end-to-end test is explicitly named
  `test_scripted_agent_trajectory.py`.
- [x] `scripts/run_real_canary.py` exists and fails closed on missing real-provider
  configuration.
- [x] Pre-migration credential-free baseline: `212 passed`.
- [x] Phase A v0.3-only baseline after removing v0.2 fixtures/tests:
  `208 passed`.

The local machine does not currently have the complete
Gemini/search/browse/upload provider configuration, so a real canary has not yet
passed. `208 passed` is a deterministic baseline, not evidence that the live system
works.

## 3. Repository Boundary and Shared Interface

The two repositories now develop independently. They communicate through an immutable
release directory and must not import each other's Python modules.

### 3.1 Data-pipeline ownership

The data project owns:

- benchmark construction and review;
- release images and content hashes;
- `manifest.json`;
- `runtime_input/cases.jsonl`;
- optional source-access policy;
- evaluator-private factual gold and acceptable evidence;
- classification protocol and classification scorer;
- release checksums and licenses.

It does not generate runtime `VisualFact`, `ResearchTask`, `Finding`, Reflection,
tool sequence, actual verdict basis, trajectories, or training masks.

### 3.2 v3 runtime ownership

The runtime project owns:

- release detection and public-row parsing;
- image hash verification and release-relative path resolution;
- Agent rollout and canonical trace persistence;
- runtime VisualFacts, tasks, Findings, Reflection, Coverage, and verdict basis;
- process scoring against evaluator-private references after rollout;
- trajectory export and later training-data derivation.

### 3.3 Active release contract

v3 consumes only the v0.3 image-only release:

| Public input | Runtime model | Policy | Status |
|---|---|---|---|
| `case_id`, `image_path`, `image_sha256` only | `ImageOnlyRuntimeCase` | `reinspect-v2` from manifest | consumer active; Agent bootstrap next |

The v0.3 public row is exactly:

```json
{
  "case_id": "case_...",
  "image_path": "assets/sha256/...",
  "image_sha256": "..."
}
```

It must not contain `claim_mode`, `user_claim`, empty claim placeholders, labels,
construction metadata, acceptable evidence, or runtime investigation state.

Mode and policy are release-level properties:

```json
{
  "schema_version": "ifv-image-only-benchmark-release-v0.3",
  "runtime_contract_version": "ifv-image-only-runtime-v1",
  "input_mode": "image_only",
  "decision_policy_version": "reinspect-v2"
}
```

The current development release is:

```text
D:\image-factual-verifier-data-pipeline\data\benchmarks\releases\image_only\
image-only-benchmark-group-001
```

Its public entrypoint is:

```text
runtime_input/cases.jsonl
```

Its evaluator-private reference is:

```text
evaluator_private/gold.jsonl
```

Its public protocols are:

```text
evaluation/classification_protocol.json
evaluation/process_reference_protocol.json
```

`source_access_policy` is optional in v0.3. The release must not fail merely because
the policy is absent when `manifest.json` declares it inactive.

The release manifest and protocol files are the canonical machine-readable contract.
Do not require the data repository to maintain another duplicate v0.3 contract tree.
The runtime may keep a small consumer regression fixture, but it must not vendor the
complete benchmark or copy construction code.

## 4. Validation Policy

Validation has three distinct layers:

1. **Contract tests** check schemas, joins, fail-closed behavior, and deterministic
   invariants.
2. **Scripted trajectory tests** check state transitions using controlled provider and
   tool responses.
3. **Real canaries** check the actual Gemini, search, upload/reverse-image, browse,
   visual-tool, trace, and scoring path.

Passing one layer does not imply passing the next:

```text
pytest green != provider protocol works != factual runtime works
```

Test additions must be proportional to risk. Prefer a small number of tests at stable
boundaries:

- one exact public-input contract test;
- one post-rollout private-data isolation test;
- focused reducer/validator tests for deterministic invariants;
- one scripted trajectory per materially different state-machine path;
- real group-001 execution as early as the runtime can accept it.

Do not create broad duplicate mock suites for every helper. Do not report a phase as
accepted until its required real canary has run.

## 5. Phase A — Consume the v0.3 Image-Only Release

**Outcome:** v3 can consume the real group-001 release without inventing claim fields
or loading private gold before rollout. Until reinspect-v2 behavior is active, the
runtime stops at an explicit engineering boundary and never projects image-only into
an embedded claim.

### A1. Add manifest-driven release detection

- [x] Replace row-shape-only release detection with a release descriptor loaded from
  the release root.
- [x] Validate `schema_version`, `runtime_contract_version`, `input_mode`, and
  `decision_policy_version`.
- [x] Resolve artifact paths from `manifest["artifacts"]`; do not hard-code
  companion locations.
- [x] Reject unknown or internally inconsistent release contracts.

Primary files:

```text
src/eval/release_adapter.py
src/eval/run_eval.py
docs/runtime-release-contract.md
test_release_adapter.py
test_eval_artifacts.py
```

### A2. Introduce the image-only runtime input model

- [x] Add strict `ImageOnlyRuntimeCase` with exactly `case_id`, `image_path`, and
  `image_sha256`.
- [x] Use an explicit runtime-input union or release descriptor; do not add nullable
  claim fields to `ImageOnlyRuntimeCase`.
- [x] Verify the resolved image SHA-256 before execution.
- [ ] Ensure canonical traces identify `input_mode=image_only` and
  `decision_policy_version=reinspect-v2` without fabricating a claim.

Primary files:

```text
src/orchestrator/state.py
src/orchestrator/ledger.py
src/workflow.py
src/eval/release_adapter.py
```

### A3. Correct release companions and rollout isolation

- [x] Treat `evaluator_private/gold.jsonl` as the private join keyed by
  `case_id`.
- [x] Remove dependencies on `evaluator_private/run_eval.jsonl` and
  `evaluation_gold/gold.jsonl`.
- [x] Load evaluator-private gold only after all Agent rollouts finish.
- [x] Load source-access policy before retrieval only when the manifest declares an
  active policy or the caller explicitly supplies one.
- [x] Record the manifest, public input, optional policy, and post-rollout private
  artifact hashes in `run_manifest.json`.

### A4. Produce classification-compatible predictions

The classification scorer accepts rows joined by `case_id` with verdict values
`real`, `fake`, or `unverifiable`. Optional runtime fields may remain.

- [x] Emit at least:

  ```json
  {"case_id": "case_...", "verdict": "fake"}
  ```

- [x] Keep confidence, verdict basis, and trace path additive.
- [x] Do not expose gold, factual status, decisive facts, or acceptable evidence in
  `predictions.jsonl`.
- [x] Do not duplicate the data project's classification scorer in v3.

### A5. Minimal validation

- [x] Add one compact v0.3 consumer fixture shaped from the finalized release
  contract.
- [x] Test exact three-field public rows and forbidden private fields.
- [x] Test optional policy behavior.
- [x] Test that private gold is first read after rollout.
- [x] Run v3 directly against the local group-001 release using a temporary scripted
  workflow to verify file/path/join/artifact behavior.

### Phase A acceptance

- [x] The actual group-001 directory is accepted as a v0.3 release.
- [x] Both images are hash-verified and routed as `ImageOnlyRuntimeCase`.
- [x] No private field reaches runtime inputs.
- [x] The run writes `predictions.jsonl`, `summary.json`, `traces/`, and
  `run_manifest.json`.
- [x] The data-pipeline classification scorer consumes v3 predictions unchanged.

Phase A is complete as an interface milestone. A direct group-001 boundary run
currently produces two explicit engineering errors and scorer Accuracy `0.0`, because
VisualFact/reinspect-v2 execution is intentionally not active. It makes no Gemini
calls and does not enter the claim-driven path. A real image-only investigation still
requires Phases B through E.

## 6. Phase B — Bootstrap VisualFacts and Initial Tasks

**Outcome:** the Agent starts from the image rather than a fabricated root claim.

### B1. Add strict runtime schemas

- [ ] Add `InvestigationBrief`.
- [ ] Add stable image-grounded `VisualEntity`.
- [ ] Add `VisualFact` kinds:
  `attribute`, `relation`, `internal_consistency`, and `text_claim`.
- [ ] Add bounded `ResearchTask`.
- [ ] Add evidence-backed `Finding`.
- [ ] Add canonical image-only trace collections.

Suggested focused module:

```text
src/orchestrator/investigation_models.py
```

### B2. Build an immutable image-only brief

- [ ] Derive the brief deterministically from `ImageOnlyRuntimeCase`.
- [ ] Use an inquiry objective, not an event/location/person hypothesis.
- [ ] Prevent the brief from containing evaluator-private facts or search queries.
- [ ] Keep the original case and brief immutable throughout the run.

### B3. Bootstrap perception and retrieval anchors

- [ ] Convert scene, OCR, logos, markings, entities, and normalized regions into
  candidate VisualFacts with pixel provenance.
- [ ] Preserve uncertainty; OCR or perception output is not automatically a decisive
  factual proposition.
- [ ] Run initial reverse-image search as Discovery only.
- [ ] Build retrieval anchors from visible evidence without treating result titles or
  snippets as Evidence.

### B4. Select initial tasks

- [ ] Select at most four high-value, evidence-routable initial tasks.
- [ ] Require every task to reference one or more facts.
- [ ] Keep task questions concrete and tool-actionable.
- [ ] Do not let initial planning write Evidence, Finding, fact status, or verdict.

### Phase B real checkpoint

Run the two group-001 images as soon as bootstrap planning can complete. The checkpoint
does not require correct final verdicts yet. Inspect whether:

- the astronaut image creates a location/event relation anchored to the visible
  `KENNEDY SPACE CENTER` sign and Moon Tree plaque;
- the vessel image creates an identity fact anchored to `HENRY B. BIGELOW`, NOAA,
  and `R 225`;
- retrieval tasks are specific enough to search;
- no gold statement is reproduced before private evaluation data is loaded.

## 7. Phase C — Dynamic Tasks and Periodic Reflection

**Outcome:** the single Agent can revise its investigation based on real observations
without unrestricted graph editing.

### C1. Deterministic task/fact store

- [ ] Implement stable IDs and append-only history for facts, tasks, and Findings.
- [ ] Enforce valid fact and task state transitions.
- [ ] Resolve tasks only through qualified Findings.
- [ ] Block/exhaust tasks only through recorded failures or explicit bounded
  data-void reasons.
- [ ] Never physically delete facts or tasks; retirement requires provenance.

Suggested module:

```text
src/orchestrator/task_store.py
```

### C2. Reflection contract

- [ ] Run structured Reflection after cumulative real tool actions 4, 8, 12, ...
- [ ] Count success, valid error, access block, and empty-result tool calls as real
  actions.
- [ ] Do not count Planning, Reflection, Coverage, Judgment, protocol corrections, or
  rejected tool calls.
- [ ] Allow Reflection to reprioritize tasks, add grounded tasks, identify gaps, and
  propose decisive-fact activation.
- [ ] Forbid Reflection from creating Evidence, writing a verdict, altering the brief,
  deleting history, or cancelling pending ReInspect.

Suggested module:

```text
src/orchestrator/reflection.py
```

### C3. Enforce design budgets

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

- [ ] Enforce budgets in deterministic code, not prompts alone.
- [ ] Preserve pending ReInspect obligations across Reflection.
- [ ] Treat repeated invalid Reflection as engineering failure according to the
  approved design.

### C4. Retire fixed-question replanning only in v2 mode

- [ ] In image-only v2 mode, let the task store plus latest Reflection determine the
  working view.
- [ ] Do not maintain two authoritative dynamic planning states.

### Phase C real checkpoint

Run group-001 with real providers and inspect the first eight real actions. Require:

- at least two Reflection checkpoints when the action count reaches eight;
- grounded task changes rather than generic repeated searches;
- no raw discovery promoted directly into a Finding;
- no fixed scripted tool order.

## 8. Phase D — Decisive-Fact Coverage and reinspect-v2 Judgment

**Outcome:** final verdicts are determined from explicit fact state and auditable
evidence chains.

### D1. Fact-level Evidence and Finding qualification

- [ ] Bind each Finding to task IDs, fact IDs, real Evidence IDs, and source families.
- [ ] Preserve the existing exact-span, artifact-hash, retrieval-time, access-policy,
  and injection checks.
- [ ] Require visual Findings to identify successful visual observations and regions.
- [ ] Prevent Discovery-only records from qualifying as Findings.

### D2. Decisive-fact activation

- [ ] Activate decisive facts only when they are image-grounded, decision-relevant,
  externally checkable or visually testable, and non-duplicative.
- [ ] Cap decisive facts and newly activated facts per Reflection.
- [ ] Keep unsupported speculative candidates non-decisive.

### D3. Deterministic v2 Coverage

- [ ] Audit each active decisive fact as supported, refuted, conflicted, blocked,
  exhausted, or unresolved.
- [ ] Keep exhausted mandatory ReInspect dependencies unresolved.
- [ ] Exclude discoveries and task churn from substantive progress.
- [ ] Stop only on:

  ```text
  coverage_complete
  information_saturated
  hard_budget_exhausted
  ```

- [ ] Use two consecutive Reflection intervals without substantive gain for
  saturation, with no pending ReInspect and no unattempted high-priority task.

Suggested module:

```text
src/orchestrator/coverage.py
```

### D4. Verdict basis compiler and validator

- [ ] Compile a deterministic expected verdict and allowed basis from decisive fact
  states.
- [ ] Require `policy_rule_id=reinspect-v2`.
- [ ] Require every final factual statement to trace:

  ```text
  VisualFact -> Finding -> Evidence -> successful tool call
  ```

- [ ] A `fake` verdict must identify at least one decisive refuted fact and its
  mechanism.
- [ ] A `real` verdict requires every decisive fact to be supported.
- [ ] Otherwise return `unverifiable` with fact-specific typed reasons.
- [ ] Reject model-selected IDs or verdicts that disagree with deterministic state.

Suggested module:

```text
src/orchestrator/verdict.py
```

### D5. Trace renderer and strict auditor

- [ ] Render v2 brief, facts, tasks, Findings, Reflections, coverage, and verdict
  basis.
- [ ] Extend `scripts/audit_real_trace.py` for image-only input and v2 invariants.
- [ ] Make strict audit fail on private-gold leakage, broken basis chains, invalid
  Reflection cadence, impossible state transitions, or ungrounded Findings.

## 9. Phase E — Real group-001 Acceptance

**Outcome:** prove the complete live path on actual data before expanding tests,
benchmark size, or training work.

### E1. Update the real canary

- [ ] Read required input mode and decision policy from the release manifest.
- [ ] Stop defaulting to required external/embedded claim modes.
- [ ] For image-only, require traces to contain `ImageOnlyRuntimeCase`,
  `InvestigationBrief`, v2 state, and `policy_rule_id=reinspect-v2`.
- [ ] Continue requiring real Gemini, search, image upload/reverse-image search,
  visit, and visual observation.
- [ ] Continue rejecting fake/scripted model names and engineering errors.

### E2. Run both development cases

Use a new empty output directory:

```powershell
$release = "D:\image-factual-verifier-data-pipeline\data\benchmarks\releases\image_only\image-only-benchmark-group-001"
$run = "D:\image-factual-verifier-runs\group-001-canary"

python scripts/run_real_canary.py `
  --benchmark "$release\runtime_input\cases.jsonl" `
  --output-dir $run `
  --limit 2
```

Then run the data-owned classification scorer:

```powershell
Push-Location D:\image-factual-verifier-data-pipeline
python -m scripts.benchmark.score_image_only_release `
  --release-root $release `
  --predictions "$run\predictions.jsonl" `
  --output "$run\benchmark-score.json"
Pop-Location
```

And audit every trace:

```powershell
python scripts/audit_real_trace.py "$run\traces" --json --strict-scheduler
```

### E3. Inspect real artifacts, not only exit codes

- [ ] Open both canonical traces and inspect task/fact/evidence chains.
- [ ] Inspect failed and empty provider results.
- [ ] Confirm the astronaut case investigates and refutes the depicted Kennedy
  location using independent source evidence for Johnson Space Center.
- [ ] Confirm the vessel case identifies NOAA Ship Henry B. Bigelow from visible
  markings and qualified external evidence.
- [ ] Confirm image upload, reverse-image search, visit, and visual observations are
  real successful calls.
- [ ] Confirm no evaluator-private decisive fact or acceptable-evidence text appears
  in pre-rollout model inputs.

### E4. Interpret development-subset metrics correctly

group-001 contains one supported and one refuted case, but no unverifiable case:

```json
{
  "release_stage": "development_subset",
  "all_verdict_classes_present": false
}
```

Therefore:

- perfect two-case predictions yield Accuracy `1.0`;
- observed-class Macro-F1 can reach `1.0`;
- fixed three-class Macro-F1 can reach only `0.666667` because the absent
  unverifiable class has zero support;
- this release proves a development loop, not formal three-class benchmark quality.

### Phase E acceptance

- [ ] Both cases complete without engineering error.
- [ ] Both predictions are correct.
- [ ] Every strict trace audit passes.
- [ ] Every verdict has a valid v2 basis.
- [ ] Required live tool classes were exercised.
- [ ] Artifacts were manually inspected.

If any item fails, record the real failure and fix the runtime before adding more mock
coverage. Do not redefine the canary to fit the current output.

## 10. Phase F — Process Evaluation

**Outcome:** v3 evaluates investigation quality separately from classification.

The data project supplies references through:

```text
evaluation/process_reference_protocol.json
evaluator_private/gold.jsonl
```

v3 implements the process scorer and emits:

```text
process_metrics.jsonl
trajectory_scores.jsonl
```

Do not combine these with classification into an opaque total score.

### F1. Implement deterministic reference alignment

- [ ] `decisive_fact_alignment`
- [ ] `acceptable_evidence_hit_rate`
- [ ] `citation_precision`
- [ ] `evidence_to_vision_bridge_completion`
- [ ] `verdict_basis_alignment`

- [ ] Canonicalize URLs and match artifact SHA/exact spans where supplied.
- [ ] Treat acceptable evidence as a reference set, not the only theoretically valid
  evidence; report unmatched high-quality evidence separately for review.
- [ ] Keep scorer access to private gold strictly post-rollout.

Suggested module:

```text
src/eval/process_scoring.py
```

### F2. Separate case metrics and trajectory diagnostics

`process_metrics.jsonl` contains stable per-case metric values.
`trajectory_scores.jsonl` contains diagnostic episode/task/fact details useful for
analysis and later data selection.

- [ ] Do not expose either file to the Agent.
- [ ] Do not let process score alter the factual verdict.
- [ ] Include protocol and gold artifact hashes in score metadata.

### Phase F acceptance

- [ ] The process scorer runs on group-001 after rollout.
- [ ] Every metric can be traced to runtime IDs and private reference objects.
- [ ] Missing runtime state is scored explicitly rather than silently ignored.
- [ ] Classification and process outputs remain separate.

## 11. Phase G — Trajectory Export and Student Training Preparation

This phase remains downstream. Do not start it merely because schemas or unit tests
exist.

### G1. Versioned trajectory export

- [ ] Derive Planning, ReAct, Reflection, and Judgment examples from canonical traces.
- [ ] Preserve tool/function-call structure and observation boundaries.
- [ ] Add tokenizer-aligned spans and loss masks only after the trace schema is stable.
- [ ] Exclude evaluator-private data and scorer annotations from model inputs/targets.
- [ ] Record source run, runtime commit, release ID, and protocol versions.

### G2. Freeze and audit a teacher dataset

Before training:

- [ ] real image-only canaries are stable across more than group-001;
- [ ] all selected traces pass strict audit;
- [ ] process metrics identify useful and defective trajectories;
- [ ] duplicate/near-duplicate and leakage checks pass;
- [ ] train/dev/test release boundaries are fixed.

### G3. Separate training plan

Create a new plan for Student training after the audited dataset exists. The intended
sequence remains:

1. ReAct/tool-use SFT;
2. Planning plus tool-use SFT;
3. Reflection SFT;
4. optional Judgment SFT;
5. preference optimization or RL only after stable supervised baselines.

Model choice, tokenizer, distributed strategy, hardware layout, and trainer
configuration do not belong in this runtime implementation plan.

## 12. Implementation Order and Commit Discipline

Execute in this order:

```text
Phase A interface
  -> Phase B bootstrap
  -> early real bootstrap checkpoint
  -> Phase C Reflection
  -> early real dynamic checkpoint
  -> Phase D Coverage/Judgment
  -> Phase E complete real acceptance
  -> Phase F process scoring
  -> Phase G export/training preparation
```

Use scoped commits. Recommended boundaries:

1. `feat: consume image-only v0.3 releases`
2. `feat: bootstrap image-only visual facts`
3. `feat: add bounded task reflection`
4. `feat: activate reinspect-v2 judgment`
5. `test: accept real image-only canary`
6. `feat: score image-only investigation process`

For each runtime phase:

- inspect the current dirty state before editing;
- preserve unrelated user changes;
- run focused tests first;
- run the full deterministic suite when shared runtime state changes;
- run the earliest possible real checkpoint;
- inspect artifacts;
- commit only the scoped runtime changes.

Do not modify or commit the data pipeline's current parallel work from the v3
worktree. When the data release is finalized, record its release ID and artifact
hashes; do not depend on its Python package or Git worktree state.

## 13. Next Immediate Task

Start with Phase B in one scoped implementation:

1. add strict `InvestigationBrief`, `VisualEntity`, and candidate `VisualFact`;
2. create the immutable brief from `ImageOnlyRuntimeCase`;
3. run real perception/OCR on both group-001 images;
4. reduce visible entities, text, logos, markings, regions, and relations into
   image-grounded candidate facts;
5. create at most four initial evidence-routable tasks;
6. persist the new state in canonical traces;
7. run the earliest real bootstrap checkpoint before adding Reflection.

Do not begin with process metrics, trajectory export, or training code. First make the
real Agent understand what is visibly present and what should be investigated.

## 14. Final Completion Checklist

- [x] v0.3 image-only releases use three-field
  `ImageOnlyRuntimeCase/reinspect-v2`.
- [x] Release routing is manifest-driven.
- [x] Private gold is inaccessible until rollout completes.
- [x] Optional inactive source policy is accepted.
- [x] No fabricated claim is introduced for image-only input.
- [ ] VisualFacts and tasks are image-grounded and bounded.
- [ ] Reflection fires only at cumulative real actions 4, 8, 12, ...
- [ ] Discoveries never become Findings or verdict evidence without qualification.
- [ ] Exhausted mandatory ReInspect remains unresolved.
- [ ] Every verdict basis follows
  `VisualFact -> Finding -> Evidence -> successful tool call`.
- [ ] `real`, `fake`, and `unverifiable` match decisive-fact state.
- [ ] Both group-001 real cases pass classification and strict trace audit.
- [ ] Process metrics and classification remain separate.
- [ ] No evaluator-private data enters traces used for training.
- [ ] Student training has its own later plan based on audited real trajectories.

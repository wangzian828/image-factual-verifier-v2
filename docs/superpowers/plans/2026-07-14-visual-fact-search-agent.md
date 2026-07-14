# VisualFact Search Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve the current fixed-question image verifier into the approved image-only, single-agent VisualFact search runtime while preserving its Gemini Interactions transport and strict evidence provenance.

**Architecture:** Work proceeds through independently shippable phases. Phase 0 repairs current correctness and stopping defects; Phase 1 adds dormant compatibility schemas; Phase 2 introduces bounded dynamic tasks and four-action Reflection checkpoints behind a feature flag; Phase 3 migrates Coverage and LedgerJudgment to decisive VisualFacts with a legacy compatibility projection. Trajectory export and Student training remain downstream consumers and do not enter the runtime critical path until it is stable.

**Tech Stack:** Python 3.11, Pydantic v2 strict models, Gemini Interactions API, pytest, canonical JSON traces, gpu-13 `ifv-agent` Conda environment.

---

## Plan Boundaries

The approved specification is `docs/superpowers/specs/2026-07-14-visual-fact-search-agent-design.md`. This implementation is intentionally split into six shippable plans within one document:

1. **Phase 0A:** fix exhausted ReInspect gating.
2. **Phase 0B:** fix substantive progress and configuration drift.
3. **Phase 1:** add dormant VisualFact/task/finding schemas and legacy adapters.
4. **Phase 2:** add periodic Reflection and dynamic tasks behind `IFV_DYNAMIC_INVESTIGATION=1`.
5. **Phase 3:** add VisualFact-aware Coverage/Judgment and `verdict_basis` behind `reinspect-v2`.
6. **Phase 4/5:** export trajectories, evaluate process quality, then train the Student in increasing scope.

Each phase must be green and committed before the next begins. Do not mix benchmark-pipeline work already present in the dirty local worktree into these commits.

## File Structure

### Existing files modified

- `src/orchestrator/pipeline.py`: thin orchestration only; Phase 0 gates, then delegate Reflection/Coverage/Judgment helpers rather than adding more policy code.
- `src/orchestrator/stage_runner.py`: invoke a supplied checkpoint hook after cumulative real tool-action counts; expose compact dynamic control state.
- `src/orchestrator/state.py`: canonical strict runtime schemas and aggregate trace fields.
- `src/orchestrator/investigation_state.py`: ReInspect dependency state; no free-form planning.
- `src/orchestrator/ledger.py`: immutable evidence compilation and fact/claim status projection.
- `src/orchestrator/context.py`: compact ReAct, Reflection, and Judgment views.
- `src/orchestrator/stages/planning.py`: legacy Planning remains during compatibility phases.
- `src/orchestrator/stages/verification.py`: dynamic-task-aware ReAct instructions.
- `src/orchestrator/stages/judgment.py`: v2 fact/basis output contract.
- `src/workflow.py`, `src/eval/run_eval.py`: configuration propagation and manifests.
- `src/trace_viewer.py`, `src/render_trace_html.py`: render new optional collections while supporting legacy traces.
- `scripts/audit_real_trace.py`: strict task/fact/basis and Reflection trace validation.
- `AGENTS.md`, `CLAUDE.md`, `docs/architecture.md`, `docs/agent-prompt-and-runtime-guide.md`, `docs/operations/gpu13.md`: update the active contract only when the corresponding phase becomes active.

### New focused modules

- `src/orchestrator/investigation_models.py`: `InvestigationBrief`, `VisualEntity`, `VisualFact`, `ResearchTask`, `Finding`, Reflection deltas, and activation proposals.
- `src/orchestrator/investigation_adapter.py`: explicit legacy `VerificationCase`/`PerceptionReport`/`InvestigationQuestion`/`EvidenceRecord` projections.
- `src/orchestrator/task_store.py`: deterministic task/fact/finding state transitions and budgets.
- `src/orchestrator/reflection.py`: checkpoint schema, prompt, validation, and application; no tool execution.
- `src/orchestrator/coverage.py`: substantive progress signatures and task/fact-aware deterministic audit.
- `src/orchestrator/verdict.py`: decisive-fact activation, deterministic expected verdict/basis, and v2 Judgment validation helpers.
- `src/trajectory/schema.py`: versioned derived training records.
- `src/trajectory/exporter.py`: canonical trace to Planning/ReAct/Reflection/Judgment examples.
- `src/trajectory/scoring.py`: deterministic process metrics and teacher episode score.

### New tests

- `test_investigation_models.py`
- `test_task_store.py`
- `test_reflection.py`
- `test_visual_fact_coverage.py`
- `test_verdict_basis.py`
- `test_trajectory_export.py`

Keep existing root-level test convention; do not introduce a second test tree in this migration.

## Server Validation Protocol

All meaningful tests run on gpu-13. Every implementation task follows this transport sequence after its local commit:

```bash
# Local Windows checkout
# Push only the scoped commit; never include .env, tmp/, outputs, or unrelated dirty files.
git push origin codex/gemini-interactions-agent

# gpu-13
cd /gs/home/wza/projects/image-factual-verifier-v2
bash scripts/server/update_gpu13_checkout.sh codex/gemini-interactions-agent
bash scripts/server/bootstrap_gpu13.sh
```

Every server test command starts with:

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent
```

This wrapper supplies the project proxy/data root and enforces `OMP_NUM_THREADS=1`. Stop if the server checkout is dirty; never edit or reset it.

---

# Phase 0A: ReInspect Correctness

### Task 1: Reproduce exhausted visual dependency bypass

**Files:**
- Modify: `test_evidence_grounding.py`
- Reference: `src/orchestrator/pipeline.py:772`
- Reference: `src/orchestrator/investigation_state.py:380`

- [ ] **Step 1: Write the failing gate regression**

Add imports for `InvestigationState` and `VisualQuestion`, then add:

```python
def test_exhausted_required_visual_question_reopens_web_supported_claim() -> None:
    ledgers = VerificationLedgers(
        claims=[
            ClaimRecord(
                claim_id="claim-q0",
                question_id="q0",
                text="The visible banner reads X.",
                claim_scope="visible_content",
                status="supported",
            )
        ]
    )
    investigation = InvestigationState(
        visual_questions=[
            VisualQuestion(
                visual_question_id="vq-q0",
                claim_id="claim-q0",
                source_evidence_id="e-web",
                target_bbox=[0.1, 0.1, 0.9, 0.4],
                expected_property="Whether the banner reads X",
                recommended_tools=["ocr_with_position"],
                status="exhausted",
                failed_attempts=2,
            )
        ]
    )

    Orchestrator._gate_claims_on_pending_visual_questions(ledgers, investigation)

    assert ledgers.claims[0].status == "open"
    assert "could not be observed after two real attempts" in (
        ledgers.claims[0].unresolved_distinction
    )
```

- [ ] **Step 2: Commit the red test without running locally**

```bash
git add test_evidence_grounding.py
git commit -m "test: expose exhausted ReInspect claim bypass"
git push origin codex/gemini-interactions-agent
```

- [ ] **Step 3: Run the targeted test on gpu-13 and verify red**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q \
  test_evidence_grounding.py::test_exhausted_required_visual_question_reopens_web_supported_claim
```

Expected: FAIL because status remains `supported`.

### Task 2: Enforce exhausted dependency in the gate

**Files:**
- Modify: `src/orchestrator/pipeline.py:790-803`
- Modify: `test_evidence_grounding.py`

- [ ] **Step 1: Implement the minimal exhausted gate**

Replace the exhausted loop with:

```python
        for claim in ledgers.claims:
            if (
                claim.claim_scope != "external_fact"
                and claim.claim_id in exhausted_claims
            ):
                claim.status = "open"
                claim.unresolved_distinction = (
                    "A search-conditioned reference could not be observed after two real attempts."
                )
```

This is intentionally conservative in Phase 0A: a fact that spawned a mandatory visual dependency cannot be closed by the same web-conditioned route after observation failure. Independent-route modeling arrives with VisualFacts in Phase 3.

- [ ] **Step 2: Add typed-reason assertion**

Extend the test:

```python
    reasons = derive_unverifiable_reasons(ledgers, "coverage_complete")
    assert reasons == [UnverifiableReason.UNREADABLE_REGION]
```

Import `derive_unverifiable_reasons` and `UnverifiableReason`.

- [ ] **Step 3: Commit and push**

```bash
git add src/orchestrator/pipeline.py test_evidence_grounding.py
git commit -m "fix: keep exhausted visual dependencies unresolved"
git push origin codex/gemini-interactions-agent
```

- [ ] **Step 4: Run focused gpu-13 tests**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q \
  test_evidence_grounding.py \
  test_unit.py -k "visual or judgment or unverifiable"
```

Expected: PASS.

### Task 3: Add native trace regression for two failed observations

**Files:**
- Modify: `test_full_native_agent_trace.py`
- Reference: `test_full_native_agent_trace.py:451-533`

- [ ] **Step 1: Add a scripted exhausted-ReInspect fixture**

Create a second response builder based on `_build_responses` that emits:

```text
visit evidence for a visible-content claim
→ first matching visual tool call returns status=error
→ second matching visual tool call returns status=error
→ final LedgerJudgment verdict=unverifiable and reason=unreadable_region
```

Use the existing `ScriptedInteractionsBackend` and `StaticTool`; configure the visual tool result as:

```python
{"status": "error", "error": "region could not be read"}
```

- [ ] **Step 2: Assert the end state**

```python
assert result["verdict"] == "unverifiable"
claim = result["state"]["ledgers"]["claims"][0]
assert claim["status"] == "open"
assert result["state"]["investigation_state"]["visual_questions"][0]["status"] == "exhausted"
assert "unreadable_region" in result["state"]["coverage_audits"][-1]["unverifiable_reasons"]
```

- [ ] **Step 3: Run the new trace test on gpu-13**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_full_native_agent_trace.py
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add test_full_native_agent_trace.py
git commit -m "test: cover exhausted visual revisit trace"
git push origin codex/gemini-interactions-agent
```

### Phase 0A acceptance

Run on gpu-13:

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q \
  test_unit.py test_evidence_grounding.py test_failure_contracts.py \
  test_native_interactions.py test_full_native_agent_trace.py
```

Accept only if all pass and the exhausted fixture produces `unverifiable`, never `real` or `fake`.

---

# Phase 0B: Substantive Progress and Config Alignment

### Task 4: Define substantive progress independently of discoveries

**Files:**
- Modify: `test_unit.py`
- Modify: `src/orchestrator/pipeline.py:1691-1708`

- [ ] **Step 1: Add a failing progress-signature test**

```python
def test_new_discovery_url_is_not_substantive_information_gain() -> None:
    before = VerificationLedgers()
    after = VerificationLedgers(
        discoveries=[
            DiscoveryRecord(
                discovery_id="d1",
                claim_id="claim-q0",
                function_call_id="call-1",
                tool_name="text_search",
                candidate_url="https://example.test/new-lead",
                candidate_type="serp",
            )
        ]
    )

    before_sig = Orchestrator._investigation_progress_signature(before)
    after_sig = Orchestrator._investigation_progress_signature(after)

    assert not any(new - old for old, new in zip(before_sig, after_sig))
```

Import `DiscoveryRecord` and `VerificationLedgers`.

- [ ] **Step 2: Push and verify red on gpu-13**

Expected: FAIL because the discovery URL appears in the current signature.

- [ ] **Step 3: Remove discoveries from the substantive signature**

Change `_investigation_progress_signature` to return:

```python
        evidence = frozenset(item.evidence_id for item in ledgers.evidence)
        claim_statuses = frozenset(
            f"{item.claim_id}:{item.status}" for item in ledgers.claims
        )
        source_families = frozenset(item.source_family for item in ledgers.sources)
        return evidence, claim_statuses, source_families
```

- [ ] **Step 4: Add positive controls**

Add tests showing a new eligible evidence ID and a claim status transition each produce gain.

- [ ] **Step 5: Run and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_unit.py -k "information_gain or saturation"
```

Expected: PASS.

```bash
git add src/orchestrator/pipeline.py test_unit.py
git commit -m "fix: exclude raw discoveries from substantive progress"
git push origin codex/gemini-interactions-agent
```

### Task 5: Prove lead-only iterations saturate

**Files:**
- Modify: `test_unit.py`

- [ ] **Step 1: Add a lead-only two-window orchestrator fixture**

Subclass `FakeOrchestrator`; script two verification windows that each return successful `text_search` calls with distinct URLs but no eligible evidence or fact-changing visual observation. Set:

```python
self.min_verification_iterations = 2
self.low_information_gain_patience = 2
self.max_verification_iterations = 4
```

- [ ] **Step 2: Assert deterministic stop**

```python
assert state["coverage_audits"][-1]["stop_reason"] == "information_saturated"
assert state["coverage_audits"][-1]["low_information_gain_streak"] == 2
assert result["verdict"] == "unverifiable"
```

- [ ] **Step 3: Run gpu-13 integration test**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_unit.py -k "lead_only or information_saturated"
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add test_unit.py
git commit -m "test: saturate investigations without evidence gain"
git push origin codex/gemini-interactions-agent
```

### Task 6: Make the action-window default 8 everywhere

**Files:**
- Verify/Modify: `src/orchestrator/pipeline.py:79`
- Verify/Modify: `src/workflow.py:51`
- Verify/Modify: `src/eval/run_eval.py:92`
- Modify: `AGENTS.md:96`
- Modify: `docs/architecture.md:146,217`
- Modify: `docs/operations/gpu13.md:227`
- Modify: `docs/agent-prompt-and-runtime-guide.md:129`
- Modify: `test_unit.py`

The current working-tree implementation already uses 8 in all three code defaults. Keep 8 as the migration default because Phase 2 will replace this legacy iteration cap with four-action Reflection segments; raising it back to 12 immediately before removal adds no value.

- [ ] **Step 1: Add a default-consistency test**

```python
def test_verification_action_window_default_is_eight() -> None:
    assert WorkflowConfig().max_rounds_verification == 8
    signature = inspect.signature(Orchestrator.__init__)
    assert signature.parameters["max_rounds_verification"].default == 8
```

- [ ] **Step 2: Update every active document from 12 to 8**

Use the term “eight verification actions before each coverage checkpoint,” not “twelve turns.” Preserve `MAX_INVESTIGATION_ACTIONS=24` as the global cap.

- [ ] **Step 3: Run docs/config focused tests on gpu-13**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_unit.py test_eval_artifacts.py
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/orchestrator/pipeline.py src/workflow.py src/eval/run_eval.py \
  AGENTS.md docs/architecture.md docs/operations/gpu13.md \
  docs/agent-prompt-and-runtime-guide.md test_unit.py
git commit -m "docs: align verification action window defaults"
git push origin codex/gemini-interactions-agent
```

### Phase 0B acceptance

Run the full credential-free gpu-13 contract suite:

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q \
  test_unit.py test_evidence_grounding.py test_failure_contracts.py \
  test_provider_failure_propagation.py test_native_interactions.py \
  test_gemini_interactions_contract.py test_gemini_vlm_interactions.py \
  test_full_native_agent_trace.py test_trace_viewer.py
```

Then run:

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python test_workflow_smoke.py
```

---

# Phase 1: Dormant Compatibility Data Layer

### Task 7: Add VisualFact investigation schemas

**Files:**
- Create: `src/orchestrator/investigation_models.py`
- Create: `test_investigation_models.py`
- Reference: `src/orchestrator/state.py`

- [ ] **Step 1: Write strict model tests**

Test the exact approved shapes:

```python
def test_visual_fact_requires_grounding() -> None:
    with pytest.raises(ValueError):
        VisualFact(
            fact_id="vf-1",
            kind="relation",
            subject_entity_id="person-1",
            predicate="supported_by",
            status="candidate",
        )


def test_research_task_references_fact_and_origin() -> None:
    task = ResearchTask(
        task_id="t1",
        fact_ids=["vf-1"],
        question="Can the apparent support relation be explained?",
        purpose="Resolve a potentially decisive visual relation.",
        priority=1,
        status="pending",
        origin_ids=["region-person-1", "region-ground-1"],
    )
    assert task.fact_ids == ["vf-1"]
```

- [ ] **Step 2: Run red on gpu-13**

Expected: import failure.

- [ ] **Step 3: Implement strict schemas**

Define in `investigation_models.py`:

```python
class InvestigationBrief(StrictModel):
    brief_id: str = Field(min_length=1, max_length=200)
    input_mode: Literal["image_only"] = "image_only"
    objective: str = Field(min_length=1, max_length=1000)
    required_output: List[str] = Field(default_factory=list, max_length=16)
    stop_policy: Literal["coverage_or_bounded_unresolved"] = "coverage_or_bounded_unresolved"

class VisualEntity(StrictModel):
    entity_id: str = Field(min_length=1, max_length=200)
    entity_type: str = Field(min_length=1, max_length=100)
    name: str = Field(default="", max_length=300)
    bbox: Optional[List[float]] = None
    origin_ids: List[str] = Field(default_factory=list, min_length=1, max_length=8)

class VisualFact(StrictModel):
    fact_id: str = Field(min_length=1, max_length=200)
    kind: Literal["attribute", "relation", "internal_consistency", "text_claim"]
    subject_entity_id: str = Field(min_length=1, max_length=200)
    predicate: str = Field(min_length=1, max_length=200)
    object_entity_id: Optional[str] = Field(default=None, max_length=200)
    observed_value: Optional[str] = Field(default=None, max_length=1000)
    status: Literal["candidate", "active", "supported", "refuted", "conflicted", "blocked", "exhausted", "retired"] = "candidate"
    role: Literal["contextual", "supporting", "decisive"] = "contextual"
    basis_ids: List[str] = Field(default_factory=list, min_length=1, max_length=16)
    origin_ids: List[str] = Field(default_factory=list, max_length=16)
    superseded_by_fact_id: Optional[str] = None

class ResearchTask(StrictModel):
    task_id: str = Field(min_length=1, max_length=200)
    fact_ids: List[str] = Field(min_length=1, max_length=8)
    question: str = Field(min_length=1, max_length=1000)
    purpose: str = Field(min_length=1, max_length=1000)
    priority: int = Field(default=1, ge=1, le=3)
    status: Literal["pending", "active", "resolved", "blocked", "exhausted"] = "pending"
    parent_task_id: Optional[str] = None
    origin_ids: List[str] = Field(min_length=1, max_length=16)
    suggested_tools: List[str] = Field(default_factory=list, max_length=6)
    suggested_queries: List[str] = Field(default_factory=list, max_length=6)
    finding_ids: List[str] = Field(default_factory=list, max_length=16)
    failure_ids: List[str] = Field(default_factory=list, max_length=16)
    attempt_count: int = Field(default=0, ge=0)

class Finding(StrictModel):
    finding_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=200)
    fact_ids: List[str] = Field(min_length=1, max_length=8)
    statement: str = Field(min_length=1, max_length=2000)
    stance: Literal["support", "refute", "neutral"]
    evidence_ids: List[str] = Field(min_length=1, max_length=16)
    source_family_ids: List[str] = Field(default_factory=list, max_length=16)
    quality: Literal["strong", "moderate", "weak"]
```

Add bbox validators equivalent to `Entity.validate_bbox` and a model validator requiring `retired` facts to provide `superseded_by_fact_id` or evidence-bearing origin IDs.

- [ ] **Step 4: Run green and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_investigation_models.py
```

```bash
git add src/orchestrator/investigation_models.py test_investigation_models.py
git commit -m "feat: add VisualFact investigation schemas"
git push origin codex/gemini-interactions-agent
```

### Task 8: Add explicit legacy adapters

**Files:**
- Create: `src/orchestrator/investigation_adapter.py`
- Modify: `test_investigation_models.py`

- [ ] **Step 1: Write round-trip tests**

Test:

```text
VerificationCase → InvestigationBrief
Entity/TextRegion/PerceptionReport → VisualEntity/candidate VisualFacts
InvestigationQuestion → ResearchTask
EvidenceRecord → Finding proposal
```

Assertions must preserve legacy IDs, scope, immutable text, bbox, tool call, and evidence IDs.

- [ ] **Step 2: Implement explicit converters**

Required functions:

```python
def brief_from_case(case: VerificationCase) -> InvestigationBrief: ...
def entities_from_perception(report: PerceptionReport) -> list[VisualEntity]: ...
def candidate_facts_from_perception(report: PerceptionReport) -> list[VisualFact]: ...
def task_from_question(question: InvestigationQuestion) -> ResearchTask: ...
def finding_from_evidence(record: EvidenceRecord, task_id: str) -> Finding: ...
```

Use stable `_id`-style SHA-256 prefixes over canonical JSON inputs; do not use array positions alone. Candidate perception facts remain contextual and candidate; adapters must not infer physical impossibility or event identity.

- [ ] **Step 3: Run gpu-13 tests and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_investigation_models.py test_unit.py -k "context_rendering"
```

```bash
git add src/orchestrator/investigation_adapter.py test_investigation_models.py
git commit -m "feat: project legacy state into investigation records"
git push origin codex/gemini-interactions-agent
```

### Task 9: Add dormant investigation state to canonical traces

**Files:**
- Modify: `src/orchestrator/state.py:424-500`
- Modify: `src/orchestrator/pipeline.py:275-335`
- Modify: `test_trace_viewer.py`
- Modify: `test_unit.py`

- [ ] **Step 1: Add failing serialization assertions**

Assert new trace keys exist but do not change current verdict/control behavior:

```python
assert state["investigation_brief"]["input_mode"] == "image_only"
assert state["visual_entities"]
assert isinstance(state["visual_facts"], list)
assert isinstance(state["research_tasks"], list)
assert isinstance(state["findings"], list)
```

- [ ] **Step 2: Add aggregate state fields**

Add to `VerificationState`:

```python
investigation_brief: Optional[InvestigationBrief] = None
visual_entities: List[VisualEntity] = field(default_factory=list)
visual_facts: List[VisualFact] = field(default_factory=list)
research_tasks: List[ResearchTask] = field(default_factory=list)
findings: List[Finding] = field(default_factory=list)
reflection_records: List[Any] = field(default_factory=list)
```

Serialize each with `model_dump(mode="json")` in `to_dict`.

- [ ] **Step 3: Populate dormant projections after perception/planning**

In `Orchestrator.run`, after perception and planning succeed, call the adapters. Do not read these fields in Coverage, ReAct, or Judgment yet.

- [ ] **Step 4: Run regression suite and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_unit.py test_trace_viewer.py test_full_native_agent_trace.py
```

Expected: all legacy assertions remain valid; new keys are additive.

```bash
git add src/orchestrator/state.py src/orchestrator/pipeline.py \
  test_unit.py test_trace_viewer.py
git commit -m "feat: persist dormant VisualFact investigation state"
git push origin codex/gemini-interactions-agent
```

### Phase 1 acceptance

Run the Phase 0B full suite plus `test_investigation_models.py`. Confirm the same controlled trace verdict and ledger IDs as before; only additive trace keys may differ.

---

# Phase 2: Dynamic Tasks and Periodic Reflection

### Task 10: Implement deterministic task/fact store

**Files:**
- Create: `src/orchestrator/task_store.py`
- Create: `test_task_store.py`

- [ ] **Step 1: Write transition tests**

Cover:

```text
initial tasks ≤ 4
total tasks ≤ 12
new tasks per Reflection ≤ 3
parent depth ≤ 2
decisive facts ≤ 6
new decisive facts per Reflection ≤ 2
resolved task requires Finding IDs
blocked task requires Failure IDs
duplicate task/fact IDs or semantic duplicates are rejected
retired facts retain provenance
```

Representative test:

```python
def test_resolved_task_requires_existing_finding() -> None:
    store = TaskFactStore(tasks=[task("t1")])
    with pytest.raises(ValueError, match="finding"):
        store.update_task("t1", status="resolved", finding_ids=[])
```

- [ ] **Step 2: Implement `TaskFactStore`**

Required API:

```python
class TaskFactStore:
    def add_task(self, task: ResearchTask, *, reflection_new_count: int) -> None: ...
    def update_task(self, update: TaskUpdate, findings: Mapping[str, Finding], failures: Mapping[str, FailureRecord]) -> None: ...
    def add_finding(self, finding: Finding, evidence: Mapping[str, EvidenceRecord]) -> None: ...
    def activate_fact(self, proposal: FactActivationProposal, available_tools: set[str]) -> None: ...
    def open_tasks(self) -> list[ResearchTask]: ...
    def substantive_signature(self) -> tuple[frozenset[str], ...]: ...
```

Store transitions must be deterministic and side-effect-free on rejection.

- [ ] **Step 3: Run and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_task_store.py
```

```bash
git add src/orchestrator/task_store.py test_task_store.py
git commit -m "feat: add deterministic investigation task store"
git push origin codex/gemini-interactions-agent
```

### Task 11: Define structured Reflection contract

**Files:**
- Modify: `src/orchestrator/investigation_models.py`
- Create: `src/orchestrator/reflection.py`
- Create: `test_reflection.py`

- [ ] **Step 1: Add strict Reflection schema tests**

Define and test:

```python
class TaskUpdate(StrictModel):
    task_id: str
    status: Literal["active", "resolved", "blocked", "exhausted"]
    finding_ids: List[str] = []
    failure_ids: List[str] = []
    reason: str

class NewTaskProposal(StrictModel):
    question: str
    purpose: str
    fact_ids: List[str]
    priority: int
    parent_task_id: Optional[str]
    origin_ids: List[str]
    suggested_tools: List[str]
    suggested_queries: List[str]

class FactActivationProposal(StrictModel):
    fact_id: str
    requested_role: Literal["decisive"]
    reason: str
    origin_ids: List[str]
    candidate_routes: List[str]

class ReflectionOutput(StrictModel):
    task_updates: List[TaskUpdate] = Field(max_length=12)
    new_tasks: List[NewTaskProposal] = Field(max_length=3)
    proposed_decisive_facts: List[FactActivationProposal] = Field(max_length=2)
    recommended_next_task_ids: List[str] = Field(max_length=4)
    remaining_gaps: List[KnowledgeGap] = Field(max_length=6)
    ready_to_finish: bool = False
```

- [ ] **Step 2: Implement compact Reflection rendering**

In `reflection.py`, implement:

```python
def render_reflection_context(
    brief, tasks, facts, findings, ledgers, investigation_state, remaining_actions
) -> str:
    ...
```

The context contains one-line task/fact/finding summaries, source-family coverage, grouped failures, pending/exhausted visual questions, and budget. Never include raw webpage bodies.

- [ ] **Step 3: Implement validation/application**

```python
def apply_reflection(
    output: ReflectionOutput,
    store: TaskFactStore,
    ledgers: VerificationLedgers,
    available_tools: set[str],
    after_action_count: int,
) -> ReflectionRecord:
    ...
```

Generate stable task/reflection IDs from canonical proposal content; reject invalid deltas without partially mutating the store.

- [ ] **Step 4: Run and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_reflection.py test_task_store.py
```

```bash
git add src/orchestrator/investigation_models.py src/orchestrator/reflection.py \
  test_reflection.py
git commit -m "feat: add structured investigation reflection"
git push origin codex/gemini-interactions-agent
```

### Task 12: Trigger Reflection after every four real tool actions

**Files:**
- Modify: `src/orchestrator/stage_runner.py:434-812`
- Modify: `src/orchestrator/pipeline.py:454-769`
- Modify: `src/orchestrator/context.py`
- Modify: `test_native_interactions.py`
- Modify: `test_reflection.py`

- [ ] **Step 1: Write checkpoint cadence tests**

Use a scripted native backend with nine successful/error tool calls. Assert checkpoint hook receives cumulative counts `[4, 8]` only. Protocol errors and rejected final output do not count.

- [ ] **Step 2: Add an optional hook to `StageRunner`**

```python
reflection_hook: Optional[
    Callable[[int, Sequence[StageStep]], Awaitable[Optional[ReflectionRecord]]]
] = None
```

After recording a real `tool_call` step, compute cumulative verification actions using `episode_actions_before + action_turns`. Call the hook only when divisible by 4 and not previously reflected.

- [ ] **Step 3: Build the hook in `Orchestrator._run_verification`**

The hook:

1. renders the Reflection context;
2. runs a no-tool, schema-constrained Gemini Interactions call with `thinking_level=minimal`;
3. permits one same-interaction correction;
4. applies the delta through `TaskFactStore`;
5. stores `ReflectionRecord` in `state.reflection_records`;
6. returns compact focus/task/fact updates for the next `agent_control_state`.

Two consecutive failed Reflection checkpoints raise engineering error. One failed checkpoint retains old state and permits one more four-action segment.

- [ ] **Step 4: Extend control context**

Add to `_agent_control_state`:

```json
{
  "active_tasks": [],
  "decisive_fact_ids": [],
  "latest_reflection_focus": [],
  "next_checkpoint_in": 4
}
```

Keep the payload bounded; use IDs plus one-line text.

- [ ] **Step 5: Run and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_reflection.py test_native_interactions.py \
  test_gemini_interactions_contract.py
```

```bash
git add src/orchestrator/stage_runner.py src/orchestrator/pipeline.py \
  src/orchestrator/context.py test_native_interactions.py test_reflection.py
git commit -m "feat: run Reflection every four investigation actions"
git push origin codex/gemini-interactions-agent
```

### Task 13: Add feature flag and dual-run trace comparison

**Files:**
- Modify: `src/workflow.py`
- Modify: `src/eval/run_eval.py`
- Modify: `src/orchestrator/pipeline.py`
- Modify: `test_unit.py`
- Modify: `test_eval_artifacts.py`

- [ ] **Step 1: Add flag tests**

```python
assert WorkflowConfig().dynamic_investigation is False
assert run_manifest["config"]["dynamic_investigation"] is True
```

- [ ] **Step 2: Add explicit config**

```python
dynamic_investigation: bool = False
reflection_interval: int = 4
max_research_tasks: int = 12
max_decisive_facts: int = 6
```

CLI flags:

```text
--dynamic-investigation
--reflection-interval 4
--max-research-tasks 12
--max-decisive-facts 6
```

Do not read a hidden env var inside domain logic; config is propagated explicitly and recorded in manifests/traces.

- [ ] **Step 3: Gate behavior**

When disabled, no Reflection call occurs and current plan/replanning behavior is byte-compatible. When enabled, dynamic tasks/Reflection run but legacy Coverage/Judgment remain authoritative until Phase 3.

- [ ] **Step 4: Run and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_unit.py test_eval_artifacts.py test_reflection.py
```

```bash
git add src/workflow.py src/eval/run_eval.py src/orchestrator/pipeline.py \
  test_unit.py test_eval_artifacts.py
git commit -m "feat: gate dynamic investigation runtime"
git push origin codex/gemini-interactions-agent
```

### Task 14: Replace fixed-question replanning under the flag

**Files:**
- Modify: `src/orchestrator/pipeline.py:954-1307,2662-2680`
- Modify: `src/orchestrator/context.py:183-296`
- Modify: `src/orchestrator/stages/planning.py:88-107`
- Modify: `test_unit.py:388-601`
- Modify: `test_full_native_agent_trace.py`

- [ ] **Step 1: Add dynamic-gap integration test**

Script:

```text
initial plan omits old-image reuse
→ RIS/visit produces a 2023 source Finding
→ Reflection creates a new provenance task/fact
→ ReAct investigates it in the next segment
```

Assert the new task ID exists, has `parent_task_id`, and the trace contains a Reflection record; no `replanning` stage is invoked when dynamic mode is on.

- [ ] **Step 2: Branch the outer loop**

In dynamic mode, skip `_replan_verification`; the TaskFactStore and latest Reflection determine the next working view. Keep legacy methods untouched for disabled mode until Phase 3 migration is accepted.

- [ ] **Step 3: Remove full-plan assumptions from verification context**

Render active ResearchTasks and facts first; include legacy immutable question/claim bindings only as compatibility IDs.

- [ ] **Step 4: Run and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_unit.py test_reflection.py \
  test_full_native_agent_trace.py
```

```bash
git add src/orchestrator/pipeline.py src/orchestrator/context.py \
  src/orchestrator/stages/planning.py test_unit.py \
  test_full_native_agent_trace.py
git commit -m "feat: drive dynamic investigations from Reflection tasks"
git push origin codex/gemini-interactions-agent
```

### Phase 2 acceptance

On gpu-13 run the complete suite, then the real Gemini probe:

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python test_workflow_smoke.py
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python scripts/probe_gemini_interactions.py --model gemini-3-flash-preview
```

Run both flag modes against the controlled fixture. Accept only if legacy mode is unchanged and dynamic mode produces bounded, valid Reflection/task records.

---

# Phase 3: VisualFact-Aware Coverage and Judgment

### Task 15: Extract task/fact-aware Coverage Audit

**Files:**
- Create: `src/orchestrator/coverage.py`
- Create: `test_visual_fact_coverage.py`
- Modify: `src/orchestrator/pipeline.py:1464-1708`
- Modify: `src/orchestrator/state.py`

- [ ] **Step 1: Write deterministic fact-coverage tests**

Cover:

```text
no decisive facts → not real/coverage-complete
a refuted decisive fact → fake-eligible
all decisive facts supported → real-eligible
open/conflicted/blocked/exhausted decisive fact → unverifiable-eligible
pending visual dependency blocks completion
lead-only growth does not reset substantive streak
```

- [ ] **Step 2: Add fact/task coverage records**

```python
class FactResolution(StrictModel):
    fact_id: str
    status: Literal["candidate", "active", "supported", "refuted", "conflicted", "blocked", "exhausted", "retired"]
    finding_ids: List[str] = []
    evidence_ids: List[str] = []
    remaining_gap: str = ""

class InvestigationCoverage(StrictModel):
    fact_resolutions: List[FactResolution] = []
    unresolved_decisive_fact_ids: List[str] = []
    stop_reason: Literal["continue", "coverage_complete", "information_saturated", "hard_budget_exhausted"]
    expected_verdict: Literal["real", "fake", "unverifiable"]
```

Store it alongside legacy `CoverageAudit` during migration.

- [ ] **Step 3: Implement pure `audit_visual_facts`**

Inputs are tasks, facts, findings, ledgers, ReInspect state, action budget, and prior substantive signature. Do not call models or mutate evidence.

- [ ] **Step 4: Delegate from pipeline in dynamic mode**

Leave legacy `_audit_plan_coverage` for legacy mode; dynamic mode writes both a compatibility `CoverageAudit` and canonical `InvestigationCoverage`.

- [ ] **Step 5: Run and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_visual_fact_coverage.py test_unit.py \
  test_evidence_grounding.py
```

```bash
git add src/orchestrator/coverage.py src/orchestrator/state.py \
  src/orchestrator/pipeline.py test_visual_fact_coverage.py
git commit -m "feat: audit decisive VisualFact coverage"
git push origin codex/gemini-interactions-agent
```

### Task 16: Add deterministic verdict basis compiler

**Files:**
- Create: `src/orchestrator/verdict.py`
- Create: `test_verdict_basis.py`
- Modify: `src/orchestrator/investigation_models.py`

- [ ] **Step 1: Write aggregation and provenance tests**

```python
def test_any_refuted_decisive_fact_compiles_fake() -> None: ...
def test_all_supported_decisive_facts_compile_real() -> None: ...
def test_open_decisive_fact_compiles_unverifiable() -> None: ...
def test_basis_requires_finding_and_evidence_chain() -> None: ...
def test_contextual_refutation_cannot_compile_fake() -> None: ...
```

- [ ] **Step 2: Define basis model**

```python
class VerdictBasisEntry(StrictModel):
    fact_id: str
    fact_kind: Literal["attribute", "relation", "internal_consistency", "text_claim"]
    statement: str
    status: Literal["supported", "refuted", "conflicted", "blocked", "exhausted"]
    mechanism: Literal[
        "reused_old_image",
        "wrong_event",
        "landmark_mismatch",
        "impossible_causal_dynamics",
        "inconsistent_lighting_or_reflection",
        "synthetic_presented_as_documentary",
        "other_verified_relation",
    ]
    finding_ids: List[str]
    evidence_ids: List[str]
```

- [ ] **Step 3: Implement compiler**

```python
def compile_expected_verdict(
    facts: Sequence[VisualFact],
    findings: Mapping[str, Finding],
    evidence: Mapping[str, EvidenceRecord],
) -> ExpectedVerdict:
    ...
```

Validate `VisualFact → Finding → Evidence → successful tool call` ownership before including a basis entry.

- [ ] **Step 4: Run and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_verdict_basis.py
```

```bash
git add src/orchestrator/verdict.py src/orchestrator/investigation_models.py \
  test_verdict_basis.py
git commit -m "feat: compile evidence-bound VisualFact verdict basis"
git push origin codex/gemini-interactions-agent
```

### Task 17: Add `reinspect-v2` LedgerJudgment contract

**Files:**
- Modify: `src/orchestrator/state.py:391-410`
- Modify: `src/orchestrator/stages/judgment.py`
- Modify: `src/orchestrator/context.py:112-181`
- Modify: `src/orchestrator/pipeline.py:805-1435`
- Modify: `test_verdict_basis.py`
- Modify: `test_full_native_agent_trace.py`

- [ ] **Step 1: Add backward-compatible fields**

```python
class LedgerJudgment(StrictModel):
    ...
    verdict_basis: List[VerdictBasisEntry] = Field(default_factory=list)

class FinalJudgment(StrictModel):
    ...
    verdict_basis: List[VerdictBasisEntry] = Field(default_factory=list)
```

Legacy constructions must still parse with an empty list.

- [ ] **Step 2: Add v2 Judgment context**

Render only decisive facts, their valid Findings/Evidence, unresolved typed states, and the deterministic expected verdict/basis. Do not expose discoveries or raw unsupported hypotheses.

- [ ] **Step 3: Add v2 validator**

Require:

```text
policy_rule_id == reinspect-v2
model verdict equals deterministic expected verdict
model verdict_basis equals deterministic basis as a set
all fact/finding/evidence IDs exist and ownership/stance match
all refuted decisive facts appear in fake basis
all unresolved decisive facts have typed reasons
legacy claim_decisions remain present as a compatibility projection during migration
```

- [ ] **Step 4: Store basis in FinalJudgment and trace**

Compile final prose from validated fact/finding/evidence text; no free-form model fact may enter `reasoning_chain`.

- [ ] **Step 5: Add v1/v2 trace tests**

Legacy `reinspect-v1` trace has empty/missing-compatible basis. Dynamic `reinspect-v2` trace has a fully linked basis and the expected three-class result.

- [ ] **Step 6: Run and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_verdict_basis.py test_full_native_agent_trace.py \
  test_evidence_grounding.py
```

```bash
git add src/orchestrator/state.py src/orchestrator/stages/judgment.py \
  src/orchestrator/context.py src/orchestrator/pipeline.py \
  test_verdict_basis.py test_full_native_agent_trace.py
git commit -m "feat: validate VisualFact-driven ledger judgments"
git push origin codex/gemini-interactions-agent
```

### Task 18: Trace renderer and strict auditor

**Files:**
- Modify: `src/trace_viewer.py`
- Modify: `src/render_trace_html.py`
- Modify: `scripts/audit_real_trace.py`
- Modify: `test_trace_viewer.py`
- Create: `test_audit_real_trace.py`

- [ ] **Step 1: Write legacy/new render tests**

Assert HTML contains `Visual Facts`, `Research Tasks`, `Reflection Checkpoints`, and `Verdict Basis` for v2 traces. Assert a legacy trace without these keys still renders.

- [ ] **Step 2: Render new optional sections**

Keep canonical JSON as source; HTML remains derived.

- [ ] **Step 3: Add strict audit rules**

Audit:

```text
task/fact/finding IDs unique
all origin/finding/evidence refs exist
decisive activation limits respected
Reflection cadence at cumulative actions 4,8,...
no discovery ID used as verdict evidence
verdict_basis chain complete and matches verdict
exhausted visual dependencies are not decisive-supported
```

- [ ] **Step 4: Run and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_trace_viewer.py test_audit_real_trace.py
```

```bash
git add src/trace_viewer.py src/render_trace_html.py \
  scripts/audit_real_trace.py test_trace_viewer.py test_audit_real_trace.py
git commit -m "feat: audit and render VisualFact investigation traces"
git push origin codex/gemini-interactions-agent
```

### Task 19: Activate dynamic image-only runtime and update contracts

**Files:**
- Modify: `src/workflow.py`
- Modify: `src/eval/run_eval.py`
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `docs/architecture.md`
- Modify: `docs/agent-prompt-and-runtime-guide.md`
- Modify: `docs/operations/gpu13.md`
- Modify: `test_eval_artifacts.py`

- [ ] **Step 1: Make v2 explicit, not silent**

Add CLI/config:

```text
--decision-policy reinspect-v1|reinspect-v2
```

Require `reinspect-v2` only when `--dynamic-investigation` is enabled. Continue defaulting to v1 until the frozen-real acceptance run passes.

- [ ] **Step 2: Record policy in case, trace, and manifest**

Reject mismatches between `VerificationCase.decision_policy_version`, runtime policy, and Judgment `policy_rule_id`.

- [ ] **Step 3: Update active docs after acceptance**

Replace fixed `Planning → iterative verification → Replanning` as the primary path only after v2 acceptance. Document v1 as compatibility mode with a removal milestone.

- [ ] **Step 4: Run full gpu-13 suite and probe**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python test_workflow_smoke.py
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python scripts/probe_gemini_interactions.py --model gemini-3-flash-preview
```

- [ ] **Step 5: Commit**

```bash
git add src/workflow.py src/eval/run_eval.py AGENTS.md CLAUDE.md \
  docs/architecture.md docs/agent-prompt-and-runtime-guide.md \
  docs/operations/gpu13.md test_eval_artifacts.py
git commit -m "feat: expose VisualFact investigation policy"
git push origin codex/gemini-interactions-agent
```

### Phase 3 frozen-real acceptance

Launch a small unique v2 run on gpu-13:

```bash
cd /gs/home/wza/projects/image-factual-verifier-v2
run_id="visual-fact-v2-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/server/start_eval_gpu13.sh \
  --benchmark "$IFV_DATA_ROOT/benchmarks/candidates/real_seed_v0/averimatec/evaluation.jsonl" \
  --source-access-policy "$IFV_DATA_ROOT/benchmarks/candidates/real_seed_v0/averimatec/source_access_policy.json" \
  --output-dir "$IFV_DATA_ROOT/runs/eval/$run_id" \
  --dynamic-investigation \
  --decision-policy reinspect-v2 \
  --limit 3 \
  --concurrency 1
```

After completion:

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python scripts/audit_real_trace.py \
  "$IFV_DATA_ROOT/runs/eval/$run_id/traces" \
  --json --strict-scheduler
```

Accept only if every trace passes, every final statement has a fact/finding/evidence/tool-call chain, task/fact budgets hold, and no exhausted visual dependency is closed.

---

# Phase 4: Trajectory and Evaluation Layer

### Task 20: Add versioned trajectory schema and exporter

**Files:**
- Create: `src/trajectory/__init__.py`
- Create: `src/trajectory/schema.py`
- Create: `src/trajectory/exporter.py`
- Create: `test_trajectory_export.py`

- [ ] **Step 1: Write schema/export tests**

Required records:

```python
class PolicyExample(StrictModel):
    trajectory_version: Literal["ifv-policy-v1"]
    episode_id: str
    step_id: str
    example_type: Literal["planning", "react", "reflection", "judgment"]
    runtime_observation_refs: List[str]
    policy_input: Dict[str, Any]
    policy_action: Dict[str, Any]
    action_valid: bool
    terminated: bool
    fatal_boundary: bool = False
```

Test one controlled trace produces one Planning example, one example per policy tool action, one per Reflection, and one Judgment example.

- [ ] **Step 2: Implement pure trace exporter**

Do not alter canonical traces. The exporter reads them and writes derived JSONL under `IFV_DATA_ROOT/generated/trajectories/<version>/`.

- [ ] **Step 3: Enforce visibility/masking boundaries**

`policy_input` contains exactly the stage view seen by the policy. It must exclude benchmark gold, access-policy exclusions, evaluator outputs, deterministic future state, and private tool runtime metrics.

- [ ] **Step 4: Run and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_trajectory_export.py
```

```bash
git add src/trajectory test_trajectory_export.py
git commit -m "feat: export versioned policy trajectories"
git push origin codex/gemini-interactions-agent
```

### Task 21: Add tokenizer-aligned spans and loss masks

**Files:**
- Modify: `src/trajectory/schema.py`
- Modify: `src/trajectory/exporter.py`
- Modify: `test_trajectory_export.py`

- [ ] **Step 1: Define tokenized record**

```python
policy_input_token_ids: List[int]
policy_action_token_ids: List[int]
policy_action_loss_mask: List[int]
```

Mask `1` only for policy-authored tool arguments/actions, Reflection output, Planning output, and optional Judgment output. Mask `0` for system/developer/user text, schemas, images/placeholders, tool results, reducers, ledgers, audits, corrections, padding, and evaluator data.

- [ ] **Step 2: Add fake-tokenizer tests**

Use a deterministic test tokenizer to assert exact mask boundaries, including fatal trajectories where post-fatal tokens are zero-masked.

- [ ] **Step 3: Implement tokenizer adapter interface**

Keep model-specific tokenizer loading out of runtime modules. The exporter accepts an injected tokenizer adapter.

- [ ] **Step 4: Run and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_trajectory_export.py
```

```bash
git add src/trajectory/schema.py src/trajectory/exporter.py \
  test_trajectory_export.py
git commit -m "feat: add policy action loss masks"
git push origin codex/gemini-interactions-agent
```

### Task 22: Add process metrics and teacher scoring

**Files:**
- Create: `src/trajectory/scoring.py`
- Modify: `src/eval/run_eval.py`
- Modify: `test_trajectory_export.py`
- Modify: `test_eval_artifacts.py`

- [ ] **Step 1: Define deterministic process metrics**

Compute:

```text
valid Finding precision
necessary-gap coverage when rubric data exists
false task/activation rate
evidence-to-vision bridge validity
ReInspect resolution rate
duplicate action rate
premature finish rate
decisive evidence per tool action
cost, latency, and first error
```

Metrics requiring privileged rubric data run only in evaluator-private code and never enter model-visible traces.

- [ ] **Step 2: Define teacher score**

```python
score = (
    result_reward
    + grounded_finding_reward
    + gap_coverage_reward
    + bridge_reward
    + stop_calibration_reward
    - duplicate_action_penalty
    - invalid_task_penalty
    - normalized_cost_penalty
)
```

Store every component separately; never persist only an opaque scalar.

- [ ] **Step 3: Add eval artifacts**

Write `process_metrics.jsonl` and `trajectory_scores.jsonl` alongside, not inside, canonical traces.

- [ ] **Step 4: Run and commit**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q test_trajectory_export.py test_eval_artifacts.py
```

```bash
git add src/trajectory/scoring.py src/eval/run_eval.py \
  test_trajectory_export.py test_eval_artifacts.py
git commit -m "feat: score investigation process trajectories"
git push origin codex/gemini-interactions-agent
```

---

# Phase 5: Student Training Sequence

Phase 5 must be a separate training implementation plan after Phase 4 data audits. Do not start model training merely because exporter tests pass.

### Task 23: Freeze and audit the teacher dataset

**Files:**
- Create: `scripts/trajectory/export_dataset.py`
- Create: `scripts/trajectory/audit_dataset.py`
- Create: `configs/training/ifv_policy_v1.yaml`
- Create: `test_trajectory_dataset.py`

- [ ] **Step 1: Export split-safe examples**

Group by investigation-world/source-family split keys; never split steps from one episode across train/validation/test.

- [ ] **Step 2: Audit before training**

Require:

```text
no gold/access-policy leakage
all references resolvable
all loss masks aligned
no duplicate episodes across splits
teacher score distribution reported
failure/action/verdict balance reported
```

- [ ] **Step 3: Run audit on gpu-13**

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python scripts/trajectory/audit_dataset.py \
  --input "$IFV_DATA_ROOT/generated/trajectories/ifv-policy-v1" \
  --strict
```

Expected: exit 0 with a JSON summary.

### Task 24: Train in increasing policy scope

Do not combine these into one run:

1. **T1 Tool policy SFT:** train only ReAct action examples.
2. **T2 Planning + tool SFT:** add Planning examples after T1 frozen-real improvement.
3. **T3 Reflection SFT:** add Reflection only after task/gap false-positive rates are acceptable.
4. **T4 Optional Judgment SFT:** add final synthesis only if a trained Judgment improves cost without degrading basis validity.
5. **T5 Preference/RL:** use componentized process/result rewards only after SFT baselines are stable.

For each stage, require:

```text
same frozen-real cases
same tool/action budget
same evidence validator
same strict trace audit
baseline vs Student result and process metrics
```

Exact trainer/model configuration belongs in the follow-up Phase 5 plan because it depends on the selected Qwen model, tokenizer, distributed strategy, and audited dataset size. The present runtime plan deliberately stops before choosing those implementation details.

---

## Final Verification Checklist

Before marking this plan complete during execution:

- [ ] Every Phase 0 bug has a red test observed on gpu-13 before its fix.
- [ ] Legacy mode remains available through Phase 3 acceptance.
- [ ] Dynamic mode cannot create ungrounded tasks, facts, Findings, or evidence.
- [ ] Raw discoveries never count as substantive progress or verdict evidence.
- [ ] Reflection fires only after cumulative real actions 4, 8, 12, ...
- [ ] Task, fact, Reflection, and visual-attempt budgets are enforced deterministically.
- [ ] Exhausted ReInspect dependencies remain unresolved.
- [ ] `real`, `fake`, and `unverifiable` always match decisive VisualFact state.
- [ ] Every verdict basis traces `VisualFact → Finding → Evidence → successful tool call`.
- [ ] Legacy and v2 traces both render and audit.
- [ ] Every accepted real trace passes `--json --strict-scheduler` on gpu-13.
- [ ] Training artifacts are derived from canonical traces and contain no evaluator-private data.

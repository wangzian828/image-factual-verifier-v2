# VisualFact-Driven Single-Image Search Agent

**Date:** 2026-07-14  
**Status:** Implemented locally; gpu-13 real-canary validation pending
**Scope:** Active image-only, single-agent runtime specification. The search-control
corrections in `docs/superpowers/plans/2026-07-16-search-control-and-tool-correctness-audit.md`
supersede the earlier multi-decisive-fact and target-refresh design.

## 1. Goals

Given a single image, the agent must:

1. recover image-grounded entities, attributes, relations, internal-consistency candidates, OCR, and retrieval anchors;
2. use reverse-image search, Web search, browsing, and targeted visual tools to investigate high-value facts;
3. dynamically add bounded investigation tasks when evidence exposes meaningful gaps;
4. preserve exact evidence provenance and evidence-conditioned visual reinspection;
5. produce an auditable `real`, `fake`, or `unverifiable` verdict, with a precise `verdict_basis`.

The intended high-level pattern is aligned with mainstream search agents:

```text
Planning → ReAct tool use → periodic reflection → final synthesis
```

The system remains a **single agent**. It does not introduce a supervisor, sub-researchers, a general claim graph, online critic debates, or unrestricted graph-edit actions.

## 2. Non-goals

- Do not remove native Gemini Interactions function calling.
- Do not weaken source provenance, exact-span, artifact, source-independence, access-policy, or trace invariants.
- Do not let a model directly create evidence, modify tool outputs, or change final facts without evidence.
- Do not rely on a hard-coded catalogue of anomaly prompts such as “detect flying people.”
- Do not turn the system into a generic knowledge graph or multi-agent scheduler.
- Do not make image origin (`camera`, `generated`, `edited`) synonymous with factual verdict.

## 3. Core Principle

The image itself may not supply a strong initial external claim. Therefore the runtime must not begin by inventing a fixed event/location root claim and then search only for support.

Instead, it begins with image-grounded observations and maintains a small set of inspectable **VisualFacts**:

```text
VisualEntity → VisualFact → ResearchTask → Evidence/Finding → VisualFact status → Verdict
```

A `VisualFact` represents an attribute, relation, internal consistency condition, or explicit text proposition. It is not a generic graph node. A task can reference one or more facts, and a task may optionally record a single `parent_task_id` solely for provenance.

## 4. Runtime Architecture

```text
Image
  ↓
Bootstrap perception
  ├─ scene, entities, OCR, regions
  ├─ candidate VisualFacts
  └─ retrieval anchors
  ↓
Bootstrap reverse-image search
  └─ initial visual-neighbor discoveries
  ↓
Initial task planning
  └─ select high-value, evidence-routable ResearchTasks
  ↓
┌───────────── Single-Agent Investigation Loop ─────────────┐
│ ReAct: choose one current task and one tool               │
│   ↓                                                        │
│ Runtime: execute, validate, persist Discovery/Evidence/Failure │
│   ↓                                                        │
│ Reducer: update Finding candidates, facts, ReInspect state │
│   ↓                                                        │
│ Every four real tool actions: structured Reflection        │
│   └─ reorder/add bounded routes for the same core fact     │
└───────────────────────────────────────────────────────────┘
  ↓
Deterministic Coverage Audit
  ↓
LLM LedgerJudgment
  ↓
Deterministic validator
  ↓
real | fake | unverifiable + verdict_basis
```

### 4.1 Existing invariants retained

The new architecture retains these active-runtime properties:

- Gemini Interactions and native `function_call` / `function_result` chains;
- strict `success` / `error` tool-result contract;
- distinct `sources`, `evidence`, `discoveries`, and `failures` collections;
- exact fetched passages, offsets, hashes, retrieval times, source-family independence, injection checks, and source-access policy;
- evidence-conditioned ReInspect and visual-action provenance;
- canonical, redacted JSON traces;
- engineering failure remaining distinct from a factual `unverifiable` result.

## 5. Core Data Model

### 5.1 InvestigationBrief

`InvestigationBrief` is immutable and intentionally weak. It specifies the inquiry contract rather than an invented event claim.

```json
{
  "brief_id": "brief-001",
  "input_mode": "image_only",
  "objective": "Induce, investigate, and audit image-grounded factual targets using open-web and visual evidence.",
  "required_output": ["verdict_target", "verdict", "verdict_basis", "evidence", "unresolved_gaps"],
  "stop_policy": "coverage_or_bounded_unresolved"
}
```

It does not contain search queries, provisional attribution hypotheses, evidence states, or an ungrounded root event.

### 5.2 VisualEntity

A `VisualEntity` is an image-grounded object, text region, scene region, or later identified external entity. It has a stable ID, type, provenance, and a normalized region when it comes from the input image.

Examples: `person-1`, `ground-1`, `sign-1`, `building-1`, `event-candidate-2023`.

### 5.3 VisualFact

A `VisualFact` is the basic investigation unit. It is one of:

```text
attribute
relation
internal_consistency
text_claim
```

Example relation:

```json
{
  "fact_id": "vf-12",
  "kind": "relation",
  "subject_entity_id": "person-1",
  "predicate": "supported_by",
  "object_entity_id": "ground-1",
  "status": "candidate",
  "basis_ids": ["region-person-1", "region-ground-1"],
  "decision_relevance": "unknown"
}
```

Example provenance relation added after a qualified discovery:

```json
{
  "fact_id": "vf-14",
  "kind": "relation",
  "subject_entity_id": "image-1",
  "predicate": "depicts",
  "object_entity_id": "event-candidate-2023",
  "status": "active",
  "origin": {"type": "web_discovery", "origin_ids": ["d-17"]}
}
```

Valid states:

```text
candidate | active | supported | refuted | conflicted | blocked | exhausted | retired
```

`retired` requires a provenance-bearing reason: it must be superseded by a more precise fact or supported as irrelevant/duplicate. Facts are never silently deleted.

### 5.4 ResearchTask

A `ResearchTask` is a concrete work item for investigating one or more facts.

```json
{
  "task_id": "t3",
  "fact_ids": ["vf-14"],
  "question": "What is the earliest verifiable context in which this image appeared?",
  "purpose": "Resolve the image-to-event attribution relation.",
  "priority": 1,
  "status": "active",
  "parent_task_id": "t1",
  "origin_ids": ["d-17"],
  "suggested_tools": ["reverse_image_search", "text_search", "visit"],
  "suggested_queries": [],
  "attempt_count": 0
}
```

Task status is:

```text
pending | active | resolved | blocked | exhausted
```

Tasks are not physically removed. A task can be resolved only with qualified Finding IDs; it can be blocked only with actual Failure IDs or an explicit data-void explanation.

### 5.5 Discovery, Evidence, Finding

The existing ledger separation remains mandatory:

```text
Search/RIS result → Discovery only
Fetched page/visual observation → eligible Evidence
Qualified Evidence → Finding
Finding → update VisualFact status
```

A `Finding` is a concise evidence-backed interpretation for a task/fact. It must cite one or more real evidence IDs and never use a snippet, title, reverse-image candidate, or model memory as its sole basis.

```json
{
  "finding_id": "f7",
  "task_id": "t3",
  "fact_ids": ["vf-14"],
  "statement": "The image was published no later than October 2023 in coverage of event E.",
  "stance": "support",
  "evidence_ids": ["e18"],
  "source_family_ids": ["family-reuters"],
  "quality": "decisive"
}
```

### 5.6 VisualQuestion

The existing evidence-conditioned `VisualQuestion` remains. It binds a source discovery/evidence ID, a target image region, an expected property, and allowed visual tools. It has at most two real attempts.

A pending visual question prevents its dependent visual fact from becoming decisively supported/refuted. An exhausted visual question produces typed insufficiency. Critically, exhausted state must not let unrelated Web evidence re-close the same dependent fact; only an independent evidence path that does not require that observation may decide it.

## 6. Bootstrap and Initial Task Induction

### 6.1 Bootstrap perception

Bootstrap perception emits only low-commitment observations:

- entities and normalized regions;
- OCR text and text regions;
- attributes and spatial/semantic relations;
- internal-consistency candidates;
- retrieval anchors such as text, logo, landmark candidate, distinctive object, or scene pattern.

It may state that a relation is uncertain, but it must not directly state that a scene violates physics, is generated, belongs to a given event, or is factually false.

### 6.2 Initial route selection

There is no fixed bootstrap reverse-image call. Target Planning establishes one
image-grounded `CoreVerdictFact`, and ReAct selects the first route for its open
evidence gap. Reverse-image search is available as an explicit Lens or semantic
branch; its results are discoveries, not evidence.

### 6.3 Initial planning

Initial Planning receives entities, candidate facts, OCR, retrieval anchors, and
initial tasks. It selects one core fact and candidate evidence routes. Runtime code
validates the core and keeps at most four high-value, evidence-routable initial tasks.

Task induction is not based on a hard-coded anomaly taxonomy. It follows the source-grounded sequence:

```text
image observation / OCR / retrieval anchor
  → candidate VisualFact
  → high-value, checkable ResearchTask
```

A fact/task must be grounded in one or more image regions, OCR observations, or discoveries. Unsupported guesses such as “this may be Paris” are rejected.

## 7. Control Roles

### 7.1 ReAct: next action

The primary investigation agent chooses one active task and one concrete tool call. It may use task priority, findings, discoveries, failures, pending ReInspect, budget, and the compact working view.

ReAct cannot create/close tasks, activate facts, change the brief, set final verdicts, or mutate evidence.

### 7.2 Reducer: factual state persistence

The deterministic reducer validates observations and records discovery/evidence/failure state, task attempt statistics, finding candidates, source-family coverage, and ReInspect state. It does not independently determine strategy.

### 7.3 Structured Reflection: next-stage strategy

Reflection runs automatically every four real tool actions. It is an independent schema-constrained call, not a tool and not a multi-agent debate.

Its allowed output is a bounded task/fact delta:

```json
{
  "task_updates": [],
  "new_tasks": [],
  "recommended_next_task_ids": [],
  "remaining_gaps": [],
  "ready_to_finish": false
}
```

Reflection can identify gaps, create grounded tasks, and update task ordering. It
cannot change the CoreVerdictFact, create evidence, delete history, modify the brief,
cancel pending ReInspect, or write a verdict.

Every update is validated:

- a resolved task must cite Findings;
- a blocked task must cite Failures/data void;
- a new task must cite real origin IDs and serve a permitted fact/output purpose;
- duplicate semantic tasks are rejected;
- task/fact state cannot bypass a pending or exhausted visual requirement.

### 7.4 Coverage Audit: finish authority

The deterministic audit determines whether investigation continues, completes, saturates, or exhausts its budget. It does not create queries or plan routes.

An agent/Reflection finish suggestion triggers an immediate coverage check; it never exits directly to Judgment.

### 7.5 LedgerJudgment: constrained final synthesis

LLM `LedgerJudgment` is retained to align with mainstream search-agent final
synthesis. It reads only the core fact, eligible findings/evidence, final task states,
stop reason, and deterministic constraints.

The validator enforces exact IDs, valid evidence ownership, matching stance, valid typed insufficiency, and consistency with the resulting verdict. The LLM cannot add facts or bypass evidence rules.

## 8. Core Fact Ownership and Verdict

### 8.1 Ownership gate

A run owns one `CoreVerdictFact`. It is established only through deterministic
`reconcile_core_verdict_fact(...)`; Planning, Attribution, Reflection, and tools
cannot directly mutate ownership. Eligible roots are:

1. an explicit, OCR-grounded factual proposition in the image;
2. qualified RIS/Web evidence that supplies a competing source, time, place, or event attribution;
3. a central, localized entity/internal or entity-to-entity inconsistency with an executable route and a competing explanation;
4. an evidence-to-vision bridge that poses a localized visual fact capable of changing image interpretation.

The core needs valid origin IDs, connection to the image or an explicit pixel claim,
at least one executable route, and no semantic duplication. One already-resolved,
same-subject, atomic refinement may replace it. Optional title, creator, date,
platform, asset ID, and general visual-integrity metadata remain supporting.

Budgets:

```text
core verdict facts: exactly 1 after activation
core fact refinements: at most 1
new core facts per Reflection: 0
```

### 8.2 Fact-level evidence policy

The core fact is resolved from qualified evidence under the existing rules:

- compatible moderate/strong visual observation;
- one qualified official direct source; or
- two independent eligible non-UGC, non-risky source families.

Support and refutation both decisive yields `conflicted`. Open, blocked, exhausted, and conflicted facts are not supported.

### 8.3 Verdict aggregation

```text
Core fact refuted with required binding → fake
Core fact supported with required binding → real
Otherwise → unverifiable
```

Only the validated core fact participates. A background detail cannot independently
cause `fake`.

Every `fake` result includes `verdict_basis`, with at least one fact, supporting Finding/Evidence IDs, and a mechanism such as:

```text
reused_old_image
wrong_event
landmark_mismatch
impossible_causal_dynamics
inconsistent_lighting_or_reflection
synthetic_presented_as_documentary
```

`unverifiable` must expose fact-specific typed reasons such as absent decisive evidence, source conflict, access limitation, unreadable region, saturation, or budget exhaustion.

## 9. Budgets and Stopping

Initial defaults:

```text
MAX_TOOL_ACTIONS = 24
REFLECTION_INTERVAL = 4 real tool actions
MAX_REFLECTIONS = 6
INITIAL_TASKS_MAX = 4
TOTAL_TASKS_MAX = 12
NEW_TASKS_PER_REFLECTION_MAX = 3
CORE_VERDICT_FACTS = 1
CORE_FACT_REFINEMENTS_MAX = 1
VISUAL_QUESTION_ATTEMPTS_MAX = 2
```

Real tool actions include successful calls, valid error observations, access blocks, and empty-result calls. Planning, Reflection, audit, Judgment, protocol errors, and correction calls do not consume this action count.

The runtime distinguishes:

```text
Lead Gain: new candidate URL or RIS neighbor
Evidence Gain: new eligible evidence or independent source family
Decision Gain: valid finding, fact-state transition, resolved visual question, or material conflict
```

Only Evidence Gain or Decision Gain is substantive. Pure lead gain cannot reset saturation.

Coverage runs after every accepted action. A resolved core fact stops immediately.
With an unresolved core, no executable route or two consecutive action checkpoints
without qualified core progress ends the run as `information_saturated`. Other
terminal states are `verdict_determined` and `hard_budget_exhausted`.

## 10. Context Views

The system renders different compact contexts rather than replaying the full trace.

### ReAct working view

- current/high-priority tasks;
- relevant findings/discoveries;
- recent related tool results and failed routes;
- pending visual questions;
- remaining tool actions and checkpoint distance.

### Reflection global view

- all task one-line summaries;
- findings and fact states;
- source-family coverage;
- grouped failures;
- pending/exhausted visual questions;
- remaining budget.

### Judgment view

- the one core fact;
- eligible findings/evidence;
- final task states;
- typed stop/insufficiency state.

No view exposes full raw webpages except within the tool path that needs them. Solved-task history is compressed but retained in canonical trace storage.

## 11. Failure and Recovery Policy

- Tool-level exceptions and contract failures become typed error observations and are returned to the agent; they do not become evidence.
- A failed URL is route-local. Other sources/tasks remain investigable.
- Each task maintains compact route memory: queries, URLs/domains, source families, failures, and rejected discoveries.
- Exact repeated routes are rejected; revisiting a URL for a materially distinct immutable evidence goal can be allowed.
- Invalid Reflection gets one same-interaction correction. If still invalid, retain the old task list, record `reflection_failure`, and permit one further tool segment. Two consecutive Reflection failures are engineering failure, not `unverifiable`.
- Context overflows follow a deterministic compaction order: solved-task details,
  low-priority discoveries/context tasks, then only active core-gap tasks, pending
  visuals, critical findings, failure summaries, and budget. The brief, winning
  evidence IDs, and unresolved core fact are never dropped.

## 12. Migration Phases

### Phase 0: stabilize current runtime

- Fix the exhausted-ReInspect/web-evidence closure bug.
- Define substantive information gain so raw discoveries do not prevent saturation.
- Align active action-budget documentation/configuration.
- Add state consistency and regression tests.

### Phase 1: compatibility data layer

- Add `InvestigationBrief`, `VisualEntity`, `VisualFact`, `ResearchTask`, and `Finding` schemas.
- Adapt current `InvestigationQuestion` into an initial task projection.
- Extend canonical trace without changing the existing main loop.

### Phase 2: dynamic tasks and periodic reflection

- Execute Reflection every four real tool actions.
- Validate task updates and new routes serving open core EvidenceGaps.
- Make StageRunner task-aware.
- Replace fixed-question-only replanning.
- Make Coverage Audit task/fact-aware.

### Phase 3: VisualFact-driven Judgment

- Render Judgment from the core fact, findings, evidence, and final task state.
- Require `verdict_basis`.
- Retain deterministic judgment validation over the current core-fact ledger.

### Phase 4: trajectories and evaluation

- Export Planning, ReAct, Reflection, and Judgment training examples.
- Add token spans/loss masks and teacher trajectory scores.
- Establish frozen-real pilots and process metrics.

### Phase 5: single-agent student training

1. tool-action SFT;
2. Planning + tool-action SFT;
3. Reflection SFT;
4. optional final-synthesis SFT;
5. preference optimization/RL against result and process reward.

## 13. Server-Based Validation

All project validation runs on gpu-13, following `docs/operations/gpu13.md`. The local Windows checkout is the only source-editing location; code is committed and pushed locally, then the clean server checkout is fast-forwarded. Server source files are never edited directly.

Every project command runs through `scripts/server/run_gpu13.sh` in the isolated `ifv-agent` Conda environment. This guarantees the committed proxy/data-root configuration and the mandatory `OMP_NUM_THREADS=1` guard. Datasets, caches, traces, logs, and evaluation outputs remain under `IFV_DATA_ROOT=/gsdata/home/wza/image-factual-verifier-v2-data`, outside the Git checkout.

The validation ladder is:

1. focused contract and state-machine tests on gpu-13 after each implementation slice;
2. full orchestration and trace suite on gpu-13 before completing a phase;
3. the real Gemini Interactions probe after changes to model/protocol boundaries;
4. small frozen-real image-only regression runs with unique output directories;
5. `scripts/audit_real_trace.py --json --strict-scheduler` on every accepted real trace;
6. formal background evaluation only through the committed gpu-13 launcher, with explicit run IDs and concurrency controls.

Remote updates must use the committed updater/bootstrap workflow. A dirty server checkout blocks installation and validation. Credentials remain in the untracked server `.env` or process environment and never enter commands, Git, traces, or this specification.

## 14. Tests and Evaluation

### Required regressions

- RIS/Web evidence discovers an omitted old-image route; Reflection may create a
  bounded task for the existing core gap, but cannot change verdict ownership.
- Event/location/time attribution error produces evidence-grounded `fake`.
- Entity-internal or entity-relation inconsistency produces a `fake` only through qualified visual/external evidence.
- Raw new URLs cannot prevent saturation.
- Discoveries/snippets never become Findings or verdict evidence.
- Exhausted visual revisit cannot be bypassed by Web evidence.
- Weak/no anchors and an unresolved core fact yield `unverifiable`, not an invented
  target.
- Every final statement traces through `VisualFact → Finding → Evidence → tool call`.

### Metrics

Result metrics:

- three-class accuracy/macro-F1 and `unverifiable` calibration;
- fact-level support/refute accuracy;
- evidence/citation precision and source-independence compliance.

Process metrics:

- target/fact utility;
- necessary-gap recall;
- false task and false activation rate;
- valid Finding precision;
- evidence-to-vision bridge precision;
- ReInspect resolution rate;
- duplicate action and premature-finish rate;
- decisive evidence per tool action;
- over-investigation, cost, latency, and first-error attribution.

## 15. Acceptance Criteria

The architecture is accepted only when:

1. image-only bootstrap uses observations/anchors/RIS discoveries rather than an ungrounded fixed event root;
2. all dynamic tasks/facts have validated provenance and bounded growth;
3. core-fact ownership cannot be manipulated through ungrounded model prose;
4. evidence and visual reinspection gates remain machine-verifiable;
5. final `real`/`fake`/`unverifiable` is backed by fact-level evidence and a traceable `verdict_basis`;
6. runs saturate on lack of evidence/decision progress rather than endless new URLs;
7. canonical traces supply isolated Planning, ReAct, Reflection, and Judgment training samples without exposing evaluation gold.

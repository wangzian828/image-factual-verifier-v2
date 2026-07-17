# VisualFact Search Agent v3 Architecture

## 1. Supported runtime

The repository has one supported runtime:

```text
release: v0.3 image_only
public case: case_id + image_path + image_sha256
decision policy: reinspect-v2
```

`reinspect-v2` names the current v3 decision policy. It is not compatibility with an
older project version. The removed claim-mode runtime and `reinspect-v1` are not
accepted by the release adapter, public workflow, strict auditor, or real canary.

## 2. End-to-end control flow

```text
ImageOnlyRuntimeCase
  -> verify image SHA-256
  -> Gemini perceive_scene
  -> layered ocr_with_position (optional PP-OCR service, EasyOCR fallback)
  -> deterministic InvestigationBrief / VisualEntity / VisualFact bootstrap
  -> deterministic initial ResearchTasks (maximum 4)
  -> stateful Gemini main investigation chain
       Target Planning receives the original image once
       later policy stages inherit that image through previous_interaction_id
  -> model-driven target Planning grounded in image, visible facts, and OCR
  -> one stable CoreVerdictFact + bounded EvidenceGaps
  -> native Gemini Interactions ReAct
       model selects the first evidence route
       one accepted tool call per action turn
       deterministic observation reducer after every real action
       sparse Gemini Evidence Decision at material control boundaries
       optional one-time visual-slot refinement that preserves the core relation
       deterministic Coverage immediately after each Evidence Decision/action
       structured Reflection at actions 4, 8, 12, 16, 20, 24
  -> deterministic verdict and verdict_basis compiler
  -> constrained Gemini Judgment
  -> real | fake | unverifiable
```

The hard action budget is 24. Every accepted ReAct tool call counts as one action;
the first route is selected by Planning/ReAct and is not fixed to reverse-image
search. Structured output, Planning, Reflection, Judgment, rejected calls, and
protocol corrections do not count as tool actions.

The original image is not uploaded again for every stage. Target Planning creates
the one stored main-chain root with the image attached. ReAct, Evidence Decision,
Reflection, and Judgment continue from that root through
`previous_interaction_id`. Query Concept Extraction, Query Replan, webpage
extraction, OCR, and tool-internal VLM calls remain independent auxiliary
interactions. A tool-bearing segment may leave one pending `function_result`; the
next main-chain stage submits it together with a `user_input` step containing the
new structured runtime context.

## 3. Runtime input and isolation

`ImageOnlyRuntimeCase` has exactly:

```json
{
  "case_id": "case_...",
  "image_path": "/resolved/runtime/path/image.jpg",
  "image_sha256": "64 lowercase hex characters"
}
```

The release row remains relative and immutable; the adapter resolves it within
`runtime_input/`, prevents path escape, and the runtime re-hashes the image.

Before rollout, the evaluator may read the manifest, public protocols, public rows,
images, and an active source-access policy. It must not read evaluator-private gold.
Gold is loaded only after every rollout finishes, then joined by `case_id` for process
and classification evaluation.

## 4. Perception and bootstrap

Perception is a fixed two-tool schedule:

1. `perceive_scene`: a Gemini Interactions image request producing literal visible
   entities, normalized boxes, scene description, and image type.
2. `ocr_with_position`: optional PP-OCR-compatible service with EasyOCR fallback,
   producing accepted text, rejected low-confidence candidates, normalized geometry,
   and an artifact hash for the actual image or crop.

Gemini, not Codex, interprets benchmark pixels at runtime. OCR is an independent
observation and does not automatically become a decisive factual proposition.

`build_bootstrap_investigation()` deterministically creates:

- immutable `InvestigationBrief`;
- image/OCR-grounded `VisualEntity`;
- `RetrievalAnchor` records for visible text, logos, entities, and scene patterns;
- candidate `VisualFact` records of kind `attribute`, `relation`,
  `internal_consistency`, or `text_claim`;
- at most four evidence-routable `ResearchTask` records.

Stable IDs are hashes of immutable inputs. Bootstrap does not write Evidence, Finding,
fact resolution, or verdict.

## 5. ReAct and action reduction

`StageRunner` exposes the verification registry:

| Capability | Tools |
| --- | --- |
| Discovery/retrieval | `reverse_image_search`, `text_search`, `visit` |
| Focused visual work | `ocr_with_position`, `crop_and_inspect` |
| Comparison/consistency | `compare_with_reference`, `check_consistency`, `analyze_visual_anomalies` |
| Runtime context | `current_time` |

Gemini chooses the active task and tool. Each native `function_call` must contain a
known active task ID as `question_id`; deterministic code validates schemas, source
policy, semantic route duplicates across segments, budgets, and provider selection
before execution.

Before ReAct begins, Target Planning sees the original image and must establish one
atomic, externally checkable core proposition grounded in the image and pixel/OCR
facts. Bootstrap `appears_to_depict` prose
remains perception state and cannot own the verdict. If the bounded Planning
correction cannot produce a valid core, the runtime fails explicitly as an
engineering error.

One policy action maps to one bounded external operation: one text query, one page,
or one explicit Lens/semantic image-search branch. `crop_and_search` and
`count_objects` remain standalone diagnostics and are not exposed to the Agent loop.
Traces separately report policy actions and real provider subcalls.

Blocked duplicate or already-resolved route proposals are retained as route-control
warnings rather than executed tool actions. They do not invalidate a factual canary,
but any episode containing them fails the teacher-data quality gate.

Task Planning tools describe possible first hops. Once retrieval creates a
task-owned Discovery, the runtime derives the matching inspection permission:
candidate pages enable `visit`, and validated reference images enable
`compare_with_reference`. Dynamic schemas bind those tools to the remaining
Discovery URL instead of relying on the model to have predicted every follow-up
tool during Planning.

Candidate selection is bounded by retrieval batch rather than source class. Up to
four URLs from the latest uninspected batch are exposed for one model-selected
inspection. Source class remains evidence provenance and risk metadata; it does not
discard unknown or UGC leads before their content is checked.

Every real tool call is reduced into separate collections:

| Collection | Meaning |
| --- | --- |
| `discoveries` | Search/reverse-image leads. Never verdict evidence. |
| `evidence` | Exact fetched spans or successful visual observations with provenance. |
| `findings` | Task-owned factual support/refutation backed by runtime Evidence. |
| `failures` | Typed provider/tool/access/protocol failures. Never Evidence. |
| `tasks` / `facts` | Current investigation state with append-only provenance. |

Tool output must be a JSON object with `status=success|error`. Tool exceptions and
malformed output are serialized as errors and retained in the trace.

### Web Evidence

Eligible web Evidence requires:

- a fetched page rather than a search snippet;
- an existing selected passage and exact offsets;
- canonical URL, artifact SHA-256, and retrieval time;
- directness, stance, relevance, and `evidence_eligible=true`;
- no prompt-injection flags;
- a successful immutable `function_call_id`.

One official direct source can refute an exact event/place/identity slot. Supporting
a proposition about what the input image depicts additionally requires visual
binding. An exact same-capture match on an original official source can directly bind
the scene; a match on a non-original or unknown host also needs a fetched direct
source assertion. Generic pages about a logo, landmark, or entity cannot independently
prove that it is present in the input pixels.

### Visual Evidence

Eligible visual Evidence records the successful visual tool call, image region or
reference comparison, image/reference artifact hash, retrieval time, stance, and
source family. A neutral observation remains Evidence but does not resolve a fact.

General VLM consistency and anomaly opinions are diagnostics, not calibrated forensic
Evidence. A clean scan cannot support authenticity, and an anomaly opinion cannot by
itself refute authenticity.

When qualified support and refute evidence coexist, the reducer compares claim/scene
binding, source originality, directness, source risk, temporal alignment, and
independence. `conflicted` means more discriminating evidence is required; it is not
itself a final reason for `unverifiable`.

## 6. Semantic Evidence Decision

The observation reducer preserves fetched text, URLs, hashes, offsets, visual
comparison output, and tool-call provenance. It does not decide whether page text
semantically supports the active proposition.

A separate Gemini Evidence Decision runs sparsely:

- after directly inspected Evidence that may change the conclusion;
- after a same-capture binding is followed by direct page text;
- when multiple qualified independent Evidence rows accumulate;
- immediately before Reflection/Replan;
- immediately before returning unresolved or `unverifiable`.

It outputs:

```text
supported | refuted | conflicted | insufficient

none | text_sufficient | same_capture_helpful | same_capture_required
```

Reliable text can close ecological, geographic, temporal, and other world relations
without finding the exact source image. Same-capture Evidence is required only when
the decision depends on proving that a source assertion describes the exact input
pixels.

One `insufficient|conflicted` decision may narrow one unknown visible slot. For
example:

```text
orange-and-black butterflies occur in Antarctica
  -> monarch butterflies occur in Antarctica
```

The runtime validates that the refinement uses pixel/OCR facts and newly reviewed
Evidence, remains atomic, preserves the original relation and non-target slots, and
does not promote creator/title/date/platform metadata. The refinement budget is one.

## 7. Reflection and deterministic state transitions

Reflection runs after every four cumulative actions. It may:

- reprioritize existing tasks;
- add at most three grounded tasks serving open core EvidenceGaps;
- recommend next tasks;
- describe remaining gaps.

It may not create Evidence or Findings, modify the brief, delete history, write a
verdict, change the CoreVerdictFact, or resolve/block tasks without real
Finding/Failure IDs.

Global limits:

```text
MAX_TOOL_ACTIONS = 24
REFLECTION_INTERVAL = 4
MAX_REFLECTIONS = 6
INITIAL_TASKS_MAX = 4
TOTAL_TASKS_MAX = 12
NEW_TASKS_PER_REFLECTION_MAX = 3
CORE_VERDICT_FACTS = 1
CORE_FACT_REFINEMENTS_MAX = 1
```

Two consecutive invalid Reflections are an engineering failure.

## 8. Coverage and Judgment

Coverage audits the one CoreVerdictFact as:

```text
supported | refuted | conflicted | blocked | exhausted | unresolved
```

Coverage consumes the latest semantic Evidence Decision for the active fact.
Qualified gain is a change in its status, selected winning Evidence, required
binding, conflict state, or accepted visual-slot refinement. New IDs, discovery URLs,
optional metadata, and task churn do not count.

Stop states:

- `verdict_determined`: the core fact is supported or refuted and every required gap
  is resolved;
- `information_saturated`: no executable core route remains, or two consecutive
  action checkpoints produced no qualified gain;
- `hard_budget_exhausted`: 24 tool actions used.

The verdict compiler is deterministic:

- core fact refuted with required binding -> `fake`;
- core fact supported with required binding -> `real`;
- otherwise -> `unverifiable` with fact-specific gaps.

It compiles `VerdictBasis` from the smallest sufficient winning
fact/Finding/Evidence chain. A fake basis contains one strongest decisive refutation.
The final Gemini `ImageOnlyJudgment` continues the stored main investigation chain
and must return the same verdict, policy, ID sets, and unresolved gaps. Model prose
cannot add facts or citations.

The strict chain is:

```text
VisualFact -> Finding -> Evidence -> successful tool call
```

## 9. Failure contract

Engineering failures include:

- missing credentials or required tools;
- image hash mismatch;
- Gemini transport exhaustion or non-retryable HTTP errors;
- malformed Interactions envelopes or structured output;
- invalid tool arguments after correction budget;
- required perception/OCR failure;
- all image-only investigation tool calls failing;
- two consecutive invalid Reflections;
- timeout or impossible state transition.

Engineering failures end before factual Judgment. Evaluation writes diagnostics and an
error trace, but no row to `predictions.jsonl`. They are never converted to
`unverifiable`.

## 10. Canonical trace

The canonical trace stores:

- three-field runtime case, input mode, and decision policy;
- merged perception;
- full `ImageOnlyInvestigationState`;
- all native model/tool steps and interaction-chain metadata;
- policy request/action snapshots;
- timings, token/call accounting, provider health, termination, and errors;
- deterministic verdict basis and constrained Judgment.

Credentials and signed authentication parameters are redacted before persistence.
`src.trace_viewer` renders a v3-only diagnostic HTML view; JSON remains canonical.

## 11. Evaluation and training handoff

`src.eval.run_eval` emits:

```text
run_manifest.json
predictions.jsonl
run_results.jsonl
process_metrics.jsonl
reference_chain_metrics.jsonl
trajectory_scores.jsonl
policy_trajectories.jsonl
summary.json
traces/*.json
```

Classification remains data-pipeline-owned. Process scoring is post-rollout and
componentized; it reports fact alignment, actual visual binding, verdict-basis
alignment/minimality, conflict resolution, semantic duplicates,
post-determination actions, low-value actions, cost, and first error.

Reference-chain recovery is emitted separately with four metrics. It measures only
the private chain from decisive fact through visual binding and acceptable evidence.
URL, span, snapshot, and SHA identity remain data-pipeline audit properties and are
not scored as Agent capabilities. Conservative matching recognizes same-source page
versions and official same-capture image assets. An optional LLM judge is restricted
to qualified unresolved edges and cannot invent new evaluation targets. Off-chain
material is neutral unless selected into the final verdict basis.

`ifv-policy-v1` exports Target Planning, ReAct, Evidence Decision, Reflection, and
Judgment request/action boundaries. Bootstrap remains deterministic. Discovery stays
separate from Evidence. Public title, creator, date, platform, and asset metadata may
remain retrieval context or supporting trace records, but cannot replace the core
fact. Only the one bounded visual-slot refinement described above may do so.

Policy snapshots preserve the text, schemas, interaction IDs, and a
`runtime_image=true` media reference. They never persist the original image base64
inside the trace or tokenize it as text. The public runtime case and perception
trajectory retain the corresponding `image_path` and `image_sha256`.

Initial Planning may propose alternatives, but the runtime selects one core factual
relation. Other facets remain supporting. For example, a screenshot can yield:

```text
source record match
visible manipulation integrity
```

The first binds visible account/text/date/thread details to an original or archived
record. The second checks only visible editing and layout anomalies. This is a
model-proposed plan grounded in current facts and anchors, not a hard-coded screenshot
pipeline. Other images may produce different targets and tool routes.

Only episodes that pass the explicit trajectory-quality gate are included in the
policy dataset. Excluded episodes and reasons are preserved as metadata. The default
tokenizer adapter is `utf8-byte-v1`; model tokenizers can be injected.
Fatal-boundary actions receive zero loss mask. Dataset export groups episodes sharing a
source family into one split and audits leakage, references, duplicates, masks, and
split isolation.

Training remains disabled in `configs/training/ifv_policy_v1.yaml`.

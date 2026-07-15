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
  -> EasyOCR ocr_with_position
  -> deterministic InvestigationBrief / VisualEntity / VisualFact bootstrap
  -> deterministic initial ResearchTasks (maximum 4)
  -> smallest central decisive-fact set
  -> initial reverse-image search (Discovery only)
  -> native Gemini Interactions ReAct
       one accepted tool call per action turn
       deterministic observation reducer after every real action
       structured Reflection at actions 4, 8, 12, 16, 20, 24
       deterministic evidence adjudication after every action
       Coverage after Reflection and when a verdict becomes determined
  -> deterministic verdict and verdict_basis compiler
  -> constrained Gemini Judgment
  -> real | fake | unverifiable
```

The hard action budget is 24. The initial reverse-image search counts as a real action.
Structured output, Reflection, Judgment, rejected calls, and protocol corrections do
not count as tool actions.

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
2. `ocr_with_position`: EasyOCR producing text and normalized geometry.

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
| Focused visual work | `ocr_with_position`, `crop_and_search`, `crop_and_inspect`, `count_objects` |
| Comparison/consistency | `compare_with_reference`, `check_consistency`, `analyze_visual_anomalies` |
| Runtime context | `current_time` |

Gemini chooses the active task and tool. Each native `function_call` must contain a
known active task ID as `question_id`; deterministic code validates schemas, source
policy, semantic route duplicates across segments, budgets, and provider selection
before execution.

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
binding. A full scene proposition needs both a same-capture comparison and a fetched
direct source assertion. Generic pages about a logo, landmark, or entity cannot
independently prove that it is present in the input pixels.

### Visual Evidence

Eligible visual Evidence records the successful visual tool call, image region or
reference comparison, image/reference artifact hash, retrieval time, stance, and
source family. A neutral observation remains Evidence but does not resolve a fact.

When qualified support and refute evidence coexist, the reducer compares claim/scene
binding, source originality, directness, source risk, temporal alignment, and
independence. `conflicted` means more discriminating evidence is required; it is not
itself a final reason for `unverifiable`.

## 6. Reflection and deterministic state transitions

Reflection runs after every four cumulative actions. It may:

- reprioritize existing tasks;
- add at most three grounded tasks;
- propose at most two new decisive facts;
- recommend next tasks;
- describe remaining gaps.

It may not create Evidence or Findings, modify the brief, delete history, write a
verdict, or resolve/block tasks without real Finding/Failure IDs.

Global limits:

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

Two consecutive invalid Reflections are an engineering failure.

## 7. Coverage and Judgment

Coverage audits every active decisive fact as:

```text
supported | refuted | conflicted | blocked | exhausted | unresolved
```

Substantive gain is a new Evidence ID or a decisive fact-status change. New discovery
URLs and task churn do not count.

Stop states:

- `verdict_determined`: a decisive refutation has survived conflict adjudication, so
  unrelated supporting facts cannot change the `fake` result;
- `coverage_complete`: every decisive fact is supported or refuted;
- `information_saturated`: two consecutive Reflection intervals without substantive
  gain and no unattempted priority-1 task;
- `hard_budget_exhausted`: 24 tool actions used.

The verdict compiler is deterministic:

- any decisive refuted fact -> `fake`;
- all decisive facts supported -> `real`;
- otherwise -> `unverifiable` with fact-specific gaps.

It compiles `VerdictBasis` from the smallest sufficient winning
fact/Finding/Evidence chain. A fake basis contains one strongest decisive refutation.
The final Gemini `ImageOnlyJudgment` must return the same verdict, policy, ID sets,
and unresolved gaps. Model prose cannot add facts or citations.

The strict chain is:

```text
VisualFact -> Finding -> Evidence -> successful tool call
```

## 8. Failure contract

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

## 9. Canonical trace

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

## 10. Evaluation and training handoff

`src.eval.run_eval` emits:

```text
run_manifest.json
predictions.jsonl
run_results.jsonl
process_metrics.jsonl
trajectory_scores.jsonl
policy_trajectories.jsonl
summary.json
traces/*.json
```

Classification remains data-pipeline-owned. Process scoring is post-rollout and
componentized; it reports fact alignment, acceptable-evidence hits, citation
precision, actual visual binding, verdict-basis alignment/minimality, conflict
resolution, semantic duplicates, post-determination actions, low-value actions, cost,
and first error.

`ifv-policy-v1` exports actual ReAct, Reflection, and Judgment request/action
boundaries. Bootstrap is deterministic, so no fictional Planning example is emitted.
The schema reserves `planning` for a future real policy stage.

Only episodes that pass the explicit trajectory-quality gate are included in the
policy dataset. Excluded episodes and reasons are preserved as metadata. The default
tokenizer adapter is `utf8-byte-v1`; model tokenizers can be injected.
Fatal-boundary actions receive zero loss mask. Dataset export groups episodes sharing a
source family into one split and audits leakage, references, duplicates, masks, and
split isolation.

Training remains disabled in `configs/training/ifv_policy_v1.yaml`.

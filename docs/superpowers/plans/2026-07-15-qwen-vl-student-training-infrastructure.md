# Qwen-VL Student Training Infrastructure Plan

**Date:** 2026-07-15  
**Status:** proposed  
**Runtime project:** `image-factual-verifier-v3`  
**Training project:** new, separate `image-factual-verifier-training` repository  
**Teacher:** Gemini  
**Student:** Qwen-VL, eventually owning perception, ReAct, Reflection, and Judgment

## 1. Objective

Build a reproducible SFT and RL training system that distills accepted Gemini
trajectories into a Qwen-VL student without changing or destabilizing the Gemini
runtime.

The final student must be able to replace Gemini for:

```text
image perception
ReAct tool selection
Reflection
Judgment
```

The deterministic runtime remains responsible for:

```text
input validation
VisualFact/task bootstrap
tool execution
Discovery/Evidence/Finding/Failure reduction
support/refute conflict adjudication
Coverage
verdict_basis compilation
trace persistence
evaluation
```

Qwen-VL is not trained to replace deterministic state transitions or private
evaluation.

## 2. Architecture decision

### 2.1 Training is a separate project

Do not install training frameworks into the v3 runtime repository or its
`ifv-agent` Conda environment.

Create:

```text
Windows:
D:\image-factual-verifier-training

gpu-13:
/gs/home/wza/projects/image-factual-verifier-training
```

Reasons:

- ms-swift, DeepSpeed, vLLM, Transformers, CUDA extensions, and RL dependencies have
  a much larger and less stable dependency surface than the runtime;
- Qwen3.5 requires a newer isolated stack than the current Gemini runtime;
- training experiments generate checkpoints, optimizer states, rollout caches, and
  logs that do not belong in the Agent source tree;
- Gemini must remain runnable from the accepted `ifv-agent` environment throughout
  Qwen development;
- a failed training upgrade must not change runtime provider behavior.

The runtime repository keeps only:

- provider-neutral state/tool protocols;
- Gemini and Qwen inference adapters;
- canonical trace and dataset exporters;
- checkpoint/serving consumer contracts;
- Gemini/Qwen evaluation and regression commands.

### 2.2 Reuse mature frameworks

Primary framework:

```text
ms-swift
```

Use it for:

- multimodal SFT;
- LoRA/QLoRA and, only if justified, full-parameter training;
- Qwen chat template and processor integration;
- Agent/tool-call training data;
- GRPO;
- custom multi-turn tool environments;
- distributed launch through DeepSpeed or Megatron-SWIFT.

Do not build a custom trainer before ms-swift has been proven insufficient.

Serving/rollout engines:

```text
Qwen3-VL baseline -> LMDeploy or vLLM compatibility probe
Qwen3.5 candidate -> vLLM first; Transformers only for correctness probes
```

`veRL` remains a deferred alternative for large-scale online RL. It is not part of
the first implementation because introducing two RL frameworks before the reward and
environment contracts are stable would add unnecessary complexity.

## 3. Gemini/Qwen compatibility contract

Gemini and Qwen must coexist as explicit provider profiles:

```text
teacher-gemini
student-qwen-local
student-qwen-api
```

Rules:

1. Gemini continues to use native Interactions and its existing environment.
2. Qwen uses an independent OpenAI-compatible local endpoint or DashScope endpoint.
3. Provider selection is explicit in run configuration and run manifests.
4. There is no automatic provider fallback after an error.
5. Both providers consume the same public runtime case and deterministic state
   schemas.
6. Both providers emit the same canonical policy actions, trace objects, and verdict
   contract.
7. Provider-specific wire messages are adapters; they are not training targets.
8. Every Qwen change must run the unchanged Gemini canary before acceptance.

The runtime must be able to execute:

```text
Gemini perception + Gemini policy
Qwen perception + Qwen policy
Qwen perception + Gemini policy       # migration diagnostic only
Gemini perception + Qwen policy       # migration diagnostic only
```

The mixed profiles are diagnostic. The intended final profile is Qwen for both
perception and policy.

## 4. Current hardware and starting assets

Verified gpu-13 resources:

```text
8 x NVIDIA A100-SXM4-40GB
128 CPU threads
377 GiB RAM
/gsdata: approximately 393 TiB free at audit time
```

Existing local model directories:

```text
/gsdata/home/wza/models/Qwen2.5-VL-7B-Instruct
/gsdata/home/wza/models/Qwen3-VL-8B-Thinking
```

Current runtime assets:

- Qwen/DashScope OpenAI-compatible LLM credentials and base-URL resolution;
- Qwen VLM client;
- local `lmdeploy` provider profile;
- Qwen3-VL-8B-Thinking default local model path;
- canonical ReAct/Reflection/Judgment policy examples;
- loss masks, split-safe dataset export, quality gates, and strict audits.

Missing assets:

- perception-stage training examples;
- Qwen processor/tokenizer-aligned training dataset;
- training repository and environments;
- SFT trainer configuration;
- checkpoint and adapter manifests;
- local Qwen protocol conformance suite;
- online tool-RL environment;
- private reward service;
- RL replay and rollout audit.

## 5. Model selection

### 5.1 Stable baseline

The non-blocking baseline is:

```text
Qwen3-VL-8B-Thinking
```

It is already present on gpu-13 and referenced by the existing local serving adapter.
The serving path still requires a real protocol probe. The checkpoint must remain the
non-blocking baseline even if the Qwen3.5 experiment is rejected.

Before SFT, compare the existing Thinking checkpoint against the corresponding
Instruct checkpoint if available. Freeze one Qwen3-VL baseline; do not train multiple
variants in parallel during infrastructure bring-up.

### 5.2 Qwen3.5 candidate

Qwen3.5 is allowed to become the primary student only after passing the gate below.
Start with a dense model that fits the current eight-A100 node; prefer a 9B-class
checkpoint when available, with a 4B-class checkpoint as an infrastructure smoke-test
fallback.

Qwen3.5 is not assumed to be compatible merely because weights can be loaded.

Required gate:

1. image input and OCR-sensitive JSON perception output;
2. provider-neutral structured Reflection/Judgment output;
3. one-tool-call-per-turn behavior through an OpenAI-compatible server;
4. tool result continuation across at least eight actions;
5. model processor and chat template round-trip;
6. 100-step multimodal LoRA SFT without NaN/OOM;
7. checkpoint save, resume, merge/export, and reload;
8. 20-step multi-turn GRPO smoke run with a deterministic mock tool environment,
   including a non-zero multimodal `image_seqlen`/image-token audit;
9. real two-case Qwen canary with strict trace audit;
10. unchanged Gemini canary after the Qwen environment is installed.

Decision:

```text
all gates pass     -> Qwen3.5 may become the primary student
any critical fail  -> continue with Qwen3-VL; record Qwen3.5 as deferred
```

Known planning constraint: Qwen3.5 training currently requires a newer isolated
Python/Transformers/vLLM stack, and LMDeploy TurboMind cannot be assumed to support
its vision encoder. This is why Qwen3.5 cannot share the Gemini runtime environment.

### 5.3 Qwen3.5 maturity judgment

Current framework evidence is sufficient to treat Qwen3.5 as a serious candidate:

```text
multimodal SFT     usable through ms-swift
LoRA/full training usable through ms-swift
GRPO               implemented, but requires a local smoke gate
Agent/tool data     supported by ms-swift templates
serving             vLLM-first; LMDeploy vision support is not assumed
```

Therefore Qwen3.5 is neither rejected nor selected by documentation alone. It becomes
the primary student only through the measured gate in section 5.2.

## 6. Environment topology

Keep three isolated environments:

```text
ifv-agent
  accepted Gemini runtime
  Python 3.11
  never receives ms-swift/DeepSpeed/vLLM training dependencies

ifv-qwen-train
  Python 3.12
  ms-swift source install
  PyTorch, Transformers, DeepSpeed/Megatron, training vLLM dependencies

ifv-qwen-serve
  serving-only dependencies
  LMDeploy for accepted Qwen3-VL profile
  vLLM for Qwen3.5 compatibility profile
```

Every environment gets an explicit lock/manifest:

```text
environment name
Python version
CUDA/PyTorch versions
package lock hash
framework commit/version
model and processor revisions
creation timestamp
```

No training package may be installed into `ifv-agent`.

## 7. Repository and data flow

```text
image-factual-verifier-data-pipeline
  -> immutable benchmark/training releases

image-factual-verifier-v3 + Gemini
  -> accepted teacher traces
  -> perception examples
  -> policy examples
  -> teacher/reward diagnostics

image-factual-verifier-training
  -> model-specific preprocessing
  -> SFT / RL
  -> checkpoint and adapter manifests
  -> serving export

Qwen local serving endpoint
  -> image-factual-verifier-v3
  -> identical runtime evaluation
```

Repositories communicate through files and schemas. They do not import each other's
Python packages.

## 8. Shared machine-readable contracts

### 8.1 Teacher dataset release

The runtime exports a versioned training release containing:

```text
manifest.json
episodes.jsonl
perception.jsonl
react.jsonl
reflection.jsonl
judgment.jsonl
excluded_episodes.jsonl
assets/sha256/<image>
SHA256SUMS
```

Every row includes:

```text
dataset_version
episode_id
case_id
stage
provider/model teacher identity
runtime commit
public image reference
model-visible messages
expected assistant action
runtime object references
training eligibility
loss-mask policy
```

Evaluator-private gold, reference evidence, scorer output, and source-access policy
must not enter model-visible fields.

### 8.2 Model-specific preprocessing

The training repository converts provider-neutral examples into Qwen messages using
the exact Qwen processor and chat template.

Important:

- `utf8-byte-v1` token IDs are audit fixtures, not Qwen training tokens;
- Qwen tokenization happens in the training repository;
- image placeholders, tool schemas, assistant spans, and loss masks are verified
  after the model template is applied;
- teacher reasoning/thought text is not a target; only canonical visible stage output
  receives loss, and any Qwen thinking span is disabled or masked;
- pretokenized artifacts record processor and model revisions;
- a tokenizer/template change creates a new derived dataset version.

### 8.3 Checkpoint manifest

Every saved checkpoint release contains:

```text
base model ID and revision
processor/tokenizer revision
training dataset manifest hash
framework and environment versions
training method: LoRA / QLoRA / full
adapter configuration
global step and optimizer state availability
reward version when applicable
evaluation run IDs
artifact SHA-256 values
```

### 8.4 Serving profile

The training project emits a serving profile consumed by runtime operations:

```text
profile_id
model path
engine
base URL
wire API
tool-call parser/template
multimodal limits
context length
tensor parallel size
dtype/quantization
health probe
checkpoint manifest hash
```

Credentials and machine-specific secrets are never stored in the profile.

## 9. Dataset tracks

### 9.1 Perception SFT

Input:

```text
image + fixed perception instruction
```

Target:

```text
PerceptionReport
```

The target contains scene description, visual entities, positioned text, retrieval
anchors, and uncertainty. Gemini teacher output is accepted only after schema
validation and trace-quality review.

### 9.2 ReAct tool-action SFT

Input:

```text
immutable brief
current VisualFacts/tasks
tool history and compact route memory
available tool schemas
```

Target:

```text
exactly one canonical tool action
```

Only the assistant action is trained. Tool results remain environment observations.

### 9.3 Reflection SFT

Input:

```text
current investigation state
coverage gaps
action and reflection budget
```

Target:

```text
bounded Reflection delta
```

The target may reprioritize tasks, propose new tasks, or activate/retire facts within
the runtime schema. It may not manufacture Evidence, Findings, or verdicts.

### 9.4 Judgment SFT

Input:

```text
deterministically compiled verdict and verdict_basis constraints
```

Target:

```text
ImageOnlyJudgment matching those constraints
```

Judgment SFT is lower priority than perception and tool policy because the runtime
already determines the allowed verdict and basis.

### 9.5 Mixed curriculum

After stage-specific smoke tests, train one Qwen-VL checkpoint with explicit stage
tags and controlled sampling ratios.

Initial curriculum:

```text
perception  25%
react       50%
reflection  20%
judgment     5%
```

Ratios are configuration, not a permanent contract. Change them only from observed
held-out failure rates.

There is no Planning SFT stage in the current runtime. Bootstrap is deterministic,
and the exporter intentionally does not fabricate Planning targets. A learned planner
would require a new runtime stage, schema, trace boundary, and dataset version before
it can enter training.

## 10. SFT implementation

### S0. Framework and model probe

- [ ] Create the training repository.
- [ ] Bootstrap `ifv-qwen-train` and `ifv-qwen-serve`.
- [ ] Record gpu-13 hardware and environment manifests.
- [ ] Run Qwen3-VL and Qwen3.5 model/processor probes.
- [ ] Run tool-call and structured-output conformance probes.
- [ ] Freeze the initial student checkpoint.

### S1. Dataset adapters

- [ ] Add perception export to the runtime.
- [ ] Version the teacher dataset release contract.
- [ ] Implement Qwen processor/chat-template conversion.
- [ ] Align assistant spans and loss masks after tokenization.
- [ ] Add image/token length and truncation audits.
- [ ] Add deterministic byte-identical dataset rebuild tests.

### S2. LoRA smoke training

- [ ] Run 100 steps on a tiny reviewed dataset.
- [ ] Verify loss decreases and remains finite.
- [ ] Save and resume at least once.
- [ ] Merge/export or load the adapter through the serving engine.
- [ ] Execute one full runtime trace with the trained adapter.
- [ ] Verify Gemini still passes its unchanged canary.

### S3. First real SFT

- [ ] Train on accepted teacher episodes only.
- [ ] Keep train/validation/test source-family isolated.
- [ ] Select checkpoint by held-out runtime behavior, not training loss alone.
- [ ] Run stage-level and end-to-end evaluations.
- [ ] Preserve every experiment manifest and failure reason.

Initial strategy:

```text
LoRA first
full-parameter training only after data scale and LoRA ceiling are demonstrated
```

Do not use the current two-episode dataset as evidence that training works; it is only
an infrastructure fixture.

## 11. RL design

RL begins only after one SFT checkpoint passes real multi-case runtime evaluation.

Primary algorithm:

```text
multi-turn GRPO through ms-swift
```

The Agent environment is implemented as a custom multi-turn scheduler:

```text
model action
  -> runtime protocol validation
  -> real or controlled tool execution
  -> deterministic state reduction
  -> next model observation
  -> terminal verdict/error
```

The environment, not the model, owns tools and state.

### 11.1 Private reward service

Reward is computed after rollout and never added to model context.

Reward components:

```text
primary:
  factual result correctness

process:
  fact recovery
  evidence-chain recovery
  visual bridge completion
  grounded Finding validity
  verdict-basis alignment/minimality
  correct conflict resolution
  calibrated stopping

penalties:
  protocol-invalid action
  duplicate route
  low-value action
  action after determination
  engineering failure
  normalized tool/call cost
```

URL, snapshot, exact-span, and artifact-hash reproduction are not reward components.

Classification reward and process reward remain separately logged. The first RL
version may use a weighted scalar for optimization, but every component must remain
visible and independently auditable.

### 11.2 RL phases

#### R0. Reward replay

- [ ] Replay reward on frozen Gemini and Qwen traces.
- [ ] Confirm correct efficient traces outrank incorrect or wasteful traces.
- [ ] Test reward invariance to irrelevant off-basis evidence.
- [ ] Freeze `reward_version`.

#### R1. Deterministic mock environment

- [ ] Run 20-step GRPO smoke training with mocked deterministic tools.
- [ ] Verify multi-turn message construction and token masks.
- [ ] Verify checkpoint/resume and rollout audit.

#### R2. Cached-tool environment

- [ ] Run against frozen search/visit/reference-comparison responses.
- [ ] Remove external network variance from early RL.
- [ ] Check for reward hacking and protocol gaming.

#### R3. Controlled live-tool environment

- [ ] Enable bounded real tools on a training-only case split.
- [ ] Keep evaluator gold private.
- [ ] Rate-limit and cache expensive calls.
- [ ] Require strict trace audit for every sampled rollout batch.

#### R4. Student acceptance

- [ ] Compare SFT and RL checkpoints on the same held-out release.
- [ ] Require result quality not to regress.
- [ ] Require chain recovery and efficiency gains to be independently visible.
- [ ] Run Gemini teacher regression in parallel.

## 12. Evaluation matrix

Every accepted checkpoint is evaluated at four levels:

```text
stage:
  perception schema/grounding
  ReAct tool action validity
  Reflection delta validity
  Judgment constraint adherence

protocol:
  one tool call per turn
  tool result continuation
  route deduplication
  bounded stopping

factual:
  classification scorer
  slices and confusion matrix

process:
  reference-chain recovery
  visual bridge
  basis minimality
  conflict resolution
  cost and latency
```

Required comparison profiles:

```text
Gemini teacher
Qwen base
Qwen SFT
Qwen SFT + RL
```

## 13. Operational layout on gpu-13

All large artifacts remain under:

```text
/gsdata/home/wza/image-factual-verifier-v2-data/training/
  datasets/
  derived/
  checkpoints/
  exports/
  rollouts/
  rewards/
  logs/
  caches/
```

Source repositories remain under `/gs/home/wza/projects`.

Training launchers must:

- require an explicit experiment ID;
- reject a non-empty output directory;
- record git commits and environment manifests;
- set `OMP_NUM_THREADS=1`;
- preserve stdout/stderr under the data root;
- write PID files outside Git;
- support safe resume from an explicit checkpoint;
- never edit runtime or data-pipeline checkouts.

The Qwen serving endpoint uses a dedicated loopback port and process. Starting or
stopping it must not affect Gemini API access, Gemini environment variables, or the
Jupyter control path.

## 14. Acceptance gates

### Infrastructure complete

- [ ] Training repository exists and is independently installable.
- [ ] Gemini runtime environment hash is unchanged.
- [ ] Qwen train and serve environments are reproducible.
- [ ] Qwen3-VL baseline serves image, text, structured output, and tool calls.
- [ ] Qwen3.5 decision gate is recorded.
- [ ] Dataset/checkpoint/serving contracts have consumer and producer fixtures.

### SFT complete

- [ ] Real perception and policy datasets are large enough for a meaningful split.
- [ ] LoRA training, resume, export, serving, and runtime evaluation pass.
- [ ] Qwen SFT improves held-out stage metrics over the base model.
- [ ] Qwen SFT completes real end-to-end traces without Gemini assistance.
- [ ] Gemini teacher remains accepted.

### RL complete

- [ ] Reward replay is manually audited.
- [ ] Multi-turn GRPO completes without protocol corruption.
- [ ] No private gold enters model context.
- [ ] RL improves at least one target process metric without factual regression.
- [ ] Reward hacking checks pass.
- [ ] Qwen SFT+RL and Gemini canaries are both reproducible.

## 15. Immediate implementation order

1. Create the separate training repository and environment manifests.
2. Add provider-neutral perception-example export to the runtime.
3. Freeze the teacher dataset release schema.
4. Probe Qwen3-VL-8B-Thinking through LMDeploy and vLLM.
5. Probe Qwen3.5 9B/4B through vLLM and ms-swift.
6. Record the Qwen3.5 go/defer decision.
7. Implement Qwen tokenization and SFT dataset adapters.
8. Run LoRA smoke training and serve the adapter.
9. Add dual Gemini/Qwen runtime canaries.
10. Build reward replay before enabling online RL.

## 16. Explicit non-goals

- Do not train inside the runtime repository.
- Do not modify the Gemini Interactions protocol to resemble Qwen.
- Do not silently fall back from Qwen to Gemini or vice versa.
- Do not train on evaluator-private gold or benchmark-only evidence references.
- Do not start online RL from the base model.
- Do not introduce a second RL framework before ms-swift limitations are observed.
- Do not treat two group-001 episodes as a training corpus.
- Do not merge checkpoints into the runtime Git repository.

## 17. Primary framework references

- Qwen3-VL repository and ms-swift fine-tuning examples:  
  `https://github.com/QwenLM/Qwen3-VL`
- ms-swift Qwen3-VL best practices:  
  `https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html`
- ms-swift Qwen3.5 best practices:  
  `https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_5-Best-Practice.html`
- ms-swift Agent training:  
  `https://swift.readthedocs.io/en/latest/Instruction/Agent-training.html`
- ms-swift custom multi-turn GRPO:  
  `https://swift.readthedocs.io/en/latest/Instruction/GRPO/DeveloperGuide/multi_turn.html`
- LMDeploy Qwen3-VL support:  
  `https://lmdeploy.readthedocs.io/en/latest/multi_modal/qwen3-vl.html`
- LMDeploy Qwen3.5 constraints:  
  `https://lmdeploy.readthedocs.io/en/latest/multi_modal/qwen3_5.html`

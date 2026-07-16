# Qwen Full-Parameter Training and Canary Rollout Plan

**Date:** 2026-07-16  
**Status:** active; blocked from Qwen execution until three runtime canaries pass  
**Runtime repository:** `image-factual-verifier-v3`  
**Training repository:** separate `image-factual-verifier-training`  
**Teacher:** Gemini  
**Student:** Qwen, ultimately replacing Gemini for perception and policy

## 1. Target outcome

First finish the Gemini runtime acceptance gate on three selected cases from the v4
20-case Development Preview. Each case must have the correct verdict, a visible
relation as its core target, a usable construction-aligned evidence chain, timely
stopping, and a passing strict trace audit. Only after all three pass, deploy one
approximately 8B vision-language Qwen model on gpu-13, prove that it can be trained
with all model parameters enabled, and connect it to the deterministic Agent runtime.

This work must not modify the data-pipeline repository, the v4 release, or the
accepted `ifv-agent` Gemini environment.

## 2. Model decision

Primary candidate:

```text
Qwen/Qwen3-VL-8B-Instruct
```

Existing correctness/serving fallback:

```text
Qwen/Qwen3-VL-8B-Thinking
```

Qwen3.5 is not the initial deployment target. It may replace Qwen3-VL later only if
an approximately 8B/9B checkpoint is explicitly verified as a vision-language model
and passes the same training and serving gates. Do not substitute a text-only Qwen
checkpoint.

Use Qwen3-VL-8B-Instruct if all of the following pass on gpu-13:

1. image input and structured JSON generation;
2. multi-turn OpenAI-compatible tool calling;
3. model/processor round-trip;
4. full-parameter multimodal training with no frozen LLM, vision encoder, or
   aligner parameters;
5. distributed checkpoint save, resume, and inference reload;
6. acceptable memory behavior on up to four A100 40GB GPUs.

If a critical gate fails, record the failure and use the existing
Qwen3-VL-8B-Thinking checkpoint for diagnosis while resolving the concrete framework
issue. Do not fall back to a text-only model.

## 3. Resource policy

Before every server run, inspect live GPU ownership and memory.

```text
four free GPUs       -> use four
three free GPUs      -> use three with ZeRO-3 and CPU optimizer offload if needed
two free GPUs        -> use two for serving/protocol work; full-training smoke may
                        use ZeRO-3 offload
one free GPU         -> continue serving and data-contract work; do not disrupt jobs
zero free GPUs       -> prepare locally and wait for a safe allocation
```

Never kill, preempt, or reuse another user's process. Record selected physical GPU
IDs, CUDA visibility, framework versions, and peak allocated memory in each run.

## 4. Environment boundary

Keep three independent environments:

```text
ifv-agent
  existing Gemini runtime
  read-only dependency baseline

ifv-qwen-serve
  vLLM or SGLang serving stack
  OpenAI-compatible image and tool-call endpoint

ifv-qwen-train
  ms-swift
  DeepSpeed
  full-parameter SFT and later RL dependencies
```

Create and use:

```text
local:  D:\image-factual-verifier-training
server: /gs/home/wza/projects/image-factual-verifier-training
```

The training project may consume versioned files exported by the runtime. It must
not import runtime Python packages or install training dependencies into `ifv-agent`.

## 5. Mature framework stack

Use:

```text
training:  ms-swift
parallel:  DeepSpeed ZeRO-3 first
serving:   vLLM first for Qwen3.5; vLLM or LMDeploy for Qwen3-VL
```

The first training configuration is true full-parameter training:

```text
train_type=full
freeze_llm=false
freeze_vit=false
freeze_aligner=false
```

Do not silently substitute LoRA, QLoRA, adapters, or a quantized training model.
Activation checkpointing, Flash Attention, BF16, ZeRO partitioning, and optimizer
offload are memory techniques and are allowed.

## 6. Execution phases

### Phase 0: close three real Gemini runtime canaries

No Qwen environment or GPU service is started before this gate passes.

Current selected set:

```text
case_5cb541d78661ad17  Queen coronation vehicle substitution  expected fake
case_8d9e67f836117760  Andreea Esca sponsored-product claim   expected fake
case_e38eb5bcd2f433e4  NASA Pillars of Creation               expected real
```

Acceptance is not classification-only:

1. the core proposition must preserve the salient visible relation;
2. unrelated reference images cannot become verdict Evidence;
3. a different capture may provide discovery context but cannot be the sole terminal
   basis;
4. the evidence chain should recover the construction chain or a semantically
   equivalent source, not merely any source that yields the same label;
5. no post-determination actions, protocol rejections, or unbounded sibling sweeps;
6. strict trace audit passes;
7. low-value actions remain bounded and the stop follows the decisive semantic
   decision.

Known checkpoint on 2026-07-16:

```text
Monarch/Antarctica route-control regression
  passed: fake, 4 actions, strict audit

Queen
  visible event target and reference-evidence boundary repaired
  latest run: fake and strict audit passed
  not yet final: construction-chain recovery was still off-chain

Andreea
  old run found manipulation and the microphone stock record
  not accepted: initial target drifted into an unseen pre-edit image state and the
  run ended in a mandatory Evidence Decision validation error
  repair committed: initial cores must stay on the visible person-product relation

NASA
  latest run: real, 3 actions, strict audit, training_eligible=true
  candidate pass: same-capture visual bridge was valid
  final chain audit still needs an official NASA or semantically equivalent
  construction-chain source
```

Runtime repairs already committed:

- target planning preserves visible subject-object-event relations while removing
  authenticity wrappers;
- hidden original/unaltered/counterfactual media states cannot own the initial core;
- unrelated reference comparisons do not create Evidence;
- different-capture visual context cannot be the sole terminal basis;
- mixed visual and independent source Evidence may be adjudicated together;
- retrieval batches allow at most two total inspections, except that a useful visual
  bridge may unlock only its paired source page;
- task exhaustion is governed by the finite route inventory, not an arbitrary
  per-task action cap;
- the auditable run date is injected with `IFV_RUNTIME_DATE=2026-07-16`.

If one selected case repeatedly cannot recover a construction-aligned chain after the
runtime structure is sound, replace it with another representative v4 case rather
than weakening the acceptance standard.

### Phase A: baseline and machine audit

- Record the clean runtime commit and full Gemini canary command.
- Record `ifv-agent` Python, package, CUDA, and provider health without changing it.
- Inspect GPU processes, free memory, model directories, disk space, compiler/CUDA
  modules, and existing Conda environments.
- Confirm the v4 release paths and keep them read-only.

### Phase B: independent Qwen repository and environments

- Create the separate training repository with environment manifests.
- Pin exact framework/model revisions after the first passing smoke run.
- Add scripts for environment audit, model download, serving, health checks,
  full-training launch, checkpoint audit, and resume.
- Keep generated data, checkpoints, caches, and logs on `/gsdata`.

### Phase C: serving and protocol gate

Start the primary candidate through an OpenAI-compatible endpoint and test:

1. text completion;
2. local image input;
3. schema-constrained perception JSON;
4. exactly one tool call;
5. tool result continuation;
6. eight-action synthetic tool loop;
7. deterministic stop output;
8. concurrent health probe and clean shutdown.

The runtime provider profile must be explicit. There is no automatic Gemini fallback.

### Phase D: full-parameter training gate

Build a tiny processor-aligned multimodal dataset from public or synthetic smoke
fixtures. Do not use evaluator-private v4 gold as model-visible input.

Run in increasing cost:

```text
1 forward/backward step
3 optimizer steps
20 optimizer steps
save checkpoint
resume for 3 steps
reload for inference
```

Audit that gradients and optimizer state cover the language model, vision encoder,
and aligner. A run does not count as full training if one component is frozen or has
no gradient due to a configuration mistake.

Use short image sequences and batch size one for the infrastructure gate. This gate
proves correctness and recoverability, not model quality.

### Phase E: runtime adapter

Add a Qwen provider path without changing Gemini behavior:

```text
teacher-gemini
student-qwen-local
```

Both profiles consume the same public case, deterministic state, tool schemas, and
canonical stage output models. Provider-specific chat templates, reasoning tags, and
tool-call wire formats remain inside the Qwen adapter.

Run existing provider-neutral tests plus dedicated Qwen protocol tests. Then rerun an
unchanged Gemini canary.

### Phase F: preserve route-control behavior under Qwen

Verify that Qwen uses the repaired deterministic route controller unchanged:

- failed inspections recover through bounded sibling candidates;
- compact pixel/OCR anchors inform query choice;
- source class remains a ranking and confidence signal, not a general access gate;
- UGC and unknown sources may contribute evidence, while prompt injection, explicit
  benchmark exclusions, inaccessible pages, and serious temporal mismatch retain
  hard handling;
- a decisive semantic Evidence Decision stops immediately.

These remain runtime responsibilities and must not become Qwen prompt patches.

### Phase G: Qwen real v4 canaries

Choose three or four cases from the v4 20-case Development Preview, not legacy Apple
fixtures. The set should include:

```text
one supported scene/event case
one refuted ecological or geographic relation
one refuted provenance/context case
one additional supported or visually difficult case, if resources allow
```

The Monarch/Antarctica case remains the route-control regression. The same three
accepted Gemini cases are the initial Qwen canaries so model-quality regressions can
be separated from retrieval and runtime-control regressions.

For every case inspect:

- proposed core fact and image grounding;
- search queries and selected pages;
- failed-route recovery;
- Evidence spans and source provenance;
- semantic Evidence Decision;
- stop timing;
- final verdict;
- tool and model call counts;
- strict scheduler/trace audit.

Do not run all 20 cases during infrastructure bring-up.

## 7. Acceptance criteria

Infrastructure is accepted only when:

1. Qwen serves local images and valid structured outputs;
2. Qwen completes multi-turn tool calls through the runtime adapter;
3. a true full-parameter multimodal run saves, resumes, and reloads;
4. the training and serving environments are independently reproducible;
5. the unchanged Gemini environment and canary still pass;
6. three or four selected v4 cases complete as auditable real trajectories;
7. observed failures are separated into model-quality, retrieval/tool, and runtime
   control failures;
8. operations documentation contains exact launch, stop, health, log, and recovery
   commands.

Correct verdict alone is not sufficient. A case also needs a reasonable core target,
usable evidence chain, bounded search, and timely stopping.

## 8. Deferred work

After full-training and real-canary acceptance:

- export accepted Gemini teacher trajectories;
- build processor-aligned perception/ReAct/Reflection/Judgment datasets;
- run meaningful full-parameter SFT;
- add preference or GRPO training only after rewards and multi-turn environments are
  validated;
- expand from selected canaries to the complete 20-case development set.

RL is not allowed to block the initial Qwen deployment and full-parameter SFT gate.

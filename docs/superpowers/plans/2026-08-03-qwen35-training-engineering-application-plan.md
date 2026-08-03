# Qwen3.5 training engineering application plan

日期：2026-08-03
状态：计划；等待逐项执行与实测更新
范围：把 `modern_genai_bilibili` 中可迁移的训练、数据、诊断、serving 与未来 RL 工程经验应用到 IFV Qwen3.5-9B。

本计划延续：

- `docs/superpowers/plans/2026-07-23-qwen3.5-9b-sft-execution-plan.md`
- `training/docs/qwen35-engineering-notes-from-modern-genai-bilibili.md`

目标不是替换现有 SFT 路线，而是在不破坏已验证训练语义的前提下，系统评估能否提升：

- 四卡训练速度；
- 四卡显存余量；
- checkpoint/save/resume 稳定性；
- 数据与图像预处理吞吐；
- serving 和未来 Agent RL 的 rollout 效率。

## 1. 基准事实与约束

### 1.1 当前不可随意改变的事实

- 训练框架仍以 ms-swift 为 SFT 主路径，不自写训练循环。
- Qwen3.5-9B full-parameter SFT 继续学习经过审计的阶段输出，不学习 hidden reasoning。
- 冻结评测集仍不得进入 SFT、RL、调参或 checkpoint 选择。
- 任何训练优化不能改变 accepted rows、loss mask、stage ownership、private-gold 隔离或 runtime schema。
- 四卡 no-offload 不能因为单步更快就进入正式路径；它必须通过 train、validation、save、resume、merge/reload。

### 1.2 当前工程判断

- ZeRO-3 optimizer CPU offload 是当前已验证的安全路线。
- ZeRO-3 no-offload 在四张 A100-40GB 上仍是高风险候选。
- 5-GPU no-offload 单步成功只能证明可启动，不能证明正式训练可用。
- 旧的异常慢 offload 结果应首先按 GPU/CPU locality 和 NUMA 诊断，不应作为 tuned offload 的全局速度结论。
- sequence parallel 主要降低 activation 显存，不解决 full AdamW optimizer state 的根本占用。
- padding-free 可以优先测；full packing 对 Qwen3.5 mixed linear-attention 样本隔离有风险，必须先过 correctness gate。

## 2. 执行原则

1. 先测不改变训练语义的优化，再测改变后端/并行策略的候选。
2. 每个候选都必须保存同一套 profile artifact，不接受只看控制台日志。
3. 任何 profile 只要出现 OOM、NaN、NCCL error、save/resume failure、schema reload failure，即退出候选，不用继续长跑。
4. 不把 serving/vLLM 指标当作 SFT 显存已经解决的证据。
5. 不把 veRL 动态 batching 当作 ms-swift SFT 的现成功能。
6. 不在文档、profile、日志里写个人密码、Jupyter base URL、SSH 账号或代理细节。

## 3. Phase 0：统一基准与观测

目的：先建立同一套指标口径，避免继续用不同脚本、不同 GPU 集、不同数据 cache 比较。

### 3.1 产物

- `docs/reports/qwen35-engineering-baseline-YYYYMMDD.json`
- 每次训练的 `train.log`
- `environment.json`
- `nvidia-smi` 采样日志
- GPU topology / affinity 摘要
- dataset cache 命中摘要

### 3.2 指标

每个候选至少记录：

- physical GPU IDs；
- CUDA visible order；
- model/profile/deepspeed/fsdp config；
- max length；
- per-device train/eval batch；
- grad accumulation；
- dataset/cache mode；
- dataloader worker / persistent / prefetch；
- train seconds per step；
- samples/s；
- tokens/s，若可得；
- peak GPU memory by rank；
- CPU RSS peak，尤其 save/resume；
- first step time vs steady-state time；
- validation runtime and loss；
- checkpoint save time；
- resume/reload result。

### 3.3 基准 runs

Run A：当前安全基线

```text
backend       DeepSpeed ZeRO-3
offload       optimizer CPU offload, params on GPU
gpus          same four physical GPUs used for formal training
max_length    32768
batch         current safe profile
steps         10
must include  validation + checkpoint save
```

Run B：known-fail confirmation, bounded

```text
backend       DeepSpeed ZeRO-3 no-offload
gpus          same four physical GPUs
max_length    current minimum failed / probe length
steps         1
purpose       confirm failure mode and largest allocation
```

Run B is bounded and optional if the latest OOM evidence is already fresh. Do not spend
more time shrinking context unless a new backend materially changes memory behavior.

### 3.4 Acceptance

Phase 0 is complete only when the safe baseline can be reproduced from a fresh shell with
the same launcher, same profile, and artifacts saved outside the Git checkout.

## 4. Phase 1：safe quick wins on the current ZeRO-3 path

目的：先优化数据、cache、I/O、GPU locality，这些不改变训练目标，也不改变 optimizer/backend。

### 4.1 Dataset cache hardening

Apply:

- standardize `IFV_CACHED_DATASET` and `IFV_CACHED_VAL_DATASET`;
- write cache artifacts under the training data root, not under the Git checkout;
- record cache source dataset SHA/mtime and profile ID;
- fail fast if cached train/val paths are missing or mismatched.

Acceptance:

- cached and uncached runs produce the same row counts;
- processor/token audit remains unchanged;
- 3-step cached run has no schema or loss-mask regression.

### 4.2 Dataloader worker matrix

Test matrix on the safe ZeRO-3 optimizer-offload profile:

```text
workers:            0, 2, 4, 8
persistent_workers: false/true when workers > 0
prefetch_factor:    2, then 4 only if workers > 0 and stable
```

Measure:

- first-step time;
- steady-state step time;
- CPU utilization;
- GPU utilization;
- dataloader warnings/errors;
- memory drift across 10 steps.

Acceptance:

- choose the fastest stable setting that does not increase GPU peak enough to threaten validation/save;
- if variance is high, prefer the simpler lower-worker setting.

### 4.3 Cache root and temp root

Keep or formalize:

- `HF_HOME`;
- `HF_DATASETS_CACHE`;
- `TORCH_HOME`;
- `TORCH_EXTENSIONS_DIR`;
- `TRITON_CACHE_DIR`;
- `TMPDIR`;
- `IFV_TRAINING_CACHE_ROOT`.

Acceptance:

- no cache writes under the Git checkout;
- repeated run avoids rebuilding CUDA extensions unless versions change;
- cache location is documented but not tied to a personal account path.

### 4.4 GPU/CPU locality diagnostics

Add a lightweight diagnostic script rather than copying private-repo code verbatim:

- H2D bandwidth per GPU;
- D2H bandwidth per GPU;
- local HBM copy/add bandwidth;
- `nvidia-smi topo -m`;
- visible GPU mapping;
- CPU affinity snapshot.

Proposed file:

```text
training/scripts/server/diagnose_gpu_io.sh
training/scripts/server/measure_gpu_io.py
```

Acceptance:

- runs without training dependencies beyond torch;
- writes a small JSON/TSV report;
- redacts host/account-specific details from committed examples.

## 5. Phase 2：checkpoint/save/resume production gate

目的：把 checkpoint 峰值从“偶然遇到”变成每个候选的正式门禁。

### 5.1 Gate sequence

For every training backend candidate:

1. one train step;
2. one validation step;
3. checkpoint save;
4. process exit;
5. resume from checkpoint;
6. one additional train step;
7. merge/reload or serving-compatible reload smoke;
8. record CPU and GPU memory peaks.

### 5.2 Acceptance

A profile is production-eligible only if:

- no GPU OOM;
- no CPU OOM / SIGKILL / exit -9;
- no NCCL error;
- no NaN / inf loss;
- checkpoint contains model state and optimizer state when expected;
- resume advances from the prior global step;
- reload smoke returns valid model output and does not require private evaluator data.

## 6. Phase 3：FSDP2 no-offload branch

目的：测试是否能把四卡 no-offload 从 DeepSpeed ZeRO-3 OOM 变成 FSDP2 可行候选。

### 6.1 Required launcher changes

Current launchers assume `IFV_DEEPSPEED`. To support FSDP2 safely:

- add optional `IFV_FSDP`;
- enforce exactly one of `IFV_DEEPSPEED` or `IFV_FSDP`;
- allow `IFV_FSDP=fsdp2` or an explicit fsdp config path;
- when `IFV_FSDP` is set, pass `--fsdp "$IFV_FSDP"` and do not pass `--deepspeed`;
- add `IFV_GRADIENT_CHECKPOINTING` so FSDP2 profiles can disable regular gradient checkpointing;
- if FSDP2 activation checkpointing is used, keep it in the FSDP config/profile rather than duplicating ordinary gradient checkpointing;
- add tests for mutual exclusion and command construction.

Candidate files:

```text
training/configs/sft/qwen3.5-full-1step-4gpu-fsdp2-no-offload.env
training/configs/sft/qwen3.5-full-10step-4gpu-fsdp2-no-offload.env
training/configs/sft/qwen3.5-full-pilot30-4gpu-fsdp2-no-offload.env
```

### 6.2 Gate order

Gate A：CLI dry run / help

- confirm ms-swift accepts the FSDP2 arguments in the frozen environment;
- confirm it rejects DeepSpeed + FSDP2 together.

Gate B：one-step longest-real-row gate

- same four physical GPUs;
- no CPU optimizer offload;
- full parameter;
- max length 32768;
- save disabled for this first gate unless memory allows.

Gate C：10-step mixed-length gate

- validation enabled;
- checkpoint save enabled;
- memory/profile artifact required.

Gate D：resume and reload

- resume from saved checkpoint;
- run one extra step;
- reload or merge for inference smoke.

### 6.3 Decision

Promote FSDP2 only if it beats the safe ZeRO-3 optimizer-offload baseline on at least one
of these without losing reliability:

- lower wall-clock time for the same global batch;
- enough memory headroom to avoid known validation/save OOM;
- simpler operation with no CPU offload locality sensitivity.

Reject or defer if it:

- OOMs before checkpoint;
- saves but cannot resume;
- requires FULL_STATE_DICT save that creates CPU memory risk;
- is slower than tuned optimizer offload with no operational benefit.

## 7. Phase 4：padding-free and sequence parallel

目的：evaluate long-sequence activation optimizations after a backend is stable.

### 7.1 Padding-free

Profiles:

```text
safe baseline + IFV_PADDING_FREE=true
FSDP2 candidate + IFV_PADDING_FREE=true
```

Acceptance:

- training loss finite;
- validation pass;
- no change to row count or loss-mask audit;
- no sample-boundary correctness issue in a small logits/loss comparison.

### 7.2 Sequence parallel

Profiles:

```text
IFV_SEQUENCE_PARALLEL_SIZE=4
IFV_PADDING_FREE=true
IFV_ATTN_IMPL=flash_attn
```

Required launcher change:

- expose `IFV_SEQUENCE_PARALLEL_SIZE`;
- pass `--sequence_parallel_size "$IFV_SEQUENCE_PARALLEL_SIZE"` only when set;
- test that the argument appears in both single and curriculum launchers.

Acceptance:

- 10-step run passes with validation and save;
- activation memory reduction is visible on long rows;
- all-to-all overhead does not erase the benefit relative to the stable baseline.

### 7.3 Do not test full packing until this gate exists

Packing gate:

- same sample alone vs same sample in packed batch;
- compare loss/logits within an agreed tolerance;
- include at least one short sample following one long sample;
- include image-bearing rows;
- include tool-schema-heavy rows;
- verify no hidden recurrent/causal-conv contamination.

Until that passes, do not use packing for formal SFT.

## 8. Phase 5：serving and rollout optimizations

目的：apply vLLM notes to IFV runtime throughput and future RL, not to SFT memory.

### 8.1 Runtime serving benchmark

Benchmark:

- no prefix cache vs prefix cache;
- chunked prefill on/off;
- different `max_num_batched_tokens`;
- repeated IFV Planning/Decision prefixes;
- single-request deterministic smoke and batched-throughput smoke.

Acceptance:

- lower time-to-first-token or better throughput for repeated long prefixes;
- no schema regression;
- no change to provider protocol;
- deterministic test tolerance documented.

### 8.2 Cache-aware routing, deferred

Only consider if multiple local serving replicas are active. Same-prefix requests should
route to the same worker only when it does not harm reliability or isolation.

## 9. Phase 6：future Agent RL architecture

目的：use veRL AgentLoop ideas for architecture boundaries, not immediate migration.

### 9.1 Boundary to preserve

```text
IFV environment / tool protocol
  -> rollout manager
  -> model server
  -> trajectory artifact
  -> reward ledger
  -> GRPO trainer
```

### 9.2 Before any async rollout

Must define:

- policy version recorded per trajectory;
- rollout model hash;
- tool outcome schema;
- stale sample threshold;
- rejection or downweighting policy for stale trajectories;
- queue drain behavior on engineering errors;
- reward ledger compatibility with existing semantic reward.

Do not start fully async training until synchronous GRPO has a clean production run.

## 10. Implementation backlog

### 10.1 Code changes

1. Add FSDP2 launcher support:
   - `IFV_FSDP`;
   - `IFV_GRADIENT_CHECKPOINTING`;
   - mutual exclusion with `IFV_DEEPSPEED`;
   - conditional `--fsdp` / `--deepspeed`;
   - tests.
2. Add sequence-parallel launcher support:
   - `IFV_SEQUENCE_PARALLEL_SIZE`;
   - conditional `--sequence_parallel_size`;
   - tests.
3. Add diagnostic scripts:
   - H2D/D2H/HBM microbench;
   - topology and affinity summary;
   - JSON/TSV report.
4. Add profile-report helper:
   - parse train log;
   - extract step time, memory, validation, save/resume;
   - write `profile.json`.
5. Add cached dataset gate:
   - verify cached train/val presence;
   - record source dataset fingerprints.

### 10.2 Profile additions

Candidate profile files:

```text
training/configs/sft/qwen3.5-full-1step-4gpu-fsdp2-no-offload.env
training/configs/sft/qwen3.5-full-10step-4gpu-fsdp2-no-offload.env
training/configs/sft/qwen3.5-full-1step-4gpu-fsdp2-padding-free.env
training/configs/sft/qwen3.5-full-10step-4gpu-fsdp2-padding-free.env
training/configs/sft/qwen3.5-full-10step-4gpu-fsdp2-sp4-padding-free.env
training/configs/sft/qwen3.5-full-10step-4gpu-zero3-offload-cached.env
```

Do not delete the existing optimizer-offload profiles while testing these.

### 10.3 Documentation additions

Update after experiments:

- this plan;
- `training/docs/qwen35-engineering-notes-from-modern-genai-bilibili.md`;
- `docs/reports/qwen35-engineering-baseline-YYYYMMDD.json`;
- the July 23 SFT execution plan only if the formal SFT path changes.

## 11. Suggested execution order

### Day 1：safe path hardening

1. Re-run safe ZeRO-3 optimizer-offload baseline with full artifacts.
2. Build cached dataset profile.
3. Run dataloader worker matrix.
4. Add checkpoint save/resume/reload gate to the profile report.

Expected output:

- validated faster/same-speed safe profile;
- chosen dataloader setting;
- checkpoint gate artifact.

### Day 2：FSDP2 branch

1. Implement launcher support for `IFV_FSDP`.
2. Add FSDP2 one-step profile.
3. Run one-step gate.
4. If successful, run 10-step + validation + checkpoint.
5. If successful, run resume/reload smoke.

Expected output:

- promote, reject, or defer FSDP2 based on real evidence.

### Day 3：activation optimizations

1. Add `IFV_SEQUENCE_PARALLEL_SIZE`.
2. Test padding-free on the stable backend.
3. Test sequence parallel only after padding-free is stable.
4. Do not test full packing unless the correctness gate is implemented.

Expected output:

- activation-memory improvement decision;
- clear no-go if speed loss outweighs memory gain.

### Later：serving and RL

1. vLLM prefix-cache/chunked-prefill benchmark for IFV prompts.
2. AgentLoop-style rollout boundary design doc.
3. Synchronous GRPO first; async rollout only after version/staleness controls exist.

## 12. Final promotion criteria

A new profile becomes the recommended training path only if all are true:

- same or better training semantics;
- reproducible from a fresh shell;
- no private data leakage;
- no new dependency on a personal server setup;
- passes train/eval/save/resume/reload;
- has a profile artifact committed or archived;
- beats the current safe baseline on speed, memory headroom, or operational simplicity.

If none of the candidates clear this bar, keep the tuned ZeRO-3 optimizer-offload path and
apply only the safe data/cache/locality improvements.

## 13. Locked four-GPU experiment sequence (2026-08-03)

The following sequence is pre-registered before the remaining multi-step runs:

1. Treat FSDP2 + SP4 + padding-free + gradient accumulation 2 as a negative
   control. It reached the backward pass but OOMed while requesting 1.29 GiB with
   only about 1.10 GiB free.
2. Promote gradient accumulation 1 only as a candidate. Its one-step gate completed
   train, full validation, sharded save, and clean process exit with a 36.56 GiB
   peak per GPU.
3. Run the explicit accumulation-1 profile for 10 optimizer steps on physical GPUs
   4-7, using the frozen cached train and validation datasets.
4. Resume the resulting sharded checkpoint and advance at least one additional
   optimizer step. Check optimizer, scheduler, RNG, and global-step restoration.
5. Test checkpoint registration plus a serving-compatible model reload. Add an
   explicit FSDP2 export path if the sharded checkpoint cannot be consumed directly.
6. Re-run the four-GPU ZeRO-3 optimizer-offload baseline with the same cached
   datasets and compare examples/second as well as seconds/optimizer-step, because
   accumulation 1 changes the FSDP2 global batch from 8 to 4.
7. Only after both backends pass save/resume/reload, vary dataloader workers,
   persistence, and prefetch. Keep the fastest stable setting, preferring the simpler
   configuration when differences are within run variance.

No profile is promoted from a one-step result alone.

## 14. Steady-state recovery probes after AdamW OOM

The explicit accumulation-1 FSDP2 + SP4 + padding-free 10-step run failed on the
second training batch after the first optimizer step. The first step created full
AdamW moment state; the next backward all-gather then requested about 1.89 GiB while
less than 0.9 GiB remained free per GPU. This rules out the one-step result as a
production candidate.

Bounded two-step probes must run before any new 10-step attempt:

1. `bf16params`: set `--bf16 false --fp16 false` while keeping
   `--torch_dtype bfloat16`. This tests whether avoiding Accelerate FSDP2's FP32
   trainable-parameter upcast leaves enough memory for AdamW steady state.
2. `adafactor`: keep FSDP2/SP4/padding-free but replace AdamW with the factorized
   Adafactor optimizer. This is not AdamW-equivalent and can only become a speed /
   capacity candidate after loss, save, and resume gates pass.
3. `logits-to-keep-probe`: force `use_logits_to_keep=true` only as a compatibility
   probe. It is not eligible for promotion unless the step-1 loss matches the
   frozen reference and sequence-parallel label handling is explicitly verified.

The single-dataset launcher now forwards `IFV_GROUP_BY_LENGTH`; however, ms-swift's
sequence-parallel dataloader uses its own sampler. Therefore SP4 runs still need
real multi-step gates rather than relying on a "longest row first" assumption.

## 15. FSDP2 resume and export protocol

The FSDP2 candidate uses `SHARDED_STATE_DICT`, so the existing ordinary full-checkpoint
registrar cannot consume it directly. The locked production gate is:

1. resume `checkpoint-10` with an explicit `max_steps=11` profile;
2. require log evidence that model, optimizer, scheduler, and RNG state are loaded;
3. require global step 11, validation, a new `checkpoint-11`, and clean exit;
4. merge only `pytorch_model_fsdp_0` with Accelerate's
   `merge_fsdp_weights`; retain `optimizer_0`, scheduler, RNG, and trainer state as
   separate training-state artifacts;
5. copy tokenizer, processor, chat-template, and config assets from the frozen base
   model into the serving model directory;
6. audit language, vision, and aligner deltas against the base model;
7. load the merged model plus processor on CPU and require the expected Qwen3.5
   class and parameter count;
8. only after the CPU reload passes, test the exported directory through the frozen
   vLLM serving environment.

The source FSDP shards are never deleted by the exporter.

## 16. Unique-sample throughput correction and SP1 gate

The successful SP4 run has sequence-parallel size 4 and data-parallel size 1. Its
epoch progression proves that each optimizer step consumes one unique dataset row,
not four. Framework `train_samples_per_second` counts physical rank participation
and therefore overstates unique-example throughput for this configuration.

All comparisons must report:

- world size;
- sequence-parallel size;
- data-parallel size;
- unique samples per optimizer step;
- cumulative and steady unique samples per second.

The next bounded gates remove sequence parallel while retaining BF16 parameters and
AdamW:

1. SP1, accumulation 1: four unique samples per optimizer step;
2. SP1, accumulation 2: eight unique samples per optimizer step, matching the
   existing ZeRO-3 optimizer-offload global batch;
3. use `group_by_length=true`, which is effective on the non-SP dataloader and puts
   the longest rows into the first distributed mega-batch;
4. run two steps before any 10-step promotion, because optimizer state is created
   after the first step;
5. require the first-step loss to remain aligned with the frozen reference and
   require validation/save before promotion.

The SP1 gate failed before its first optimizer step on the longest grouped batch:
Qwen3.5 Gated DeltaNet requested another 272 MiB with only about 118 MiB free.
Because this happened in the first microbatch, reducing gradient accumulation does
not change the failure. SP1 is rejected for this dataset and 40 GiB cards.

The remaining bounded topology probes are:

- SP2 + accumulation 4: DP=2 and eight unique samples per optimizer step;
- SP2 + accumulation 1: fallback capacity/throughput measurement if accumulation 4
  fails because of no-sync gradient residency;
- SP4 + accumulation 8: global-batch-8 control. It is expected to amortize optimizer
  overhead but cannot increase the number of unique samples processed per
  microbatch.

# Qwen3.5 training engineering application plan

日期：2026-08-03
状态：核心训练、checkpoint、reload 与 serving gate 已完成；文档、测试和正式服务复查中
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

## 17. ZeRO-3 activation-recompute and dataloader protocol

The fair four-GPU ZeRO-3 optimizer-offload baseline completed ten optimizer steps,
validation, full model save, and DeepSpeed state save on the frozen cached dataset.
It processed eight unique rows per optimizer step, reached 18.63 GiB reported peak
GPU memory, and sustained 0.249 unique rows/second over the ten training steps.
This is faster in useful-row throughput than the raw SP4 FSDP2 step time suggests,
because SP4 processes only one unique row per optimizer step.

The next bounded speed hypothesis is that ordinary language-model activation
checkpointing is unnecessarily conservative on the offload path:

1. keep optimizer CPU offload, global batch eight, the same physical GPUs, cached
   rows, seed/order, FlashAttention, and visual gradient checkpointing;
2. set only `gradient_checkpointing=false`;
3. run two optimizer steps first, so the longest grouped batch and post-step
   optimizer state are both exercised;
4. reject immediately on OOM, NaN/Inf, NCCL failure, or loss divergence;
5. only after the two-step capacity gate, run ten steps with validation and save,
   then resume checkpoint 10 to step 11 and reload the saved full model.

After the backend/recompute choice is stable, run the dataloader matrix on that
choice. The primary matrix is workers `0/2/4/8`, with persistent workers and
prefetch two for nonzero workers. Only the best nonzero worker count receives
the additional persistent-off and prefetch-four ablations. Short matrix runs are
screening measurements, not promotion evidence; the selected setting must repeat
the ten-step validation/save/resume/reload gate. Compare unique rows/second and
sampled GPU utilization, not framework rank-counted throughput alone.

The no-language-checkpoint two-step capacity gate was rejected before the first
optimizer step. On the longest grouped batch, rank 0 reached about 39.32 GiB in
use and failed in a language MLP projection while requesting another 482 MiB with
only 134 MiB free. Ordinary language-model activation checkpointing therefore
remains required for this dataset, global batch, and 40 GiB cards.

The physical 4-7 GPU set is not showing a pathological offload link:

- measured H2D payload bandwidth was 23.87-24.59 GB/s;
- measured D2H payload bandwidth was 26.26-26.28 GB/s;
- local HBM read/write and stream-add measurements were about 1.35-1.37 TB/s;
- every GPU pair is connected through the reported NV12 fabric.

The ms-swift cached dataset is also narrower than an encoded-tensor cache. Its
Arrow rows contain normalized messages, image paths, and precomputed lengths.
ms-swift 4.4.2 wraps those rows in `LazyLLMDataset` and calls `template.encode`
during item retrieval, so multimodal image open/decode/processor work still occurs
during training. Dataloader tuning and a later encoded-image cache remain real
engineering tasks; the current cache only removes dataset reconstruction and
length-precomputation work.

## 18. ZeRO-3 CPUAdam OpenMP protocol

The generic server baseline remains `OMP_NUM_THREADS=1`. That value is appropriate
for shared-server control-plane commands and prevents accidental thread storms, but
it may underutilize the CPU during intentional optimizer offload. The installed
DeepSpeed 0.19.2 CPUAdam implementation contains OpenMP parallel loops, while the
four-rank training launcher previously forced every rank to one OpenMP thread.

The bounded training-specific protocol is:

1. keep the launcher default at one thread;
2. allow only an explicit `IFV_OMP_NUM_THREADS` training-profile override;
3. compare `4/8/16` threads per rank against the frozen one-thread baseline using
   the same two optimizer steps, cached rows, four physical GPUs, four dataloader
   workers, global batch eight, and activation checkpointing;
4. sample GPU utilization and host CPU pressure during every run;
5. reject a setting on CPU oversubscription, process instability, lower useful-row
   throughput, or no material gain beyond run variance;
6. only the smallest materially faster setting may advance to a ten-step
   validation/save gate and the later dataloader matrix.

This is a scoped CPU-offload exception, not a change to the general Jupyter/server
operations recommendation.

Because unrelated CPU-intensive jobs appeared after the original one-thread
baseline, the thread matrix also requires a contemporaneous `OMP_NUM_THREADS=1`
two-step control. The final thread decision compares the same-load-window
`1/4/8/16` runs; the older ten-step baseline remains useful for production
throughput but is not sufficient by itself to attribute a small OpenMP delta.

### 18.1 CPUAdam thread screening result

All four bounded profiles completed two optimizer steps, validation, checkpoint
save, and clean process exit with a reported 18.62 GiB peak:

| Threads per rank | Two-step train wall | Unique samples/s | Eval loss | Decision |
|---:|---:|---:|---:|---|
| 1 | 76.78 s | 0.208388 | 0.5911 | same-window control |
| 4 | 79.48 s | 0.201309 | 0.5911 | reject |
| 8 | 69.40 s | 0.230548 | 0.5914 | advance provisionally |
| 16 | 71.50 s | 0.223776 | 0.5908 | reject; smaller setting is faster |

The eight-thread profile was about 10.6% faster than the contemporaneous
one-thread control on observed useful-row throughput. It therefore advances to
the dataloader matrix as an explicit training-only override. The generic
operations baseline remains `OMP_NUM_THREADS=1`, and the later ten-step
production gate must still confirm that the short-run gain persists.

### 18.2 Dataloader screening result

The primary and follow-up matrices all used the same cached rows, four physical
GPUs, global batch eight, activation checkpointing, FlashAttention, and the
training-only eight-thread CPUAdam setting:

| Workers/rank | Persistent | Prefetch | Unique samples/s | Eval loss | Decision |
|---:|:---:|---:|---:|---:|---|
| 0 | n/a | n/a | 0.285205 | 0.5913 | advance |
| 2 | true | 2 | 0.258481 | 0.5913 | reject |
| 4 | true | 2 | 0.268908 | 0.5907 | best nonzero control |
| 8 | true | 2 | 0.242057 | 0.5912 | reject |
| 4 | false | 2 | 0.243309 | 0.5910 | reject |
| 4 | true | 4 | 0.228571 | 0.5906 | reject |

All candidates completed train, validation, save, and clean exit at a reported
18.62 GiB peak. Zero dataloader workers was about 6.1% faster than the best
nonzero candidate. This is consistent with the cached Arrow rows still requiring
lazy image decode and processor work while multiprocessing adds process startup,
IPC, memory, and teardown overhead. The production candidate therefore uses
`dataloader_num_workers=0` and omits persistence and prefetch arguments.

### 18.3 Production ten-step and resume result

The promoted profile combines:

- DeepSpeed ZeRO-3 optimizer CPU offload, with parameters kept on GPU;
- physical GPUs 4-7 and global batch eight;
- `OMP_NUM_THREADS=8` as a training-only override;
- `dataloader_num_workers=0`;
- FlashAttention and ordinary language/vision activation checkpointing;
- grouped cached rows and a 32768-token maximum length.

The formal ten-step gate completed training, validation, full model save, and
DeepSpeed optimizer-state save:

| Metric | Promoted profile | Earlier safe baseline |
|---|---:|---:|
| observed unique samples/s | 0.372613 | 0.248911 |
| steady unique samples/s | 0.444198 | 0.221239 |
| last-five step-wall mean | 18.01 s | not recorded as a matched steady window |
| reported peak GPU memory | 18.63 GiB | 18.63 GiB |
| eval loss | 0.5223 | 0.5260 |
| end-to-end train runtime | 270.9 s | 381.9 s |

Observed useful-row throughput improved by about 49.7%, while the steady-window
throughput approximately doubled. The checkpoint is about 123 GiB because each
of four ranks stores an approximately 28.2 GiB optimizer shard in addition to
the full BF16 model and small scheduler/RNG state.

Resume from checkpoint 10 restored the optimizer, scheduler, RNG, and global step,
advanced to global step 11, reran validation, saved checkpoint 11, and exited
cleanly. Reading the four optimizer shards is an explicit operational cost: the
resume initialization took several minutes before the additional training step
began. The step-11 profile is a recovery gate rather than a throughput benchmark.

## 19. Serving/checkpoint closeout (2026-08-03)

### 19.1 Checkpoint audit and CPU reload

The checkpoint-10 export gate passed with the training runtime initialized before
the Python reload process. The earlier reload attempt accidentally used the
system C++ runtime and failed on `CXXABI_1.3.15`; this is fixed in
`870d79b` by making `register_full_checkpoint.sh` call
`configure_training_runtime` before the audit and reload smoke.

The passing audit confirms:

- full-parameter training, with language, vision, and aligner all unfrozen;
- four BF16 model shards;
- four DeepSpeed optimizer shards, scheduler state, and per-rank RNG state;
- language, vision, and aligner weights all changed relative to the base model;
- `Qwen3_5ForConditionalGeneration`, `Qwen3VLProcessor`, and
  `9,409,813,744` parameters on CPU reload.

### 19.2 Checkpoint serving endpoint gate

The temporary checkpoint-10 vLLM service passed the bounded endpoint probe on
two independent tool roundtrips. The gate covered:

- loopback health and model discovery;
- 131,072-token context declaration;
- thinking off and thinking on with separated reasoning;
- image input;
- strict JSON Schema output;
- native tool call, `role=tool` continuation, and a second independent round.

The probe initially exposed a real protocol edge: with greedy decoding,
thinking plus JSON Schema could consume the whole output budget in `reasoning`,
leaving `content=null` and `finish_reason=length`. That response is not accepted
as structured evidence. The probe now follows the production boundary:

1. use the Qwen3.5 non-greedy sampling profile and a 1,024-token reasoning wall;
2. accept only validated visible JSON;
3. if a response has separated reasoning but no visible content, retry the same
   schema once with thinking disabled;
4. fail closed if neither attempt returns valid JSON.

The probe also uses an explicit no-proxy opener for loopback requests, so an
external proxy cannot turn a healthy local endpoint into a false 503.

Artifact:

```text
training/logs/serving/ifv-qwen3.5-9b-checkpoint10-smoke/checkpoint10-endpoint-gates.json
```

### 19.3 Final engineering decision

Promote the four-GPU ZeRO-3 optimizer-CPU-offload profile as the current IFV
Qwen3.5 training path:

```text
GPUs                    4-7
backend                 DeepSpeed ZeRO-3
optimizer offload       CPU
parameter placement     GPU
global batch            8
max length              32768
attention               FlashAttention
activation checkpoint   enabled
OMP override            IFV_OMP_NUM_THREADS=8
dataloader workers      0
```

FSDP2, sequence-parallel, no-language-checkpointing, and nonzero dataloader
worker profiles remain rejected or deferred by the measured gates in this
document. Do not reopen them without a new memory or hardware premise.

### 19.4 Operational handoff state

The temporary checkpoint service was stopped through
`manage_vllm_qwen35.sh stop`. Restarting the formal base-model service was
attempted through the same lifecycle script, but the launcher correctly refused
while another user's process occupied the selected runtime GPU set. No process
was killed or otherwise modified. A later operator may rerun:

```bash
CUDA_VISIBLE_DEVICES=4,5 bash training/scripts/serve/manage_vllm_qwen35.sh start
bash training/scripts/serve/manage_vllm_qwen35.sh status
curl --noproxy '*' -fsS http://127.0.0.1:8901/health
```

Only retry after the launcher reports the selected GPUs idle. This is an
environment-state blocker, not a model or checkpoint failure.

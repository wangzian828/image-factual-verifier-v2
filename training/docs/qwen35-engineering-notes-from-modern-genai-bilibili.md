# Qwen3.5 training engineering notes from `modern_genai_bilibili`

Source reviewed: private repository `wdkns/modern_genai_bilibili`, commit
`54f8f23748910cd687a00413ea8a0db069be5ab8`.

This note records engineering ideas worth carrying into IFV Qwen3.5 training and RL
planning. It is intentionally not a copy of that repository and does not include personal
server credentials, account names, or machine-specific access details.

Application plan:
`../../docs/superpowers/plans/2026-08-03-qwen35-training-engineering-application-plan.md`.

## High-signal takeaways

### 1. Four-GPU no-offload is not solved by the source repository

The source repository does not contain a ready-made recipe for:

- Qwen3.5-9B;
- full-parameter SFT;
- AdamW;
- 4 x A100-40GB;
- 32K context;
- no CPU offload.

Its value is in the knobs and failure analysis, not in a directly reusable training
profile. Continue treating the current IFV measured results as the source of truth:
ZeRO-3 optimizer CPU offload is the currently validated safe path; 4-GPU no-offload must
pass fresh gates before becoming a candidate.

### 2. FSDP2 is the next credible no-offload experiment

The source notes and ms-swift 4.4.2 both point to FSDP2 as the most plausible alternative
to DeepSpeed ZeRO-3. Relevant properties:

- FSDP2 shards parameters, gradients, and optimizer state.
- FSDP2 has a narrower full-parameter residency window than classic FSDP/ZeRO-style
  all-gather patterns, so it may reduce transient peak memory.
- In ms-swift 4.4.2, FSDP2 is mutually exclusive with DeepSpeed.
- With FSDP2, prefer native FSDP activation checkpointing over ordinary
  `--gradient_checkpointing true`.
- Prefer sharded state dicts for checkpoint saves; full state dicts may introduce large
  CPU/GPU memory peaks.

Proposed IFV gate:

```bash
# Conceptual only; wire through the existing IFV launcher/profile system.
swift sft \
  --fsdp fsdp2 \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --max_length 32768 \
  --attn_impl flash_attn
```

Do not combine this with `--deepspeed`. A successful one-step forward/backward is not
enough; the gate must include validation, checkpoint save, resume, and merge/reload.

### 3. Sequence parallel helps activation memory, not AdamW state

The sequence-parallel notes are valuable for long-sequence activation pressure. They do
not remove the main optimizer-state burden of full-parameter AdamW.

Use this as a second-stage experiment after FSDP2 or after a safe ZeRO-3 profile exists:

```bash
--sequence_parallel_size 4 \
--padding_free true \
--attn_impl flash_attn
```

Expected value:

- lower per-rank activation memory for long rows;
- more headroom for validation and rare long samples;
- possible speed penalty from all-to-all communication;
- no direct fix for optimizer state OOM.

Do not assume the 4-GPU no-offload OOM is fixed merely because sequence parallel starts.
The problem may remain dominated by full-parameter optimizer state and transient buffers.

### 4. Padding-free is safer than packing for Qwen3.5; packing needs a correctness gate

The source repository highlights a Qwen3.5-specific risk: mixed linear attention /
Gated DeltaNet / causal-convolution state can be wrong if multiple samples are flattened
without preserving sample boundaries all the way through the model path.

For IFV:

- `padding_free=true` is a reasonable candidate because ms-swift supports it and it avoids
  wasting compute on padding.
- full packing/bin-packing should not enter formal SFT until a correctness gate proves
  sample isolation.
- A packing gate should compare logits/loss on individual samples vs packed batches and
  include examples with different sequence lengths, images, and tool-schema text.

If the model path silently treats multiple samples as one continuous sequence, short rows
can inherit recurrent or convolution state from previous rows. That is unacceptable for
training data that should be independent.

### 5. CPU offload performance depends on PCIe and NUMA locality

The A100 node notes show the right mental model:

- GPU-to-GPU traffic on NVLink/NVSwitch can be hundreds of GB/s.
- CPU-to-GPU traffic over PCIe is much lower and depends heavily on CPU locality.
- Offload performance is therefore sensitive to which GPUs and CPU cores are used.

Operational consequences for IFV:

- avoid mixed GPU sets that cross bad CPU/NUMA locality when CPU offload is enabled;
- profile with the same physical GPU set used for the real run;
- bind or at least record CPU affinity when doing offload performance probes;
- record H2D/D2H bandwidth alongside training step time when a run is unexpectedly slow.

The old slow `0,1,2,6` style offload result should be treated as a locality pathology, not
as the global speed of the tuned four-GPU offload profile.

### 6. Dataset and image preprocessing can be a real throughput bottleneck

Useful source-repo ideas that match IFV:

- cache tokenized/processed datasets before long runs;
- put Hugging Face, torch extension, Triton, and dataset caches on fast storage;
- tune `dataset_num_proc`, `dataloader_num_workers`, `persistent_workers`, and
  `prefetch_factor` empirically;
- avoid repeated image decode and processor work when the same SFT rows are replayed;
- keep model weights and large cached datasets off the Git checkout.

The current launcher already exposes the right family of knobs:

- `IFV_CACHED_DATASET`;
- `IFV_CACHED_VAL_DATASET`;
- `IFV_DATASET_NUM_PROC`;
- `IFV_DATALOADER_NUM_WORKERS`;
- `IFV_DATALOADER_PERSISTENT_WORKERS`;
- `IFV_DATALOADER_PREFETCH_FACTOR`;
- `IFV_TRAINING_CACHE_ROOT`.

Next improvement is not another knob; it is a small benchmark matrix that records
step time, CPU utilization, GPU utilization, and cache hit behavior for worker counts
`0/2/4/8`.

### 7. Checkpoint save/resume is its own memory gate

The source notes call out checkpoint CPU memory spikes. For full AdamW, optimizer moments
are often the largest checkpoint component:

- parameters: BF16/FP16 weights;
- gradients: transient and sharded;
- AdamW state: two FP32 moment tensors;
- master weights may exist depending on framework and precision path.

For IFV Qwen3.5-9B, do not judge a training config by train-step success alone. A formal
profile must pass:

1. train step;
2. validation step;
3. checkpoint save;
4. resume from checkpoint;
5. merge or reload into the serving/evaluation path.

Any candidate that trains but fails save/resume is not production-safe.

### 8. vLLM rollout and serving knobs are useful later, not for SFT memory

The source notes on vLLM are useful for inference and future RL rollout:

- `max_model_len`;
- `max_num_batched_tokens`;
- `max_num_seqs`;
- chunked prefill;
- prefix caching;
- cache-aware routing for repeated long prefixes.

These knobs can improve IFV runtime throughput because our prompts have large repeated
system/context prefixes. They do not solve full-parameter SFT optimizer memory.

For deterministic evaluation, remember that continuous batching can introduce small
numeric differences even with fixed prompts. Treat this as an evaluation and reproducibility
concern rather than a training-memory knob.

### 9. AgentLoop architecture is useful for future RL

The veRL AgentLoop notes are conceptually useful for IFV Agent RL:

- keep the agent loop/tool environment separate from rollout scheduling;
- keep rollout server management separate from environment logic;
- log full trajectories with explicit policy version and tool outcomes;
- later consider async rollout only with freshness/staleness controls.

Do not migrate IFV runtime into veRL now. The useful piece is the boundary:

```text
AgentLoop / environment logic
  -> rollout server manager
  -> vLLM/SGLang worker(s)
  -> trajectory output
  -> reward and trainer
```

This is relevant when IFV moves from frozen SFT to online or semi-online Agent RL.

## Useful files in the source repository

These paths were the highest-signal review inputs:

- `agentic_rl/3D/fsdp_fsdp2.ipynb`
- `agentic_rl/3D/SP-序列并行.ipynb`
- `agentic_rl/3D/verl_sp.ipynb`
- `agentic_rl/3D/verl_packing.ipynb`
- `agentic_rl/verl/训练及调参经验/config-perf-tuning.ipynb`
- `agentic_rl/verl/训练及调参经验/sft.ipynb`
- `agentic_rl/verl/训练及调参经验/内存及显存.ipynb`
- `agentic_rl/verl/qwen3-5.ipynb`
- `agentic_rl/training_projs/当拿到一个A100的节点.ipynb`
- `agentic_rl/training_projs/verl-docker.ipynb`
- `agentic_rl/training_projs/data_parquet_weights.ipynb`
- `agentic_rl/training_projs/container-debug.ipynb`
- `agentic_rl/training_projs/scripts/pcie_bw_core.py`
- `agentic_rl/training_projs/scripts/hbm_bw_core.py`
- `agentic_rl/verl/agent/agent_loop_arch.ipynb`
- `agentic_rl/verl/agent/fully_async.ipynb`
- `agentic_rl/infra/inference/vllm-apc.ipynb`
- `agentic_rl/infra/inference/vllm_advanced.ipynb`
- `agentic_rl/infra/inference/Nondeterminism_in_LLM_Inference.ipynb`

## IFV experiment backlog

Run these in order; do not skip gates because an earlier micro-probe looks fast.

### A. Training memory and speed

1. FSDP2 no-offload one-step gate on the same four physical GPUs.
2. FSDP2 no-offload 10-step gate with validation and checkpoint save.
3. FSDP2 + `padding_free=true`.
4. FSDP2 + `sequence_parallel_size=4` + `padding_free=true`.
5. ZeRO-3 optimizer-offload baseline rerun with the same dataset cache and dataloader knobs.

### B. Data pipeline

1. Cached dataset export for train and validation.
2. Worker matrix: `0/2/4/8`.
3. Persistent workers on/off.
4. Prefetch factor `2/4`, only when workers are nonzero.
5. Repeated-image decode audit and processor-cache feasibility check.

### C. Correctness and production gates

1. Validation after training step.
2. Checkpoint save peak CPU/GPU memory.
3. Resume from checkpoint.
4. Merge/reload and run one inference smoke.
5. If packing is tested, compare packed vs un-packed logits/loss for sample isolation.

### D. Future RL and serving

1. Prefix-cache-aware serving benchmark for repeated IFV prompts.
2. Chunked-prefill benchmark for long IFV planning/decision prompts.
3. AgentLoop-style trajectory boundary design for future GRPO.
4. Async rollout only after policy-version logging and staleness limits are defined.

## Do not import blindly

Avoid importing these ideas directly without new IFV gates:

- full packing for Qwen3.5 mixed linear-attention samples;
- veRL dynamic batching as if it were an ms-swift SFT feature;
- FSDP2 with DeepSpeed enabled;
- ordinary gradient checkpointing plus FSDP2 activation checkpointing without checking
  which one the framework actually uses;
- CPU offload benchmarks from a different GPU/NUMA set;
- serving-side vLLM throughput knobs as evidence that training memory is solved.

# Four-H20 Qwen3.5 128K training experiments

This workflow targets four NVLink-connected H20 GPUs with approximately 96 GB per GPU.
It creates a separate Python 3.12 / CUDA 12.8 environment, without sourcing A100 profiles.
The training context ceiling is 131,072 tokens. Full-parameter tuning includes the LLM,
vision encoder and aligner, preserves native reasoning supervision, and uses no CPU offload.

Set `IFV_H20_ROOT` to the authorized server directory, `IFV_H20_RUN_ROOT` to a new
experiment directory below it, and `IFV_SFT_PACKAGE` to the unpacked formal SFT package.
The model is expected at `$IFV_H20_ROOT/models/Qwen3.5-9B-local`.

Run the committed scripts under `training/scripts/h20/`:

1. `install.sh`: isolated dependency installation and H20 SM90 CUDA extension compilation.
2. `environment.sh python h20_prepare.py`: verify image paths and real processor supervision;
   create short and stratified long training-only samples without truncating episodes.
3. `environment.sh python h20_capacity_data.py`: create a clearly synthetic, single-sequence
   approximately 130K-token capacity test, kept separate from production data.
4. `environment.sh torchrun --nproc_per_node=4 h20_kernel_check.py`: verify attention
   forward/backward and NCCL all-reduce on all four GPUs.
5. `environment.sh python h20_bench.py 1 <unique-run-name>`: bounded full-parameter benchmark.
   Use absolute script paths when invoking `environment.sh`, which changes directory.

Experiment controls: `H20_DATA=long-real.jsonl` or `capacity-synthetic.jsonl`, `H20_SP=4`,
`H20_STEPS=6`. Use `H20_SAVE=1` to save at the final step and `H20_RESUME=<checkpoint>`
with a larger total step count to check resume. FSDP2 activation checkpointing is used
instead of Transformers gradient checkpointing. Packing is disabled in the baseline.

`h20_metrics.py` summarizes sampled per-GPU memory and trainer logs. Measure steady-state
throughput separately from first-step compilation, initialization, and checkpoint I/O.
Synthetic capacity results do not establish full-dataset training throughput or model quality.
This is an experimental harness, not authorization to start an unbounded production run.
Before production, validate the complete package and bind processor verification to its hashes.

Source references:

- https://github.com/modelscope/ms-swift/blob/v4.4.2/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md
- https://github.com/modelscope/ms-swift/blob/v4.4.2/swift/config/fsdp2.json
- https://github.com/Dao-AILab/flash-attention/tree/v2.8.3
- https://huggingface.co/Qwen/Qwen3.5-9B

Changes are committed and pushed from the development workstation. The server only
fast-forwards clean checkouts and runs committed code; data, logs, environments, and
checkpoints remain outside the checkout.

## Measured configuration (2026-09-11)

Four H20 GPUs reported 97,871 MiB each, with NV18 links between every pair. The
container CPU quota was 108 cores. The separate environment passed `pip check`:
Python 3.12, PyTorch 2.10.0+cu128, Transformers 5.12.1, ms-swift 4.4.2,
FlashAttention 2.8.3, flash-linear-attention 0.5.1, causal-conv1d 1.6.2.post1,
Liger 0.8.0, and TileLang 0.1.14. TileLang is required for this tested Hopper GDN
backward path; see the [FLA correctness issue](https://github.com/fla-org/flash-linear-attention/issues/640).
All four GPUs passed FlashAttention forward/backward and NCCL checks.

The real benchmark used seven complete policy train episodes, 18,040–52,403
tokens (237,501 total). Processor probes checked native tool rendering, retained
tool responses, and supervised `<think>` tokens. This is stratified sampling,
not a complete processor audit of all 2,578 train rows. All image paths were resolved.
Packing retained all seven episodes as three packs (43,040–97,506 tokens).

| Run | Measured steps | Effective input tokens/s | Seconds/step | Highest sampled GPU MiB |
| --- | --- | ---: | ---: | ---: |
| Real SP4, batch 1, full shard | 4–6 | 1,064 | 38.19 | 73,000 |
| Real SP2, batch 1, no reshard | 4–6 | 3,783 | 15.29 | 95,882 |
| Real SP4, batch 1, packing budget 120K | 4–6 | 6,121 | 12.93 | 95,501 |
| Synthetic SP4, single 129,999-token sequence | 2–3 | 3,980 | 32.66 | 95,918 |

Token counts are divided by sequence-parallel degree to remove replicated accounting.
These short intervals exclude initial warmup but are not controlled quality comparisons:
packing changes the tokens per update, and compilation can still affect shape-dependent
timing. Synthetic data has 19,816 supervised tokens and one image per sequence; its loss
does not measure task quality. All listed runs completed full-parameter backward and
optimizer updates with finite losses and gradient norms.

The preferred tested real-data setting is SP4 + FSDP2 full shard + native activation
checkpointing + padding-free + `H20_PACKING=true H20_PACKING_LENGTH=120000`, batch 1.
Keep `max_length=131072`. This uses about 97.6% peak device memory; the near-128K
capacity run used about 98.0%, so there is little headroom for different image counts
or shapes. Do not infer that SP2 supports 128K: only SP4 passed the long capacity test.
Packing budget is not the supported context ceiling. Route episodes longer than the
packing budget through a separately validated unpacked SP4 path rather than dropping
or truncating them. Do not increase microbatch at near-full memory without a new test.

`summarize.py` writes `benchmark-summary.json` from completed runs and corrects SP
accounting and resumed-step timing. Each run retains command, source commit, trainer
logs, sampled GPU utilization/memory, and exit status. Full freeze is kept in the
server logs directory. Preserve benchmark checkpoints separately from production models.

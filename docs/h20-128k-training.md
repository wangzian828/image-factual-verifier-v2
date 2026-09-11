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

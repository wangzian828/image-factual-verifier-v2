# Teacher rollout and SFT entrypoint

Use `scripts/server/run_teacher_sft_pipeline.sh` for the complete teacher-side
pipeline. It runs the teacher rollout autopilot, strict trace audits, SFT package
audits, and the real target ms-swift processor check.

The entrypoint does not train by default. Add `--run-training` only after setting:

- `IFV_TRAINING_PYTHON` to the Python executable in the ms-swift environment;
- `IFV_MODEL_ID` to the target local checkpoint;
- `--training-model-profile` and `--training-sft-profile` to explicit profiles;
- `--training-experiment-id` to a new experiment name;
- `CUDA_VISIBLE_DEVICES` to the task's confirmed idle GPUs.

All runs require `OMP_NUM_THREADS=1`. Training inputs must have non-empty policy
`train.jsonl` and `validation.jsonl`; a package with no validation split is
rejected before training. The final status is written to
`<output-dir>/audits/pipeline-summary.json`, with processor and training status
recorded separately.

Example:

```bash
source scripts/server/ifv_env.sh
export IFV_TRAINING_PYTHON=/absolute/path/to/sft-env/bin/python
export IFV_MODEL_ID=/absolute/path/to/qwen-checkpoint
scripts/server/run_teacher_sft_pipeline.sh \
  --dataset-root <training-dataset-root> \
  --output-dir "$IFV_DATA_ROOT/generated/teacher-rollouts/<run-id>" \
  --limit 10 \
  --run-training \
  --training-model-profile "$IFV_REPO_ROOT/training/configs/models/qwen3.5-9b.env" \
  --training-sft-profile "$IFV_REPO_ROOT/training/configs/sft/qwen3.5-full-10step-4gpu-zero3-offload-accum1-bf16params-sdpa-checkpointed-16k-logits.env" \
  --training-experiment-id <experiment-id>
```

The `padding_free` optimization requires an installed flash-attention
implementation. The `qwen3.5-full-10step-4gpu-zero3-offload-accum1-bf16params-sdpa-checkpointed-8k.env`
profile is the portable fallback for environments without flash attention; it
uses full-parameter ZeRO-3 with CPU optimizer offload, disables `padding_free`,
uses a 16K sequence limit, enables gradient checkpointing, and retains only
the logits needed for the supervised labels on 40 GiB GPUs.
The launcher rejects an invalid `padding_free`/attention pair and undersized
multi-GPU datasets before starting distributed workers. The 16K profile is
chosen because the current smoke package contains rows above 8K tokens.

The separate Direct QA comparison package is evaluator-only and training-
prohibited. Build and use it through `docs/direct-qa-portable-package.md`.

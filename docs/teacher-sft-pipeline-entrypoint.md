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

The no-evaluation profile
`training/configs/sft/qwen3.5-full-10step-4gpu-zero3-offload-accum1-bf16params-sdpa-checkpointed-8k-truncated-smoke-noeval.env`
is smoke-only. It is intended to validate forward/backward execution and
checkpoint writing on a temporary expanded server dataset; it deliberately skips
validation because the installed Qwen3.5/ms-swift path can fail in the
multimodal validation collator after left truncation. A checkpoint from this
profile is a training-chain smoke artifact, not a validated production SFT
checkpoint.

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
  --training-sft-profile "$IFV_REPO_ROOT/training/configs/sft/qwen3.5-full-10step-4gpu-zero3-offload-accum1-bf16params-sdpa-checkpointed-16k-logits-512px.env" \
  --training-experiment-id <experiment-id>
```

The `padding_free` optimization requires an installed flash-attention
implementation. The
`qwen3.5-full-10step-4gpu-zero3-offload-accum1-bf16params-sdpa-checkpointed-8k.env`
profile is the portable fallback for environments without flash attention; it
uses full-parameter ZeRO-3 with CPU optimizer offload, disables `padding_free`,
uses an 8K sequence limit, caps image preprocessing at 65,536 pixels per image,
enables left truncation and gradient checkpointing, and retains only the logits
needed for the supervised labels on 40 GiB GPUs. Sequence parallelism is not
enabled because the current ms-swift release does not implement the required
training step for this model path.
The launcher rejects an invalid `padding_free`/attention pair and undersized
multi-GPU datasets before starting distributed workers. The 16K profile is
chosen because the current smoke package contains rows above 8K tokens.

The separate Direct QA comparison package is evaluator-only and training-
prohibited. Build and use it through `docs/direct-qa-portable-package.md`.

## Independent Qwen teacher and judge

The teacher rollout and frozen SFT judge are separate model configurations. A
server-deployed large Qwen teacher can use its own OpenAI-compatible endpoint:

```bash
export QWEN_TEACHER_BASE_URL=http://teacher-host:port/v1
export QWEN_TEACHER_MODEL=served-teacher-model
export QWEN_TEACHER_VISION_MODEL=served-teacher-vision-model
export IFV_SFT_ELIGIBILITY_PROVIDER=qwen_local
export IFV_SFT_ELIGIBILITY_BASE_URL=http://judge-host:port/v1
export IFV_SFT_ELIGIBILITY_MODEL=served-judge-model
export IFV_SFT_ELIGIBILITY_WIRE_API=chat_completions
export IFV_SFT_ELIGIBILITY_ENABLE_THINKING=true

scripts/server/start_teacher_rollout_autopilot.sh \
  --rollout-profile teacher-qwen-server \
  --rollout-model "$QWEN_TEACHER_MODEL" \
  --sft-judge-provider "$IFV_SFT_ELIGIBILITY_PROVIDER" \
  --sft-model "$IFV_SFT_ELIGIBILITY_MODEL" \
  --sft-judge-base-url "$IFV_SFT_ELIGIBILITY_BASE_URL" \
  --sft-judge-wire-api "$IFV_SFT_ELIGIBILITY_WIRE_API" \
  --sft-judge-enable-thinking
```

The judge receives the original image and the structured JSON schema. Qwen
responses that put the final JSON in `reasoning_content` or `reasoning` are
accepted only after strict JSON validation; provider failures remain audit
errors and cannot become passing eligibility results. The durable pipeline
state records teacher and judge provider, model, endpoint, wire protocol, and
thinking configuration separately.

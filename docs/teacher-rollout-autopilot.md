# Recoverable teacher rollout autopilot

`scripts/trajectory/run_teacher_rollout_autopilot.py` is the production entry
point for generating frozen-Gemini teacher trajectories from a unified training
manifest.

It accepts only a train manifest.  Do not pass a test manifest or a directory that
mixes training and test rows.

## Isolation and durability

The input manifest can contain labels, claims, evidence, provenance, and prompt
metadata.  The autopilot separates it into:

- `runtime-release/runtime_input/cases.jsonl`: exactly `case_id`, `image_path`,
  and `image_sha256`, plus linked/copied image assets;
- `private-gold/private_gold.jsonl`: evaluator-private targets used only after a
  rollout has reached a terminal state.

The Agent receives only the runtime release.  The private projection is not copied
into the model-visible SFT package.

The projection computes every selected image hash once.  The rollout invocation
then explicitly skips `run_cases`' otherwise redundant full preflight rehash and
records `preflight_image_hash_verification=skipped_explicitly` in each run manifest.
Ordinary `run_cases` calls retain the default full preflight verification.

For each rollout phase, the script writes one immutable `attempt-XX` directory.
Only terminal `success` + binary-verdict traces are removed from the next pending
list.  Provider/tool/transport failures are automatically queued into a fresh-seed
attempt.  A successful case is never re-run by the engineering retry loop.

At the configured attempt cap, remaining cases are retained in
`unresolved-engineering-case-list.txt`; they are not relabelled as quality failures.

After the initial audit, trajectories are separated into:

- `early_correct_judgment`: an earlier
  `image_only_discrepancy_judgment` already gave the correct label;
- `final_only_judgment`: the frozen judge accepted it, but the strict earlier
  judgment was not correct;
- `sft_rejected`: a terminal trace rejected by the frozen SFT judge;
- unresolved engineering errors.

The last two quality buckets eligible for a reroll are `final_only_judgment` and
`sft_rejected`.  The reroll itself has the same engineering retry behavior, then
receives another frozen SFT audit.  Final staging preserves accepted, rejected, and
engineering diagnostics.  Package construction is automatic; model training remains
opt-in and is never started by this command.

## gpu-13 command

Run only through the committed gpu-13 wrapper:

```bash
cd /gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python scripts/trajectory/run_teacher_rollout_autopilot.py \
  --dataset-root /gsdata/home/wza/image-factual-verifier-v2-data/datasets/route-aware-hrc-stage2-10563-final-organized-20260824-r2/unified-dataset \
  --output-dir /gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/<run-id> \
  --rollout-concurrency 10 \
  --sft-concurrency 10
```

Use `--prepare-only` to build and validate the isolated projections before starting
external model calls.  Re-running the same command and output directory resumes
from the durable attempts and audit artifacts.

For an isolated 10-case stability smoke run before a full launch, use a different
output directory and add `--limit 10 --rollout-concurrency 10 --sft-concurrency 10`.

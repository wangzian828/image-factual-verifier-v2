# H20 main pipeline handoff — 2026-09-12 03:28 China time

## Current phase: formal SFT launched after user storage confirmation

The user explicitly confirmed sufficient storage and requested immediate launch.
The prior storage hold below is superseded. Four keepers were stopped; after
their CUDA allocations released, all four GPUs were verified at1MiB. Main judge
completion and the frozen training-data SHA were checked again. Formal SFT was
dispatched with resource monitor PID203959 to
`/volume/ybo/wza/training-artifacts/h20-formal-sft-20260912`.
Read `launch.json`, `train.log`, `resource-samples.jsonl` and eventual
`resource-summary.json`; do not launch a duplicate. Command is the frozen
`training-plan.json`: one full-data epoch, full parameters, 64K packing/SP4,
128K ceiling, final full-state epoch checkpoint. Dispatch is not completion;
verify actual steps, finite losses/gradients, final checkpoint and export next.
The original utilization guard remains active, and keepers must not run while
training is using the GPUs. If training fails, inspect before restoring protection.

06:34 update: **pre-training judge finished**. `progress.json` is
`judge_complete`, 1,682/1,682, no remaining cases. Independent checks verified
unique/equal case sets, all judge response contracts, the frozen ledger SHA
and every selected trace SHA. Final judge SHA-256:
`5c70e962b44ec63e77af2a70692b086c9250510c9c39b6656173d8ca3cbd6a84`.
Report is under the judged root's `report/`. Exploratory old1682 results:
accuracy60.9394%, balanced accuracy62.7968%; 38 correct with strong evidence,
987 correct with insufficient evidence,657 wrong. Zero engineering/judge errors
does not mean zero tool failures or good evidence quality. No test-result tuning.

**Training launch is held on an unresolved storage constraint**, not judge.
Historical real full-parameter H20 resume cleanup inventory records
112,933,396,586 bytes (105.18GiB) of model/optimizer shards. Current owned-tree
usage was92GiB; a separate inference export and temporary/cache headroom are
additional. Conservatively allow at least150GiB additional usable storage.
The mount is GPFS `f0b48339[/.userfset/f0af7518]`; `quota` and `mmlsquota` are
not exposed in this container. Shared39TiB free does not verify that allowance.
Need the actual personal allowance, or a scoped way to verify it, before
committing a >105GiB full-state write. Do not silently switch to model-only
checkpoints or delete preserved data to get around this constraint. Four
keepers and guard remain running while this is unresolved.

04:00 follow-up: judge controller remained active (306 returned judgments,
zero engineering errors at the snapshot). Do not interpret the controller's
batch-level `completed: 1` as stalled: in-flight batch progress is in
`judge-1/audit-results.jsonl`. Read-only production data preflight was rerun at
03:56 and passed with zero errors:
`/volume/ybo/wza/training-artifacts/h20-formal-sft-20260912/data-preflight.json`.
It revalidated the exact train/validation/manifest/processor hashes and SP4,
131072, image1024/max_pixels262144 template binding. No training was started.

Original run:
`/volume/ybo/wza/runs/eval/qwen35-base-agent-full1682-c40-pretrain-20260911`.
Finished 1,682 cases: 1,677 successful, five engineering errors (one malformed
final report, four unusable tool-call responses). No selection used gold.

Error-only retry:
`/volume/ybo/wza/runs/eval/qwen35-base-agent-full1682-c40-pretrain-20260911-retry1`.
All five succeeded, concurrency five, same policy/model/timeout/seed defaults;
successful deterministic tools could be reused from the original run. Original
results and traces were not overwritten. Retry PID193461 has exited.

Frozen replacement ledger and independent judge controller:
`/volume/ybo/wza/runs/eval/qwen35-base-agent-full1682-c40-pretrain-20260911-judged`.
Controller PID193598 at launch, main log:
`/volume/ybo/wza/logs/base-full1682-judge-controller.log`.
Read `progress.json`, `freeze.json`, `replacements.json`, `frozen-results.jsonl`,
`judge-*/summary.json` and `judge-*/audit-results.jsonl`. Do not start a second
controller or modify a running batch. Code: `scripts/finish_agent_eval_judge.py`.

The one-case judge canary passed its response-contract check. Batch1 started
the remaining 1,681 with concurrency16, Gemini `gemini-3.1-pro-preview`, high
thinking, 8,192 output tokens, timeout240s and two transport retries. Controller
retains the canary vote and all completed positive/negative votes, then performs
at most two error-only passes at concurrency8 and4 with backoff. A parsed but
incomplete judge JSON is not accepted as a completed judgment. `not_auditable`
is retained and reported, not endlessly retried or removed from the denominator.

The controller freezes the 1,682-result order and hashes, hashes selected traces,
and refuses replacements for already successful cases. Seven focused tests pass.
It stops before training. If killed mid-judge, it refuses to overwrite partial
paid results: inspect process activity, preserve valid cached votes, and implement
an explicit missing/error-only recovery selection locally before resuming.
Do not resample negative judgments or silently restart the full batch.

Private gold is evaluator-only:
`/volume/ybo/wza/data/factcheck-test-1682-full-source/evaluator_private/private-gold-v1/test-private-gold.jsonl`.
Its manifest is the adjacent dataset root's `test-manifest.jsonl`.
These are **old 1,682 exploratory comparisons**, not the unrecovered frozen
1,527 formal test set. No test data enters training or hyperparameter selection.

## GPU protection and next transition

04:32 preparation: `training/scripts/h20/prepare_formal_sft.py` prepared
`/volume/ybo/wza/training-artifacts/h20-formal-sft-20260912/training-plan.json`.
Five focused tests passed; the server rechecked dataset/processor hashes. The
plan preserves the measured 64K/SP4 command except full training input, one
epoch (`max_steps=-1`) and full-state epoch saving (one retained checkpoint,
not model-only; no best-model selection). It does not execute training. Use
the H20 environment wrapper and resource monitor when launching after judge.
Epoch-only saving limits storage but offers no mid-epoch checkpoint. Actual
user quota is still unknown (`quota` unavailable); measured current owned-tree
usage was 92GiB. Do not use shared 39TiB free space as user allowance or remove
source data. Final full-state checkpoint and separate inference export require
a conservative storage check before launch.

After confirming no Agent/PSD rollout or training remained, the owned base-model
replicas/gateway were stopped using their checked-PID manager. Four 12GiB,
25%-duty idle GPU keepers were started; the original utilization guard remains
active. These keepers must be stopped before training. Do not mistake deliberately
stopped port8901 for an inference-service failure during external judging.

Once final judge coverage is checked, proceed to authorized one-epoch **full-
parameter SFT**, not PSD LoRA. Complete the production launcher/preflight first:
2,578 frozen complete training trajectories, 131,072 context ceiling, packing
65,536, SP4/FSDP2, existing H20 environment. Preserve a complete optimizer,
scheduler and RNG checkpoint for later additional data. The benchmark harness
disables saving by default and is not a production launcher as-is.
Then reload the trained model and run the same 1,682 Agent cases and judge.

Training artifacts:
`/volume/ybo/wza/data/ifv-h20-policy-full-20260911` includes the processor report,
strict audit, launch data gate and release hashes. Use these frozen inputs;
do not repeat data collection. Shared filesystem free space is not user quota.

PSD's CPU optimizer and new three-case feedback tests are complete as recorded
in `psd-completion-checklist.md`; do not relaunch them. GPU PSD capacity testing
is still deferred until after the main sequence, and production PSD is not started.

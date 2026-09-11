# H20 main pipeline handoff — 2026-09-12 03:28 China time

## Current phase: pre-training independent judge

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

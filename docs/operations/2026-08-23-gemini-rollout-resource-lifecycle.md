# Gemini rollout resource-lifecycle incident — 2026-08-23

## Symptom

The full 950-case canonical training rollout started normally at concurrency 10,
then degraded until it appeared stalled:

- after 31 minutes, only 46 terminal traces had been written;
- the worker had 376 threads;
- the worker owned 256 TCP sockets in `CLOSE-WAIT`;
- all 46 written traces were valid terminal `real` or `fake` successes.

`CLOSE-WAIT` means the peer has already closed the connection while the local
process has not released its socket. The growth was owned by the rollout Python
worker, not an unrelated server process.

## Root cause

`VerificationWorkflow.run_batch()` intentionally creates an isolated child
`VerificationWorkflow` for each case. This prevents mutable investigation state
and runtime-event streams from crossing cases. The child was returned or raised
from without a terminal cleanup step.

Each child can own persistent resources:

- a Gemini vision `PersistentAsyncRuntime` and its HTTP client;
- a Jina reader executor and Gemini extraction client;
- the main Agent API backend and its transport pool.

Several such resources register `atexit` cleanup callbacks. That preserves the
object until process exit, so completed child workflows did not become
collectable and their threads/connections accumulated throughout the batch.

## Fix

Commit `24eea0f` adds an explicit terminal lifecycle:

1. `VerificationWorkflow.aclose()` delegates to the child orchestrator.
2. `run_batch()` calls it in a `finally` block for every isolated child,
   including errors and timeouts.
3. `Orchestrator.aclose()` walks the known shared tool-resource graph, dedupes
   objects by identity, and closes each sync or async resource once.

The change does not change prompts, policies, sampling, tool selection, verdict
logic, or retry eligibility. It only closes resources after a case has already
reached a terminal outcome.

## Validation

- local targeted test: `python -m pytest -q test_workflow_lifecycle.py`
  (`2 passed`);
- gpu-13 targeted test with the `ifv-agent` environment (`2 passed`);
- live new-code rollout, concurrency 10:
  - after three terminal successful traces: 73 threads and 7 `CLOSE-WAIT`;
  - the old worker at 46 successful traces: 376 threads and 256
    `CLOSE-WAIT`.

Established (`ESTAB`) sockets remain expected while the ten active rollouts hold
external requests. The signal to monitor is unbounded `CLOSE-WAIT` and thread
growth as completed case count increases.

## Recovery and continuation

The old worker was stopped after preserving its 46 terminal success traces. The
continuation list contains only the 904 case IDs without a terminal successful
trace:

```text
/gsdata/home/wza/image-factual-verifier-v2-data/generated/sft/private-gold/
gemini-training-full-1051-20260823-v2/
remaining904-after-transport-cleanup-case-list.txt
```

The active automatic chain runs the 904 remaining canonical cases with the
fixed runtime, merges them with the preserved 46 and the original first 100,
then continues with the special case, full SFT audit, rejected/final-only reroll,
reroll audit, and final accepted teacher release.


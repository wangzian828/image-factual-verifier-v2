# Five-track closeout research log

| # | Date | Type | Summary |
|---:|---|---|---|
| 1 | 2026-08-03 | bootstrap | Locked five tracks: OMP1 worker selection, content-addressed image preprocessing cache, scheduler warning, checkpoint I/O, and frozen reviewed-52 Agent replay. Existing user research edits remain out of scope. |
| 2 | 2026-08-03 | bootstrap | GPU audit found only physical GPUs 2 and 7 idle; 4/5 serve the formal vLLM endpoint and 0/1/3/6 are occupied by other users. Four-GPU experiments are deferred until a safe window; offline implementation continues. |

Protocol commits must precede result commits. Free-running search rollout is not a
substitute for the frozen Agent replay protocols.

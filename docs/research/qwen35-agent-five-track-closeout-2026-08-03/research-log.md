# Five-track closeout research log

| # | Date | Type | Summary |
|---:|---|---|---|
| 1 | 2026-08-03 | bootstrap | Locked five tracks: OMP1 worker selection, content-addressed image preprocessing cache, scheduler warning, checkpoint I/O, and frozen reviewed-52 Agent replay. Existing user research edits remain out of scope. |
| 2 | 2026-08-03 | bootstrap | GPU audit found only physical GPUs 2 and 7 idle; 4/5 serve the formal vLLM endpoint and 0/1/3/6 are occupied by other users. Four-GPU experiments are deferred until a safe window; offline implementation continues. |
| 3 | 2026-08-03 | results | H2 cache gate passed: cold build wrote 343 unique encoded rows; warm fresh process hit 526/526 with zero misses/writes and exact multimodal payload comparison. |
| 4 | 2026-08-03 | results | H3 scheduler audit passed at one and ten optimizer steps. DeepSpeed global steps and scheduler steps match; warning classified as wrapper bookkeeping false positive and remains gated rather than hidden. |
| 5 | 2026-08-03 | results | H4 checkpoint profiling passed. Four-GPU historical full-state checkpoint writes 122.71 GiB in 31.18 s; model/assets and optimizer phases are separately visible; resume state is retained. |
| 6 | 2026-08-03 | results | H5 source-visible-property extractor now emits concise pixel-checkable hints for animal attributes, clothing, gloves, sky colors, relations, and counts. |
| 7 | 2026-08-03 | results | Reviewed-52 frozen Qwen batch completed 9/9 qualified cases with 9/9 focused visual runs, 9/9 Decision-2 visual consumption compliance, zero engineering failures, zero missing consumption, and zero deterministic fallback use. |
| 8 | 2026-08-03 | protocol | OMP1 closeout runner and a safe GPU waiter were installed. The waiter will stop only the managed 4/5 vLLM service after the reviewed replay is complete and physical 6/7 are idle, then restore the service after worker comparison, 10-step save, and step-11 resume. |
| 9 | 2026-08-03 | outer loop | GPU 6 remains occupied by another user, so the research continues on independent work rather than treating the hardware window as a global blocker. |
| 10 | 2026-08-03 | protocol | Locked H6: controlled focused-visual timeout/provider/schema/empty/budget/error and Decision-exhaustion matrix with fail-closed source-only semantics. |
| 11 | 2026-08-03 | protocol | Locked H7: deterministic offline classification of all 976 reviewed-52 candidate Evidence rows plus optional frozen replay linkage. |
| 12 | 2026-08-03 | protocol | Locked H8: prepare OMP1 workers0/workers4 20-step profiles and a fail-closed H1-winner-driven formal pilot path; do not run before H1 resolves. |

Protocol commits must precede result commits. Free-running search rollout is not a
substitute for the frozen Agent replay protocols.

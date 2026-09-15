# SFT-4872, three epochs: frozen post-training evaluation

## Protocol (locked before policy smoke)

- Evaluate only the completed epoch-3 / step-3084 model from
  `/volume/ybo/wza/checkpoints/h20-sft-merged4929-agent-v2-3epoch-fullstate-20260914/v0-20260914-200132/checkpoint-3084`.
  This is an independent Base-initialized three-epoch run, not continuation of
  the earlier SFT-2578 or one-epoch SFT-4872 models.
- Preserve all three full-state checkpoints and all old weights/results.
  Export a separately audited BF16 inference copy; require finite tensors,
  full-parameter update audit, step/epoch identity, and preserved training state.
- Reuse the existing frozen server runtime at commit
  `1d61abb41f147e871b5361ee3e917aaa5c06ccc5`; do not modify Agent code/prompts/tools.
- Serve four single-H20 replicas with vLLM 0.18.1, TP1, BF16, 131072 context,
  32-image limit, memory utilization 0.94, eight sequences per replica, and the
  existing least-inflight gateway. Alias: `ifv-qwen3.5-9b-sft-3084`.
- Preserve the established request envelope: 32768 output tokens and 8192
  thinking budget for both ReAct and judgment stages; real external credentials
  are loaded from the existing root-only runtime environment, never logged.
- Reuse the four fixed smoke IDs in the prior SFT-1028 canary list. Seed 1903
  is also the first formal-attempt seed. No ground-truth label or judge outcome
  participates in the smoke gate or retry selection.
- Require successful complete trajectories, strict canonical audit, real external
  tool execution and checks of thought termination, multiple-image handling,
  native tool decoding/actions and final evidence-reference contracts before
  accepting the smoke. A failure stops automatic promotion for diagnosis.
- Once accepted, preserve smoke successes and run the remaining cases in a new
  directory at concurrency 40; engineering-only retries use concurrency
  32/24/16 and seeds 2903/3903/4903. Never rerun an already successful case or
  merge results produced by another checkpoint. Preserve all failed attempts.
- Frozen public benchmark:
  `/volume/ybo/wza/evaluation/factcheck-formal1527-available1526-20260912/runtime-release/runtime_input/cases.jsonl`,
  SHA-256 `c6568c302147893f7a648ea4e5cccd29e4c11dae831e1344399330ef75d7cf32`.
  There are 1526 runnable cases. Formal metrics retain denominator 1527
  (real 377, fake 1150); the unavailable real case counts as a miss.
- Compute BAcc and both class recalls without judge. After inference, submit
  the existing v3 full-material Gemini 3.7 Flash low/32768 Batch judge for SESR.
  No additional QA baseline or PSD training is authorized by this transition.

## Locations and acceptance record

- BF16 export:
  `/volume/ybo/wza/exports/h20-sft-merged4872-3epoch-step3084-20260915`.
- Serving:
  `/volume/ybo/wza/inference/sft3084-3epoch-20260915`.
- Evaluation:
  `/volume/ybo/wza/runs/eval/qwen35-sft3084-3epoch-agent-formal1527-20260915`.

Results and acceptance evidence will be recorded after the protocol commit;
launch alone is not an accepted smoke or a completed evaluation.

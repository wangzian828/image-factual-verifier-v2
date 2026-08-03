# Five-track closeout findings

## Research question

Can the OMP1 production path, image preprocessing, scheduler/checkpoint behavior,
and reviewed-52 Agent mechanism be made production-safe without changing training
or verdict semantics?

## Current understanding

- OMP1 is mandatory server policy. Historical OMP8 speed results remain audit
  evidence only. A fail-closed runner now compares workers0 and workers4 under
  the same OMP1/cache/ZeRO-3 conditions, then performs the promoted 10-step and
  step-11 resume gate.
- The ms-swift Arrow cache still stores normalized rows and references, but the
  new content-addressed processor cache removes repeated template/image encoding.
  The warm fresh-process gate reached 526/526 hits with zero misses and exact
  multimodal payload equivalence.
- The scheduler warning is a DeepSpeed wrapper bookkeeping false positive:
  DeepSpeed performs the real optimizer step at the accumulation boundary, the
  wrapper's visible ``step`` is a no-op, and scheduler/global-step sequences
  remain aligned.
- Full resumable ZeRO-3 state is expensive to write, but optimizer/scheduler/RNG
  state must remain. Four-way checkpoint phase profiling and post-save integrity
  gates are the safe optimization; shared-memory Flash Checkpoint is not safe on
  this host.
- Frozen reviewed-52 replays now make source hints concise and pixel-checkable.
  Focused visual exhaustion is a hard engineering failure, never source support.
  The expanded Qwen batch proves the second Decision consumes resolved pixel
  Evidence in every one of 9/9 qualified cases.

## Key results

- Encode cache: cold build 343 misses/writes plus same-run hits; warm fresh
  process 526/526 hits, 0 misses, 0 writes, exact comparison gate passed.
- Scheduler audit: one-step and ten-step probes both classified
  ``wrapper_false_positive``; DeepSpeed global steps and scheduler steps match.
- Checkpoint I/O: historical four-GPU full checkpoint writes 122.71 GiB in
  31.18 s (10.23 s model/assets, 20.51 s optimizer); recovery state remains
  complete.
- Reviewed-52 candidate scan: 52 cases, 975 snapshots, 976 candidate Evidence
  records, 9 strictly qualified cases. New seed-11 Qwen replay artifact:
  ``/gsdata/home/wza/image-factual-verifier-v4/runs/replays/five-track-0e45675-qwen-seed11-20260803``.
  It completed 9/9 with 9/9 visual reinspection runs, 9/9 Decision-2 visual
  consumption compliance, 0 engineering errors, 0 missing visual consumption,
  and 0 deterministic fallback use. Two cases reached terminal ``fake`` under
  this rollout; nonterminal cases retained explicit insufficient/supporting/
  conflicted gaps rather than source-only verdicts.
- Gemini online batch was not claimed because neither the remote kernel nor the
  project environment contains ``GEMINI_API_KEY``/``GOOGLE_API_KEY``.

## Patterns and insights

- Mechanism correctness and verdict terminality must be reported separately.
  A second rollout with the same seed can vary in terminality while preserving
  the stronger invariant: focused visual Evidence is recorded, then consumed
  by the next Decision.
- Source hints should identify one observable property, not replay an entire
  source paragraph. Clothing, sky-color, glove, animal-tail, relation, and
  count patterns now have targeted extraction paths.
- A source/pixel disagreement can be supporting or insufficient when it is
  unrelated to the current high-salience claim. The reducer must not turn every
  observed mismatch into ``fake``.

## Lessons and constraints

- Do not use another user's GPU process or stop it.
- Do not mutate historical OMP8 profiles or artifacts.
- Do not weaken optimizer/scheduler/RNG recovery to claim checkpoint speed.
- Do not use free-running search to validate the source-to-pixel mechanism.
- Do not revive or stage the user's deleted research workspace.

## Open questions

- Does workers4 beat workers0 after OMP is restricted to one, and does the
  selected profile complete the full 10-step/save/resume gate?
- Can a Gemini credential be supplied for the paired online replay?

## Optimization trajectory

1. ``encode-cache-c73d0c1`` — warm hit rate 100%, exact payload gate passed.
2. ``scheduler-audit-411a8f1`` — no skipped optimizer/scheduler step.
3. ``checkpoint-io-13bb645`` — four-way write window 31.18 s with full state.
4. ``reviewed52-qwen-0e45675-seed11`` — visual-consumption compliance 9/9.

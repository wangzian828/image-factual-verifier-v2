# Raw-history Runtime Cleanup Status

Updated: 2026-09-06

## Active Boundary

The active branch is `main`.
The production path is:

```text
src/orchestrator/pipeline.py
  -> src/orchestrator/react_runtime.py
  -> src/orchestrator/stage_runner.py
  -> provider-side cumulative interaction history
  -> raw tool observations
  -> unified Judgment
```

The runtime state contains only operational fields such as schema version, case
and image identity, objective, action count, stop reason, and completion
explanation. Tool observations are read from the original `state.all_steps`
history. The active path does not restore reducer state, graph state, dual-track
semantic state, or an `unresolved` verdict.

## Completed Changes

- Raw-history runtime, exporter, audit, and SFT eligibility now use the current
  interaction-history schema.
- Continuous original tool results are retained, including consecutive targeted
  searches with no relevant match.
- Empty search results, tool errors, and access failures remain observations;
  none is silently rewritten as a factual conclusion.
- The old graph/reducer scorer and handoff state are removed from the active
  path. Historical modules remain only where old replay or fixture boundaries
  explicitly require them.
- SFT eligibility uses `target_scope` categories and rejects related-but-
  incomplete evidence rather than relabeling it as unrelated.
- Teacher rollout, strict trace audit, frozen eligibility, provider-neutral
  export, policy/perception package audit, and processor verification are wired
  through the server entrypoint.

## Teacher Smoke

The completed server teacher smoke is stored at:

```text
/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/raw-history-smoke10-20260906-4e56145
```

- Ten training cases were run through the initial rollout and three quality
  rerolls.
- Strict trace audit passed 10/10 with zero warnings.
- Frozen release selection accepted 4/10; 6/10 remain hard cases and are not
  silently promoted to training data.
- The formal SFT package remains on the server at:

```text
/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/raw-history-smoke10-20260906-c51108d/sft-training-package
```

Policy/perception audits, processor verification, and encoded dataset checks
passed for that package. Formal training data, traces, and checkpoints remain on
the server.

## SFT Smoke

The previous 8K smoke completed all ten forward/backward steps but failed after
training in the installed Qwen3.5/ms-swift multimodal validation collator. The
failure was a left-truncation shape mismatch in the validation path, not a
training-data or backward-pass failure.

Commit `a563daa` adds an explicit smoke-only `noeval` profile that preserves the
same bounded ZeRO-3, SDPA, gradient-checkpointed training path and checkpoint
save at step 10 while skipping validation. Its output is a training-chain smoke
artifact only and must not be described as a validated production SFT model.

Current experiment:

```text
raw-history-smoke10-a563daa-sft10-zero3-sdpa-8k-truncated-smoke-noeval-20260906
```

It uses only the temporary expanded smoke dataset under the server data root;
the formal SFT package is unchanged.

The resulting checkpoint is:

```text
/gsdata/home/wza/image-factual-verifier-v2-data/training/checkpoints/raw-history-smoke10-a563daa-sft10-zero3-sdpa-8k-truncated-smoke-noeval-20260906/v0-20260906-180347/checkpoint-10
```

The run completed 10/10 steps with exit code 0, final logged loss `0.007511`,
complete checkpoint I/O, and no residual training processes. The profile records
`smoke_only=true`, `run_mode=smoke_only`, and zero validation rows by design;
`passed_production_gate=false` is intentional. The four scheduler warnings
were independently audited as `wrapper_false_positive`: DeepSpeed global step
10, scheduler epoch 10, skipped steps 0, and the expected learning-rate
sequence.

## Direct QA

The server Direct QA smoke is stored at:

```text
/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/direct-qa-current-prompt-smoke10-20260906-4e56145
```

Ten requests completed with zero engineering errors and native thought on all
ten. The private-gold audit classified one as strong, four as usable, and five
as rejected.

The final evaluator-only migration package is local at:

```text
C:\Users\wangza\ifv-direct-qa-portable-final-20260906-r8
```

It was built from commit `a563daa`, contains two test-set tar shards, prompt,
runner, audit tools, Windows/Linux wrappers, and manifest hashes, and sets
`training_prohibited=true`. It is intended for another Codex and server to run
the Direct QA comparison after an API model key is supplied.

## Validation

- `PYTHONPATH=training python -m pytest training/tests -q`: 117 passed.
- `python -m pytest test_direct_qa_portable_package.py -q`: 2 passed.
- `python -m compileall -q src scripts training/ifv_training`: passed.
- The known full legacy suite failures belong to removed reducer/graph fixtures
  and old runtime API assumptions; those interfaces are not restored.

## Migration and Data Rules

- Source edits are made only in the local Windows checkout.
- The server checkout is updated only by Git fast-forward to a pushed commit.
- Server data, traces, logs, and checkpoints remain under `/gsdata`; only code,
  documentation, and the explicitly requested Direct QA package are local.
- No API key, server password, private gold, or construction metadata is stored
  in the repository.

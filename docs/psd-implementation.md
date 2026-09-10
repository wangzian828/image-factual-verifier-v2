# IFV Privileged On-Policy Self-Distillation (PSD) Implementation

This repository implements the IFV adaptation of Essam Sleiman's
[Privileged On-Policy Self-Distillation](https://canvas.inc/research/privileged-self-distillation)
(PSD) as a verifier-gated data path. A failed rollout is only raw experience;
it becomes a repair target after a hint-conditioned continuation of the frozen
round-start policy passes the task verifier.

## Model roles

PSD keeps three roles separate:

1. The **hint constructor** may be a stronger model and may inspect
   training-only privileged information. It proposes the intervention but its
   token distribution is never a self-distillation target.
2. The **frozen self-teacher** is the round-start student policy. It receives
   the original student-reached prefix plus the verified hint and supplies the
   sparse top-20 token distribution.
3. The **trainable student** starts from the same round-start checkpoint and
   receives the original prefix without the hint.

`PSDModelRoles` binds these identities and rejects a self-teacher/student
checkpoint or policy mismatch.

## Runtime contract

1. Qwen Chat Completions records the raw model-visible request, native tool
   call, tool result, request IDs, and optional numeric top-k capture.
2. `FailureSite` points to the original `state.all_steps[i]` policy decision.
3. A repair run reconstructs the exact archived request from the immutable
   runtime store. Missing runtime archive or media is a hard failure.
4. Hinted teacher and unhinted diagnostic continuations start from the same
   failed state; the hint is appended only to the teacher request.
5. A bounded suffix only creates diagnostics. Acceptance requires a complete
   hinted teacher episode produced by the frozen round-start policy.
6. The unhinted student may still fail. That is the behavior PSD is intended
   to repair, so its diagnostic result never gates candidate acceptance.

Set `IFV_CAPTURE_POLICY_TOKENS=1` and `IFV_POLICY_TOPK=20` for the original
Qwen rollout and for the teacher repair. The capture is fail-closed: missing
numeric token IDs, log-probabilities, or complete top-k positions leave the
candidate pending/rejected rather than being inferred from text.

## Training gates

- Only `causal_episode_pass` rows enter primary PSD targets.
- The original no-hint source rollout must be explicitly scored as failed.
- `hinted_local_pass` must come from an `ifv-psd-local-verification-v1`
  task-verifier artifact bound to the selected repair step. Tool-call presence
  is not a verifier.
- The complete hinted episode must match the private expected verdict, pass
  strict trace audit, and require no downstream patch.
- L5/exact-action scaffolds are never primary PSD targets.
- Missing prompt/completion/top-k capture remains pending or rejected; it is
  never reconstructed from the canonical trace.
- The verifier decides admission only. Training supervision remains the
  hinted frozen self-teacher's token distribution.

## Failure localization

Strict-audit and runtime action failures retain their exact existing step
bindings. Structurally legal but semantically wrong traces may additionally use
an `ifv-psd-semantic-localization-v1` report from a named task verifier. Each
candidate must bind a student-reached `state.all_steps[i]`, be marked
recoverable, include observed evidence, and use one of these categories:

- `wrong_search_direction`
- `incomplete_event_verification`
- `unsupported_similarity_extrapolation`
- `ignored_repeated_no_match`
- `evidence_interpretation_error`

If multiple candidates exist, the verifier must mark exactly one as selected.
A verdict mismatch by itself cannot place the repair at Judgment.

## Commands

Use the repository root on `PYTHONPATH` and the training package on the Python
path:

```powershell
$env:PYTHONPATH="training"
python -m ifv_training locate-psd-failure `
  --trace <trace.json> `
  --audit <strict-audit.json> `
  --semantic-verification <semantic-localization.json> `
  --output <failure-site.json>

python -m ifv_training build-psd-candidates `
  --run-dir <server-run> `
  --train-cases <train-manifest.jsonl> `
  --output-dir <candidate-package>

python -m ifv_training verify-psd-episode `
  --source-trace <failed-no-hint-trace.json> `
  --hinted-trace <complete-hinted-teacher-trace.json> `
  --gold <private-gold-row.json> `
  --local-verification <local-task-verification.json> `
  --repair-step-id <student-reached-step-id> `
  --output <verification.json>
```

`verify-psd-episode` is intentionally fail-closed. It scores the source as an
explicit failure, validates the local task-verifier artifact, and then runs the
strict trace audit and private-gold scorer on the complete hinted teacher
episode. A bounded suffix cannot become `causal_episode_pass`.

The server-side repair driver is:

```powershell
python scripts/run_psd_repair_driver.py `
  --trace <server-trace.json> `
  --audit <server-audit.json> `
  --image <server-image> `
  --gold <server-private-gold-row.json> `
  --public-context <server-public-context.json> `
  --semantic-verification <semantic-localization.json> `
  --policy-provider qwen_local `
  --policy-model <round-start-qwen-model> `
  --round-start-checkpoint <round-start-checkpoint-id> `
  --hint-constructor-provider <provider> `
  --hint-constructor-model <hint-constructor-model> `
  --output-dir <new-server-repair-run>
```

It writes `repair_attempts.jsonl` and a runtime archive. Without a bound
`ifv-psd-repair-verification-bundle-v1`, attempts remain pending. Bundle rows
are keyed by hint index and exact hint hash; each may point to a local verifier
artifact, a complete hinted teacher trace, and an optional unhinted student
diagnostic trace.

The rollout, runtime archive, images, and checkpoints stay on the server. Only
code and documentation belong in this repository.

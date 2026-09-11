# IFV Privileged On-Policy Self-Distillation (PSD) Implementation

This repository implements the IFV adaptation of Essam Sleiman's
[Privileged On-Policy Self-Distillation](https://canvas.inc/research/privileged-self-distillation)
(PSD) as a verifier-gated data path. A failed rollout is only raw experience;
it becomes a repair target after a hint-conditioned continuation of the frozen
round-start policy passes the task verifier.

The implementation was compared against upstream commit
`778be78bdac582b51a975ff819046583aad383e0`; see
[`psd-upstream-parity-audit.md`](psd-upstream-parity-audit.md).

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
provider, checkpoint or policy mismatch. The repair driver additionally binds
the served model path to an immutable checkpoint-manifest SHA-256 through an
`ifv-qwen-serving-profile-v1` artifact.

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
  is not a verifier. The artifact also binds the source-trace hash and exact
  hint hash, and contains named checks plus observed evidence.
- The complete hinted episode must match the private expected verdict, pass
  strict trace audit, and require no downstream patch.
- The local verifier and complete hinted episode must contain the SHA-256 of
  the exact teacher prompt/completion token IDs that become the training
  target. A passing episode cannot validate another sampled action.
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

python -m ifv_training verify-psd-round-rollout `
  --round-index 1 `
  --run-dir <server-run> `
  --train-cases <train-manifest.jsonl> `
  --serving-profile <serving-profile.json> `
  --round-start-checkpoint-manifest <checkpoint-manifest.json> `
  --output <round-rollout-gate.json>

python -m ifv_training build-psd-candidates `
  --run-dir <server-run> `
  --train-cases <train-manifest.jsonl> `
  --rollout-gate <round-rollout-gate.json> `
  --output-dir <candidate-package>

python -m ifv_training verify-psd-episode `
  --source-trace <failed-no-hint-trace.json> `
  --hinted-trace <complete-hinted-teacher-trace.json> `
  --gold <private-gold-row.json> `
  --local-verification <local-task-verification.json> `
  --repair-step-id <student-reached-step-id> `
  --hint-sha256 <exact-hint-sha256> `
  --teacher-prompt-sha256 <teacher-prompt-token-sha256> `
  --teacher-completion-sha256 <teacher-completion-token-sha256> `
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
  --policy-serving-profile <serving-profile.json> `
  --round-start-checkpoint <round-start-checkpoint-id> `
  --round-start-checkpoint-manifest <checkpoint-manifest.json> `
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

The rollout gate binds the completed run manifest, reward rows, rollout groups,
train allowlist, serving profile and immutable round-start checkpoint manifest.
Candidate construction re-hashes those inputs, so a bank cannot be changed or
silently reused after admission. After the optimizer run, register its new
checkpoint manifest and close the round:

```powershell
python -m ifv_training complete-psd-round `
  --rollout-gate <round-rollout-gate.json> `
  --training-profile <psd-run/profile.json> `
  --output-checkpoint-manifest <new-checkpoint-manifest.json> `
  --output <round-completion.json>
```

Round 2 and later must pass the preceding `round-completion.json` to
`verify-psd-round-rollout --previous-round-completion`. The next round-start
manifest must be byte-identical to the preceding output manifest and the
rollout run ID must be new.

## Sparse loss and launch gate

Teacher top-20 probabilities are multiplied by each target's effective row
weight. The loss sums contributing completion-token cross-entropies and then
averages across datums; it never divides by total weight mass, which would
erase target weighting. Both source kinds are mandatory. In the published
Qwen3.5 PSD recipe, every repair target and every retained preservation
assistant-step target has weight 1.0; aggregate repair and preservation mass is
not rebalanced. This matters when one successful preservation rollout contains
multiple retained assistant steps.

`training/scripts/train/run_psd_topk.sh` runs the datum-manifest/hash gate,
profile gate, ms-swift plugin forward/backward smoke, eight-rank distributed
SP loss smoke, frozen-environment preflight and checkpoint storage preflight
before model loading. During training it runs the same resource sampler and
watchdog used by SFT; afterward it validates checkpoint I/O and refuses a run
that misses the production gate. It also accepts an optional resume checkpoint.

The checked-in profiles use LoRA rank 32 and 128K SP8 across all eight GPUs.
The production profile matches the upstream published optimizer recipe:
top-20, learning rate `4e-5`, 32 unique targets per optimizer step, five epochs,
gradient clipping at 1.0 and seed 0. The custom loss splits `[T,K]` targets with
ms-swift's own Ulysses/ring ordering and gathers only scalar per-position losses;
it never gathers `[T,V]` logits. Cross-entropy is recomputed in bounded chunks
during backward, so it does not retain a full fp32 softmax.

Run the one-step memory probe before the production profile. Passing the CPU
distributed smoke proves ordering and gradient scaling, not 128K A100 capacity;
the latter remains gated on an idle eight-GPU real optimizer step.

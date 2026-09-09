# IFV OPSD Implementation

This repository implements the IFV adaptation of privileged self-distillation
as a verifier-gated data path. It does not treat a successful local action as a
training target by itself.

## Runtime contract

1. Qwen Chat Completions records the raw model-visible request, native tool
   call, tool result, request IDs, and optional numeric top-k capture.
2. `FailureSite` points to the original `state.all_steps[i]` policy decision.
3. A repair run reconstructs the exact archived request from the immutable
   runtime store. Missing runtime archive or media is a hard failure.
4. Teacher and student continuations start from the same failed state; the
   hint is appended only to the teacher request.
5. Continuation is bounded and retains every raw history step. A caller must
   run the normal full-episode scorer and strict trace audit before marking a
   repair as `causal_episode_pass`.

Set `IFV_CAPTURE_POLICY_TOKENS=1` and `IFV_POLICY_TOPK=20` for the original
Qwen rollout and for the teacher repair. The capture is fail-closed: missing
numeric token IDs, log-probabilities, or complete top-k positions leave the
candidate pending/rejected rather than being inferred from text.

## Training gates

- Only `causal_episode_pass` rows enter primary PSD targets.
- The recorded and private expected binary verdicts must match.
- Strict trace audit must pass and no downstream patch may be required.
- L5/exact-action scaffolds are never primary PSD targets.
- Missing prompt/completion/top-k capture remains pending or rejected; it is
  never reconstructed from the canonical trace.

## Commands

Use the repository root on `PYTHONPATH` and the training package on the Python
path:

```powershell
$env:PYTHONPATH="training"
python -m ifv_training locate-opsd-failure `
  --trace <trace.json> `
  --audit <strict-audit.json> `
  --output <failure-site.json>

python -m ifv_training build-psd-candidates `
  --run-dir <server-run> `
  --train-cases <train-manifest.jsonl> `
  --output-dir <candidate-package>

python -m ifv_training verify-opsd-episode `
  --trace <complete-student-trace.json> `
  --gold <private-gold-row.json> `
  --local-pass `
  --output <verification.json>
```

`verify-opsd-episode` is intentionally fail-closed. It requires a complete
student episode with a terminal binary judgment and report, then runs the
current strict trace audit and private-gold process scorer. A bounded suffix or
teacher-only trace cannot become `causal_episode_pass`.

The server-side repair driver is:

```powershell
python scripts/run_opsd_repair_driver.py `
  --trace <server-trace.json> `
  --audit <server-audit.json> `
  --image <server-image> `
  --gold <server-private-gold-row.json> `
  --public-context <server-public-context.json> `
  --model <large-qwen-model> `
  --output-dir <new-server-repair-run>
```

It writes `repair_attempts.jsonl` and a runtime archive. Supply
`--student-episode-trace` only after a complete no-hint student replay; without
it, attempts remain pending and are not primary PSD data.

The rollout, runtime archive, images, and checkpoints stay on the server. Only
code and documentation belong in this repository.

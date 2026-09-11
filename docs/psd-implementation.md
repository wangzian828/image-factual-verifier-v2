# IFV Privileged On-Policy Self-Distillation (PSD) Implementation

> Current acceptance status and remaining gates are maintained in
> [psd-completion-checklist.md](psd-completion-checklist.md). Implemented does
> not mean a real PSD round or a capability improvement has been demonstrated.

## Automatic repair verification and recovery (2026-09-12)

`psd_gemini_judge.py` now implements the training-only semantic localizer and
PSD repair judge. This is separate from the final-answer evaluation judge.
It sends actual archived images through native Gemini Interactions, preserves
raw failed/empty tool observations and checks source failure, the selected
decision, a grounded complete repaired episode and procedural hint safety.
Positive reviews need literal source/repaired-step evidence. Source, hint,
token IDs, full episode, private reference, prompt, model and image hashes bind
the artifact. Runtime checks cannot be overridden by a model's boolean answer.

The driver auto-localizes semantic failures and judges persisted repairs by
default. `--skip-auto-judge` explicitly leaves manual-verification work pending.
`--train-cases` and `--source-access-policy` are mandatory; the latter must
match the original rollout bank and is reused for tools and final strict audit.
The hint constructor receives native image blocks, not base64 inside JSON prose.

Use the same inputs plus `--resume` to continue. Completed hint proposals,
teacher episodes and judge responses are saved before subsequent validation.
Malformed or negative completed judge responses are not resampled until they
pass. Only failed/incomplete generations have bounded retries, with new archive
namespaces. Offline finalization itself makes no provider calls.

`scripts/postprocess_psd_training.py` derives training-only rewards and strict
audits from private gold; it does not export perception or SFT examples.
`scripts/continue_psd_canary.py --root <prepared-training-canary> --snapshot
<frozen-base-snapshot>` resumes the concrete rollout → candidates → repair →
judge → target → datum diagnostic. Every input case must be on the original
train split, with held-out case/image exclusions checked at preparation.
It does not launch production training.

> H20 update (2026-09-11): image conditioning is now carried through immutable
> `psd_media` tensors, exact-token Transformers teacher scoring and the native
> Qwen3.5 Swift template. Missing or changed media is rejected. The real 9B CPU
> canary passed top-20 scoring, a LoRA optimizer step, visual gradients and
> adapter/optimizer reload. This is not a 128K H20 capacity measurement.
> See [the parity audit](psd-upstream-parity-audit.md) for validation scope.

## Multimodal execution

The repair driver reconstructs the actual source and teacher requests and
requires identical ordered image inputs. Preservation assembly reconstructs
each passing step's own archived request. Repeated images stay repeated. Only
archived image bytes are accepted; remote image URLs are never refetched.
The processor's tensors, grids, configuration and ordered image hashes are
stored under the server run's `media/` directory. Every target, cache and datum
binds this artifact. The student receives pixels and token IDs, not hint text.

For visual targets use `collect-psd-topk --backend transformers --device cuda:0`
with the existing target/profile/checkpoint-manifest arguments. This loads a
frozen local round-start model (and its adapter when applicable), directly
scores the banked IDs plus pixels and computes completion-position top-20.
It never decodes/re-encodes the completion. `--device cpu` uses Transformers'
reference kernels for diagnostic runs. Token-only vLLM Completions remains
available for text-only targets and explicitly rejects visual targets.

Use `training/configs/psd/qwen3.5-lora-r32-h20-sp4-128k.env` as the H20 PSD
profile. Set `IFV_PSD_PROFILE_MODE=memory_probe` for one step before production.
The cap remains 131072; SP4 cooperates on one target and accumulation 32
preserves the upstream optimizer batch. No empirical memory-utilization target
is invented before the real GPU probe. Training is launched only after the
existing baseline/SFT/post-training evaluation sequence and the required
training-only repair admission gates; evaluation cases are never PSD data.

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

If the online continuation did not return all top-20 positions, collect them
from the same frozen round-start Qwen deployment without decoding or
re-encoding the banked action:

```powershell
python -m ifv_training collect-psd-topk `
  --targets <target-package/targets.jsonl> `
  --serving-profile <serving-profile.json> `
  --round-start-checkpoint-manifest <checkpoint-manifest.json> `
  --output-dir <server-topk-cache> `
  --topk 20

python -m ifv_training materialize-psd-topk `
  --targets <target-package/targets.jsonl> `
  --cache <server-topk-cache/teacher_topk_cache.jsonl> `
  --output-dir <materialized-target-package> `
  --topk 20
```

The text-only collector forces `teacher_prompt_ids + completion_ids` through the vLLM
Completions endpoint and reads prompt log-probabilities at the completion
positions, matching the upstream scorer. It accepts only loopback vLLM,
verifies the serving profile, checkpoint path, checkpoint-manifest hash and
returned token IDs, and fsyncs each successful row. Re-running the same command
in the same output directory skips verified rows and retries only missing or
failed targets. `--limit` is for a bounded collection smoke, not a complete
training package.

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
  --candidate <one-repair-seed.json> `
  --trace <server-trace.json> `
  --audit <server-audit.json> `
  --image <server-image> `
  --gold <server-private-gold-row.json> `
  --train-cases <original-train-case-split.jsonl> `
  --source-access-policy <original-rollout-source-policy.json> `
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

For Gemini, use the native Interactions path and provide the credential only
through the process environment:

```powershell
$env:GEMINI_API_KEY="<secret>"
python scripts/run_psd_repair_driver.py `
  <other arguments> `
  --hint-constructor-provider gemini `
  --hint-constructor-model <available-gemini-hint-model> `
  --hint-constructor-wire-api interactions `
  --hint-constructor-thinking-level low
```

The Gemini request uses a strict `hints[]` JSON response schema and creates no
interaction-history dependency. The proposer receives observed traces, images
and allowlisted checker feedback, **not private references or private judge
explanations**. `--private-context` is used only by the local leakage audit;
`--gold` remains mandatory for independent verification. Gemini probabilities never
enter the PSD target: numeric top-20 supervision must come from the frozen
round-start Qwen service.

By default `--search-mode feedback --repair-attempts 6 --proposal-rounds 12
--search-seconds 3600` runs one hint/complete continuation/judge cycle at a time.
The next proposer sees previous actual executions, failed checker bits and hint
audit codes. Invalid/repeated hints cannot run Qwen again. Already locally
verified advice is kept verbatim; wrong-anchor feedback triggers re-localization.
Success, exhausted budgets and an explicit no-further-hint decision stop search.
Malformed/pending judgments and execution errors pause the same child attempt.
`--resume` verifies completed snapshots and reuses provider caches; changing
inputs or budgets requires a new search directory. The time budget is soft,
checked between rounds so it never kills an in-flight paid call.

IFV's task is one image with one selected source decision, not BFCL's series of
user turns. Additional advice remains at that original anchor; we do not inject
downstream hints or train patched-prefix states as fresh original-policy data.
This is a documented domain adaptation, not exact multi-turn slate parity.

The search root writes aggregate `repair_candidates.jsonl` and
`repair_attempts.jsonl`, plus `search-state.json` and an explicit stop manifest.
Each child under `rounds/round-NN/` preserves its own native archives, proposals,
completed continuations, judge results and complete teacher episode. It writes
`repair_candidates.jsonl` with the localized canonical step/ID,
`repair_attempts.jsonl`, complete generated teacher episodes under
`episodes/`, and a runtime archive. The failed source suffix is removed before
the hinted replacement branch is serialized; the driver itself carries the
frozen teacher through terminal Judgment. An optional no-hint resample is only
a diagnostic (`--run-student-diagnostic`) and is disabled by default because
the original no-hint rollout has already established the failure.

Pass the driver's emitted `repair_candidates.jsonl` to `assemble-psd-repairs`.
When semantic localization moves the repair away from the seed's provisional
anchor, the emitted candidate is rebound to the actual source step and retains
`parent_candidate_id`. Its ID and the attempts' IDs then join correctly. The
original candidate bank is not edited.

With `--search-mode single --skip-auto-judge`, attempts without a bound
`ifv-psd-repair-verification-bundle-v1` remain pending. The automatic judge
normally creates this bundle. Bundle rows are keyed by hint index and exact hint hash and point to a
real local task-verifier artifact. Because that artifact binds token hashes
known only after generation, finish it offline in the same run directory:

```powershell
python -m ifv_training finalize-psd-repair-run `
  --run-dir <server-repair-run> `
  --source-trace <failed-no-hint-trace.json> `
  --gold <private-gold-row.json> `
  --verification-bundle <verification-bundle.json> `
  --require-all
```

Finalization makes zero provider/tool calls, verifies the already persisted
teacher episode, preserves `repair_attempts.pre-finalize.jsonl`, and atomically
updates `repair_attempts.jsonl`. Thus an engineering retry never resamples a
successful repair action merely to attach its later verifier result.

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

PSD checkpoints are LoRA adapters. For a later round, serve the immutable base
model with that adapter as the effective policy rather than trying to load the
adapter directory as a standalone model:

```bash
export IFV_VLLM_LORA_ADAPTER=<previous-round-adapter-checkpoint>
export IFV_CHECKPOINT_MANIFEST=<previous-round-checkpoint-manifest.json>
training/scripts/serve/start_vllm_qwen35.sh \
  <immutable-base-model> <new-round-profile-id> 8901 8 131072
```

The generated serving profile records the effective `model_path` (adapter),
the `engine_model_path` (base model), `deployment_mode=lora_adapter`, and the
checkpoint-manifest digest. Rollout, repair and top-20 collection all address
the adapter's profile ID, so the frozen self-teacher is the previous round's
actual policy.

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
profile gate, ms-swift plugin forward/backward smoke, configured-rank distributed
SP loss smoke, frozen-environment preflight and checkpoint storage preflight
before model loading. During training it runs the same resource sampler and
watchdog used by SFT; afterward it validates checkpoint I/O and refuses a run
that misses the production gate. It also accepts an optional resume checkpoint.

The current H20 profile uses LoRA rank 32 and 128K SP4 across four GPUs.
Historical A100 SP8 profiles are not the current deployment.
The production profile matches the upstream published optimizer recipe:
top-20, learning rate `4e-5`, 32 unique targets per optimizer step, five epochs,
gradient clipping at 1.0 and seed 0. The custom loss splits `[T,K]` targets with
ms-swift's own Ulysses/ring ordering and gathers only scalar per-position losses;
it never gathers `[T,V]` logits. Cross-entropy is recomputed in bounded chunks
during backward, so it does not retain a full fp32 softmax.

Run the one-step memory probe before the production profile. Passing the CPU
distributed smoke proves ordering and gradient scaling, not 128K H20 capacity;
the latter remains gated on an idle four-GPU real optimizer step.

Before launch set `IFV_PSD_SERVING_PROFILE` and `IFV_PSD_ROUND_START_MANIFEST`.
For a later round also set `IFV_PSD_INITIAL_ADAPTER` to that exact round-start
adapter: Swift receives `--model BASE --adapters PREVIOUS --load_args false`.
The initialization gate verifies target roles and actual model-file hashes.
A new round uses a fresh optimizer; the optional same-round resume checkpoint
instead restores optimizer, scheduler and RNG. Resume also requires an ancestor
`psd-resume-binding.json` with the identical datum manifest, initialization,
training profile, seed and clipping settings. Use a new experiment ID/output
directory for the resumed launch and pass the old checkpoint as the fifth
argument; existing logs/checkpoints are never overwritten. Round completion additionally
requires a positive optimizer step, changed weight artifacts, intact artifact
hashes, bound nonempty dataset hashes and complete resumable training state.

`training/scripts/probe/psd_real_datum_cpu_smoke.py` is a separate diagnostic:
it selects the shortest complete admitted repair and preservation datum,
performs real 9B LoRA updates and compares uninterrupted/resumed next-step
weights, plus previous-adapter initialization with a fresh optimizer. It never
truncates a datum and is not a production-batch, GPU-capacity or quality result.

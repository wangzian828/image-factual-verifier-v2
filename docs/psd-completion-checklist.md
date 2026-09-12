# PSD completion and acceptance checklist

This checklist supersedes any earlier claim that PSD was fully accepted.
Implementation, synthetic tests, live-provider tests and a real training round
are separate milestones. A component test is not an end-to-end acceptance.

## Current evidence (2026-09-12)

### Acceptance update at 21:41 China time

- The isolated real H20 SP4 optimizer/capacity probe completed at
  `/volume/ybo/wza/logs/psd-h20-sp4-real-optimizer-probe-20260912-r5`.
  It encoded all 46 immutable real datums and completed `global_step=1` in
  55.84 seconds with four-way sequence parallelism, FSDP2 and LoRA rank 32.
  Per-process GPU-memory peaks were 16,310/16,310/16,310/17,490 MiB; every H20
  reached 100% sampled utilization, with no OOM or non-finite failure. Loss
  166.6 is the configured weighted token-loss sum per datum, consistent with
  the earlier real CPU diagnostic; it is not mean cross entropy. This probe
  used `save_strategy=no`, so it proves the real optimizer/update path and
  capacity gate, not a saved adapter fingerprint.
- The immutable feedback run r2 passed a new full cost/integrity audit at
  `/volume/ybo/wza/runs/psd-feedback-canary-20260912-r2/feedback-audit.json`:
  33 proposal requests, 9 Qwen continuations, 42 unique completed Gemini judge
  requests, no changed bound snapshot and no audit error. Provider usage was
  2,048,820 proposer input tokens plus 1,382,860 judge input tokens (643,877
  cached), 11,817 judge output tokens and 152,326 thought tokens. Currency cost
  is deliberately not estimated.
- A real interrupted assembly recovery passed using one admitted repair and one
  real preservation trajectory. Resume quarantined exactly one partial output,
  rebuilt 13 files/531,109,297 bytes in 11.377 seconds, and a third invocation
  reused the completed package without rebuilding or making provider calls.
  The temporary 531 MB test output and quarantine were removed after hashes and
  `/volume/ybo/wza/runs/psd-real-interrupted-assembly-recovery-20260912-r1/result.json`
  were retained.
- The bounded three-case live pool exercised real Gemini proposals, Qwen
  trajectories and Gemini task judges concurrently. The first invocation used
  160.609 wall seconds for 381.117 summed case seconds, an observed **2.37x**
  speedup. It exposed a genuine judge binding bug when one repair emitted a
  format-error retry before its token-bound tool action. Commit `25c6246`
  allows only such parser retries, identifies exactly one bound replacement,
  and still refuses to skip a valid policy decision. Cached resume finalized
  that case as rejected, not admitted; the final three-case audit passes with
  zero errors, 3 proposal calls, 2 Qwen continuations and 5 unique judge calls.
  The 1-attempt/1-proposal probe had zero admissions and is a throughput/control
  test, not a repair-rate estimate.
- The historical r2 cost audit also showed near-quadratic proposer context
  growth because every prior full repaired trajectory was copied into every
  later prompt. Feedback-search v2 now retains all prior hints/checker outcomes
  but only the latest full repaired trajectory. Tests prove input history is
  immutable and earlier full traces are omitted. Offline projection across all
  33 real historical requests reduced canonical feedback JSON from 2,581,437
  to 1,211,913 bytes (**53.05%**) without a provider call. Resumable pool timing
  is now accumulated across invocations instead of reporting the final resume
  alone. The complete Python 3.12 H20 PSD suite is **184 passed** at `6375d1f`.
- H20 capacity/update, real asynchronous case scheduling, judge recovery,
  interrupted materialization and detailed cost gates are now closed. The
  remaining end-to-end acceptance gates are a complete next PSD training round,
  fresh on-policy rollout from its new checkpoint, and held-out capability/
  regression measurement. No production PSD training round has been started.

### Acceptance update at 02:47 China time

- Real 9B CPU diagnostic completed in 5,574.53 seconds. Its `result.json`
  reports `passed=true`, a real parameter update and nonzero visual gradients.
  Restoring adapter, optimizer, scheduler and RNG reproduced the uninterrupted
  second-step losses **and all trainable parameter fingerprints exactly**.
  Previous adapter weights with a fresh optimizer initialization also passed.
  This used the original frozen 45-datum bank (one complete repair and one
  complete preservation datum selected for the diagnostic); it is not a
  production PSD round or an H20 capacity measurement. Keep checkpoint-1 and
  the evidence; do not rerun this completed diagnostic.
- New public-only feedback canary r2 finished all three fixed failed cases:
  `e20bf448737e2505`: 12 proposals, 5 continuations, 0 accepted, proposal budget
  exhausted (2539.68s); `6f8580432ff9805b`: 12 proposals, 3 continuations,
  1 accepted (622.06s); `7e5b935397d714f0`: 9 proposals, 1 continuation,
  1 accepted (644.84s). Seven completed continuations remained rejected.
  These are three diagnostic cases, not a general repair-rate estimate.
- New assembly admitted **2 repair + 44 preservation targets**, all with frozen
  top-20 distributions; 46 datums, zero materialization rejections, 13,440 loss
  positions, maximum sequence 26,331 tokens. Independent read-only input
  preflight passed with zero errors and a 131,072-token ceiling. Datums SHA-256:
  `6b102370b2904d8f3dbd4b6b4144041c4f8b4fe21389678c18ae9ff518e4788a`.
  Final status is `search_complete_datums_materialized`; original bank unchanged.
- Full regression at029a7d0: 154 passed. Remaining gates: detailed per-attempt
  rejection/cost audit, interrupted intermediate assembly recovery, real case-pool
  throughput, isolated H20 SP4 capacity/update, full next-round fresh-rollout
  integration and held-out capability improvement. Single-anchor adaptation and
  the round-end barrier remain; do not claim BFCL multi-turn slate parity or
  fully asynchronous cross-version training. No production PSD training started.
- Main baseline remained running without interruption: 1,652 / 1,682 trace
  files at this check, not necessarily 1,652 successful cases. Main judge/SFT/
  post-evaluation ordering and the frozen 1,527 formal denominator are unchanged.

The subsections below retain earlier historical checkpoints; this update
supersedes their pending CPU and feedback-canary statuses.

### Reopened audit: feedback search was missing

The previous implementation generated a batch of hints and judged them only
afterwards. Transport retries were NOT verifier-guided repair search. Earlier
"aligned"/"complete" statements about this part were incorrect. A second gap
was sending private references to the proposer: the upstream **agentic**
controller sees execution/checker feedback, not the reference answer (other
upstream oracle/ablation scripts have different information boundaries).

New work, not yet real-provider accepted:

- `run_psd_repair_driver.py` defaults to sequential feedback search: 6 complete
  continuation attempts, at most 12 proposal rounds, 3600-second soft budget
  checked between rounds. Each proposal is judged before the next is generated.
  Invalid/repeated proposals do not spend a continuation but cannot loop forever.
- Only observed source/repaired steps, original images and allowlisted checker
  flags reach the proposer. Private reference, expected label and private judge
  explanations remain in verifier/audit artifacts. No gold-conditioned prior
  hints are imported into the new public-only diagnostic.
- Wrong-anchor feedback triggers fresh bound localization; a locally verified
  procedural hint is preserved verbatim if remaining downstream problems need
  additional advice. Completed unsuccessful calls remain cached as negatives.
  Pending/invalid judges and transport failures pause the current child attempt;
  resumption does not turn them into rejections or resample a completed judgment.
- SHA-bound per-round inputs/outputs, exclusive process-owned writer lock,
  finite budgets, explicit stop reasons and aggregation of accepted/rejected
  attempts. A successful repair stops further case search immediately.
- **Domain boundary:** IFV currently distills one selected source decision per
  image task, with its original unhinted on-policy prefix. It is NOT a port of
  BFCL's multi-user-turn slate injector. Working advice can be extended at the
  same anchor; later-turn hints/patched prefixes are not silently admitted as
  independent fresh on-policy targets. This is an explicit algorithmic
  adaptation, not a claim of exact upstream multi-turn parity.
- Local non-PyTorch PSD suite: **123 passed**. Full server suite and live fixed
  three-case feedback-search acceptance remain pending at this checkpoint.
- New real diagnostic root planned separately from frozen inputs:
  `/volume/ybo/wza/runs/psd-feedback-canary-20260912-r1`. The existing 45-row
  datum bank and its ongoing CPU optimizer diagnostic must not be rewritten.

Further live audit and fixes:

- The first feedback diagnostic (`psd-feedback-canary-20260912-r1`, PID187974)
  was stopped after finding a PSD-only visual endpoint configuration omission.
  `llm_base_url` was bound but `vlm_base_url` was not, so visual tools fell back
  to unused port8899. Main evaluation and the original on-policy canary explicitly
  use8901 and were not affected. All diagnostic artifacts remain; r1 is NOT a
  valid repair-quality comparison. No main/CPU process was stopped.
- `_policy_runtime_kwargs` now binds policy AND visual provider/model/wire API/
  endpoint to the frozen profile, and forwards the normal runtime timeouts.
- `psd-visual-endpoint-smoke-20260912-r2/result.json`: **passed**, actual archived
  visual-tool request returned `status=success`, `answer_status=observed` through
  the existing public-tool adapter and correct image SHA. The initial standalone
  probe omitted the runtime's hidden image binding and failed before a provider
  call; that diagnostic wiring was fixed, not counted as a model failure.
- Fresh fixed three-case search is running under
  `/volume/ybo/wza/runs/psd-feedback-canary-20260912-r2` (PID188817 at launch).
  The first real attempt was rejected for answer-bearing advice and unsuccessful
  visual reasoning; the actual execution/checker bits reached round01. This is
  evidence of feedback flow, **not evidence of a successful repair yet**.
- Server suite at17ff302: **142 passed**; at4313a1a: **152 passed**, including
  case-pool/live-service/storage gates. The live serving-root/128K gate also
  matched the running pretrained service without changing it.
- Added bounded completion-order asynchronous **case** scheduling, preserving
  serial dependencies within a case and the fixed checkpoint across the round.
  Controlled tests prove fast cases progress/refill while the first case waits,
  and one failed case does not cancel others. Live throughput improvement has
  NOT been measured. Current in-flight diagnostic remains serial; do not restart
  paid calls solely to switch its scheduler.
- Added live model-root/context checks before policy generations and a 2GiB
  per-search artifact soft cap. This is not stale-policy asynchronous RL training;
  round-end data validation and the training phase remain separate.

Upstream execution sources:
[agentic strategy](https://github.com/essamsleiman/psd/blob/main/experiments/bfcl/agentic/system_prompt.md),
[attempt budget](https://github.com/essamsleiman/psd/blob/main/experiments/bfcl/agentic/run_agentic_repair.py),
[checker feedback](https://github.com/essamsleiman/psd/blob/main/experiments/bfcl/agentic/attempt_runner.py).

### Earlier component and single-round evidence

- Implemented automatic native-multimodal Gemini semantic localization and PSD
  repair judging, literal-evidence/anchor/full-episode/hash checks, native image
  hints, persisted continuations/reviews, bounded generation retries and offline
  re-finalization without trajectory resampling.
- Implemented original source-exclusion-policy binding for both repair tools
  and final strict audit. Student initialization must match the frozen teacher,
  including the previous-round adapter. Round completion checks real weight
  artifacts and resumable optimizer/scheduler/RNG state.
- Server regression at `be8234e`: **116 PSD tests passed**, plus launcher shell
  syntax and completed-bank offline resume checks. Local focused suites also
  passed. Source edits/commits originate locally; server only fast-forward
  pulls. Agent implementation is unchanged.
- Live Gemini `gemini-3.1-pro-preview` repair-judge diagnostic: **25/25 correct**,
  comprising 21 distinct synthetic controls and four repeated inputs. Zero
  false admissions in this diagnostic. Seven initially rejected literal-JSON
  evidence formats were fixed and validated from cached responses, without
  extra API sampling. This is not a real-trajectory quality estimate.
- Official training archive: `jiashuhong/factcheck_train`,
  `factcheck_train-8490-20260907.tar.gz`, SHA-256
  `2fda3ca7144d899e355fcbbaa4e6b93350878dbf5225ab423402123d5e37a448`.
  Stream-verified all 24,127,361,006 compressed bytes; retained 137,227,221
  metadata bytes, no archive/image download stored. Reused delivered images
  using hardlinks. Original train split and full held-out case/image exclusions
  are preserved.
- Fresh real on-policy canary: **8/8 completed, 8/8 strict-audit passes, zero
  engineering errors, zero incomplete token captures**. Five correct complete
  trajectories form preservation candidates, three wrong complete trajectories
  form repair seeds. The fixed case selection was independent of model outcomes.
  Frozen pretrained 9B files and serving profile are hash-attested. This bank
  is a diagnostic training subset, not the official evaluation set.
- Real repair admission: one repaired trajectory passed all local, complete
  episode, private-gold, strict-audit and exact-token binding gates; two sampled
  continuations were rejected; one seed produced no admissible hint. The two
  hints for the same road-direction case were discriminated correctly: the
  uncorrected `real` response failed, while the grounded corrected `fake`
  response passed. This small diagnostic does not estimate general repair rate.
- The third attempt's review contained an explicitly elided quote. Its ordered
  literal fragments were checked against the same original field; cached
  re-finalization made no API calls and rejected the attempt. No fuzzy matching
  or changed judge vote was used.
- **45/45 frozen top-20 datums passed materialization**: 1 repair and 44 assistant
  preservation turns from five passing episodes; 13,417 supervised positions,
  zero rejected datums. Sequence lengths: min 4,370, median 14,355, p90 20,518,
  p95 23,782, max 26,331. Both source kinds keep per-target weight 1; there is
  no aggregate 1:1 rebalancing or truncation.
- Real 9B CPU optimizer/resume diagnostic is running on the full 15,740-token
  repair and 4,369-token preservation inputs, not the old short synthetic
  fixture. Until its final `result.json` passes, **do not mark the optimizer
  acceptance gate passed**. H20 capacity, production PSD rounds and held-out
  capability improvement have not been demonstrated by this diagnostic.
  Last check: process remained actively computing, no final `result.json` yet;
  the 45-datum immutable input was revalidated successfully without API calls.
- The live replay found and fixed duplicated leading system messages in the
  archived-request → native-history adapter (vLLM returned HTTP 400). The fix
  removes only the identical transport copy and rejects changed/interior system
  messages. Agent code was not modified. Failed calls remained archived.

Server evidence roots:

- `/volume/ybo/wza/runs/psd-gemini-repair-controls-20260912-r1`
- `/volume/ybo/wza/runs/psd-base-snapshot-20260912`
- `/volume/ybo/wza/runs/psd-real-training-canary-20260912-r2`
- `/volume/ybo/wza/data/psd-official-train-metadata-20260912`
- `/volume/ybo/wza/runs/psd-real-datum-9b-cpu-20260912`

Existing half-hour `h20` follow-up now includes CPU result validation, local
fix/test/commit if needed and a deferred isolated H20 SP4 one-step probe after
the main experiment. It must not promote diagnostic adapters or start an
unrequested production PSD training round. Local scheduling requires the app
and computer to remain running. The main experiment's ordering is unchanged.

### Final live diagnostic results — 2026-09-12

- The real 9B CPU optimizer/resume diagnostic completed successfully at
  `/volume/ybo/wza/runs/psd-real-datum-9b-cpu-20260912/result.json`. It used one
  admitted repair and one preservation datum, observed non-zero visual gradients,
  reproduced the second step exactly after loading model/optimizer/scheduler/RNG
  state, and verified next-round old weights with a fresh optimizer. This closes
  the CPU optimizer and full-state resume acceptance gate only; it is not an H20
  capacity result or a capability result.
- The fixed three-case feedback search completed at
  `/volume/ybo/wza/runs/psd-feedback-canary-20260912-r2`. Two cases converged to
  one admitted repair each; one exhausted all 12 proposal rounds with no admitted
  repair. The run retained nine complete continuations and seven rejected
  attempts rather than resampling judge votes.
- The resulting immutable bank has **46/46** complete top-20 datums: 2 repair and
  44 preservation targets, 13,440 supervised positions, zero materialization
  rejections, and a maximum sequence length of 26,331 tokens. The original
  frozen bank remained unchanged.
- Remaining gates are intentionally unchanged: isolated H20 SP4 optimizer/
  throughput/capacity validation must wait for the main SFT/evaluation sequence,
  and no production PSD round or held-out capability improvement has been run.

The abandoned preparation directory without `-r2` contains only a failed
projection attempt; it is not an accepted dataset/run.

## Scope and ordering

Do not modify the Agent runtime, consume evaluation cases as training data, or
interrupt the baseline → judge → SFT → post-SFT evaluation → judge experiment.
Edit/test/commit locally, push GitHub, and only fast-forward pull on the server.
Data, API artifacts, archives and checkpoints stay under `/volume/ybo/wza`.
Retain the 131072-token upper bound; do not silently truncate a PSD target.

## Acceptance matrix

| Stage | Required acceptance | Initial audit |
| --- | --- | --- |
| Fresh rollouts | Frozen training split, current checkpoint, numeric token capture, raw request archive | Provenance gates exist; no real PSD training rollout collected on H20 |
| Localization | Earliest evidenced recoverable policy decision, exact source-index/hash binding | Validator exists; automatic semantic verifier missing |
| Hint | Procedural, no answer/query/exact action leakage; private context isolated | Proposer/audits exist; no real repaired training case accepted |
| Replay | Original prefix/media unchanged; frozen current policy supplies replacement and complete suffix | Runtime adapter exists; full real-case acceptance missing |
| PSD judge | Direct image/evidence review, selected-step repair, complete grounded episode; reject fake repairs | Only external artifact validation; automatic judge missing |
| Admission | Source failed, local verifier passes, correct complete episode, strict audit, exact sampled token hashes | Existing gates require integration, persistence and tamper tests |
| Targets | Repair decision only; passing-turn preservation; unhinted student; immutable ordered media | Numeric/media component tests pass; real repair assembly not demonstrated |
| Loss/update | Frozen same-checkpoint top-20, upstream per-target weights, LoRA rank32, SP4 | Real 9B short synthetic CPU update/reload passes; not a real PSD round |
| Resume | Preserve successful calls, retry transport failures only; optimizer/scheduler/RNG resume; next round uses new weights and fresh rollouts | Top-k/offline finalizer exist; live judge resumption missing |
| Outcome | Independent held-out before/after evaluation and judge, repair/preservation breakdown | Not measured; loss reduction is not capability improvement |

## Work sequence

1. Complete the automatic semantic localizer and multimodal PSD judge; bind
   requests, images, full episode, prompt version, model and token hashes.
2. Wire them into repair generation and offline finalization, saving each
   expensive result before later stages can fail. No successful trajectory
   resampling just because a verifier timed out; no retry-until-pass judging.
3. Add positive/negative/tamper/resume tests and live blinded synthetic controls
   (including label-only, unavailable-tool, leaked-answer and prompt-injection).
   Report these as synthetic diagnostics, not real repair quality.
4. Locate training public cases plus independent private reference material;
   collect fresh bounded current-Qwen rollouts and validate real repairs. The
   transferred SFT messages alone are not raw on-policy archives/private gold.
5. Assemble verified repairs plus preservation, collect frozen top-20,
   materialize and validate datums, execute an optimizer/resume acceptance when
   GPU scheduling permits, then validate next-round checkpoint provenance.
6. Update this report with exact commands/results/commits and unresolved gates.
   Never substitute fabricated artifacts or evaluation data for missing inputs.

Reference: [PSD paper](https://www.canvas.inc/research/privileged-self-distillation)
and upstream revision `778be78bdac582b51a975ff819046583aad383e0`.

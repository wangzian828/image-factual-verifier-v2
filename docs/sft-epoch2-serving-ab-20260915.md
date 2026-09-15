# Epoch-2 checkpoint and serving acceleration investigation

## User decision

On 2026-09-15 the user requested evaluation of epoch 2 from the same three-epoch
SFT run, then explicitly paused formal inference until engineering acceleration
has been investigated. The user then explicitly authorized the sequence
**finish optimization, validate, then run the full epoch-2 Agent test set**.
The completed epoch-3 serving processes were stopped after confirming all four
replicas and the gateway had zero in-flight/waiting requests. Old checkpoints,
exports and evaluation results remain intact.

The source is `checkpoint-2056` in
`/volume/ybo/wza/checkpoints/h20-sft-merged4929-agent-v2-3epoch-fullstate-20260914/v0-20260914-200132`.
Its trainer state says step 2056, epoch 2, planned max steps 3084.
The explicit `--completed-epoch 2` export mode accepts this saved epoch boundary;
the export command still requires final training completion by default.

BF16 export completed and passed finite-tensor and full-parameter-update audits:
9,409,813,744 parameter elements; optimizer, scheduler and four RNG states
preserved. Output:
`/volume/ybo/wza/exports/h20-sft-merged4872-epoch2-step2056-20260915`.
Export manifest SHA-256:
`55dfb77f56cb175573c5e966816c8a0a0c384385190ae5b62795963f681fbd28`.
This checkpoint is not the separately trained SFT-4872 one-epoch model.

## Research and bounded measurement

The earlier successful-attempt sample attributed 80.99% of end-to-end time to
main-model requests, versus 15.45% to tools; a separate service window was
decode-dominated. These are sampled diagnostics, not the full test cohort.

The pinned [vLLM 0.18.1 optimization guide](https://github.com/vllm-project/vllm/blob/v0.18.1/docs/configuration/optimization.md)
distinguishes chunked-prefill latency and overall throughput. Its multimodal
processor cache and async scheduling already exist; they are not new proposed
features. [APC](https://github.com/vllm-project/vllm/blob/v0.18.1/docs/features/automatic_prefix_caching.md)
can reduce repeated prefills, not decode. The [Qwen recipe](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md)
still calls Mamba align caching experimental, and its tested large MoE hardware
is not our dense 9B/H20 configuration.

| Profile | GPU | Max sequences | Prefill token budget | Prefix cache |
| --- | ---: | ---: | ---: | --- |
| baseline-s8-b32k | 0 | 8 | 32768 | off |
| s16-b32k | 1 | 16 | 32768 | off |
| s16-b8k | 2 | 16 | 8192 | off |
| apc-s8-b32k | 3 | 8 | 32768 | on, mamba align |
| apc-s16-b32k (follow-up) | 3 | 16 | 32768 | on, mamba align |

All profiles use the same immutable epoch-2 BF16 weights, TP1, 131072 context,
32-image limit, memory fraction 0.94, existing native-tool/reasoning parsers,
and normal CUDA-graph/async execution. No quantization, reduced images, shortened
thinking, reduced formal output budget or Agent prompt/tool modification is introduced.

The fixed workload comprises 16 archived model requests: four decision points
from each of the four pre-existing epoch-3 smoke histories. Each request has
1–21 image slots, with repeated attachments retained; this is not 21 distinct
images. Requests keep 32768 output and 8192 thinking limits and their sampling
configuration. Identical reconstructed payloads are used in every profile.
Historical persistence can change JSON formatting; this is a common replay
benchmark, not an exact original-wire/token reconstruction claim.

The first natural-length harness hit a missing `jsonschema` dependency after
successful HTTP responses. Its validation failures and zero token totals were
harness bugs, not model failures. Those records remain preserved, and are **not
used to claim throughput**. Validation now imports the dependency before work,
records usage before validation and retains responses. The dependency was added
only to the isolated artifact directory, not the training/serving environment.

Following the user's clarification, the accepted engine benchmark forces exactly
1024 generated tokens per request (`min_tokens=max_tokens=1024`, `ignore_eos`),
uses streaming with usage validation and concurrency 16, and records server-side
TTFT, queue/prefill/decode metrics and preemption counters. This diagnostic-only
forced continuation is not an Agent trajectory, reasoning-quality test or formal
32768/8192 configuration change. All 256 fixed-token responses passed exact-count
validation. No external tools or gold are used.

A new cache salt isolates first-pass prefixes; later passes replay exact inputs.
Image processing and kernel initialization are warmed separately. Prefixes can
still be shared between requests within a new-salt batch. Warm replay is an
upper-bound reuse scenario, not a complete Agent cases/hour measurement.

| Engine configuration | New-prefix batch after initialization (s) | Repeated-input batch (s, two runs) | Warm throughput (output token/s) |
| --- | ---: | ---: | ---: |
| Original seq8 / 32768 / no APC | 94.79 | 95.44 / 95.43 | 171.68 / 171.69 |
| seq16 / 32768 / no APC | 87.04 | 86.95 / 87.26 | 188.42 / 187.75 |
| seq16 / 8192 / no APC | 88.91 | 88.27 / 88.20 | 185.61 / 185.76 |
| APC align / seq8 / 32768 | 71.15 | 39.46 / 39.80 | 415.16 / 411.69 |
| APC align / seq16 / 32768 | 61.77 | 31.83 / 32.07 | 514.71 / 510.89 |

Each row processes 16,384 output tokens per pass. The newly started APC16 engine's
initial first request pass took 124.25 seconds, including first-use work absent
from the already-exercised other engines; it is retained, not hidden. A separate
post-initialization new-salt pass (`fixed-token-apc16-cold-primed`) took 61.77
seconds / 265.22 output token/s. This is the comparable APC16 new-prefix entry.
The selected configuration improves measured throughput by about 1.53x on that
batch and 2.99x on repeated inputs. These are not predictions of Agent speedup.
All observed initial comparison preemption deltas were zero; long formal traces
still require monitoring. No MTP head, quantization or vLLM upgrade is introduced.

Artifacts: `/volume/ybo/wza/benchmarks/sft2056-serving-ab-20260915`.
Workload SHA-256: `be40282d75e702bb1d019652648ab896d2fdef50cc086bbd205b282d3c03fc2e`.
Baseline startup initially hit the Unix-domain socket path-length limit; its
temporary path was shortened under `/volume/ybo/wza/tmp`, with the failed log
preserved. This is a harness startup issue, not a checkpoint or speed result.

No benchmark automatically promotes a profile. After the measurements, a candidate must
pass real complete Agent trajectories, including reasoning termination,
multi-image handling, tool calls and final report contracts, before full inference.

## Authorized deployment and evaluation

Selected: APC `mamba-cache-mode=align`, seq16, batched tokens 32768, four TP1
replicas with the unchanged least-inflight gateway. The tested compilation cache
is reused; no checkpoint tensors are modified. All benchmark services were
drained and stopped before the four formal services were launched.

- Serving: `/volume/ybo/wza/inference/sft2056-epoch2-apc-20260915`.
- Results: `/volume/ybo/wza/runs/eval/qwen35-sft2056-epoch2-agent-formal1527-20260915`.
- Artifact scripts: `/volume/ybo/wza/training-artifacts/sft2056-evaluation-20260915`.
- Alias: `ifv-qwen3.5-9b-sft-2056`; frozen runtime commit remains
  `1d61abb41f147e871b5361ee3e917aaa5c06ccc5`.

At 19:21 China time, four-case smoke was running, not yet accepted. The isolated
pipeline runs smoke, four-replica two-distinct-image diagnostic, strict audit,
then full inference only with trace/export-hash-bound acceptance. It checks real
external subcall success per trace and unchanged main-request generation budgets;
no gold or judge influences this gate. Full inference preserves four smoke
successes, runs the other 1522 at concurrency40 and only retries missing/failed
cases at 32/24/16. The reporting denominator stays1527.

The idle guard polls each5 seconds and only sends local compute pulses after
30 seconds below10% utilization **and both running/waiting counters are zero**.
Metrics errors fail closed. If all GPUs are truly released, its fallback uses
64GiB/35%-duty matrix workers. Memory occupancy is never treated as utilization.
Before future service/training transitions, stop this guard and its scoped
`idle-guard/matrix-workers` first. The benchmark's guard was stopped at handoff.
The existing hourly app monitor was updated to epoch2 and the two pending judges,
without creating a duplicate schedule or restarting completed epoch3 inference.

### Smoke accepted; full inference running (19:26 China time)

All four complete smoke trajectories passed the strict audit and were bound to
the epoch2 export. All four replicas also passed the separate two-distinct-image
diagnostic. Main archived requests retained output32768 / think8192 and enabled
thinking. Observed: 83 native `tool_calls` finishes and four final `stop` finishes,
no `length`/`abort`, maximum1403 output tokens per request, maximum3862 reasoning
characters (not tokens), and up to25 image slots including repeated attachments.

Successful real external subcalls included23 Serper text searches,21 Jina
reranks,3 reverse searches,13 image searches,4 OCR calls,10 page fetches and10
Gemini extracts. One Jina page-fetch error remains preserved; this is not a
claim of zero external-tool failures. Acceptance is in `smoke-acceptance.json`.

The pipeline advanced to `full_inference`, attempt0, concurrency40, with1522
pending cases at start and four smoke successes preserved. This is an active
full run, not1526 completed predictions; no new BAcc/SESR is available yet.
Only failed/missing cases may be retried, using the existing frozen protocol.
Local implementation and measured engine report were pushed in commit52051ea;
this acceptance addendum is a subsequent local-only documentation update.

## Speculative decoding scope

The epoch-2 export contains 760 tensor keys and no `mtp`/`nextn` keys. The original
Base checkpoint contains 775 keys including 15 MTP auxiliary tensors. Thus a
compatible trained auxiliary head may be available for a later isolated test;
it has not been added to the SFT export or validated. Never randomly initialize
an auxiliary head or replace the SFT main weights with Base.

[Speculative decoding](https://github.com/vllm-project/vllm/blob/v0.18.1/docs/features/speculative_decoding/README.md)
targets decode latency, but performance and numerical/logprob behavior require
validation. It is not justified to promise a speedup, especially under high load
or for PSD's exact-token/top-k use. First compare the lower-risk scheduling and
cache profiles above; MTP is not enabled in this initial A/B.

## Concurrent judge submission

The two completed sources each have 1526 unique successful cases and retain the
1527 reporting denominator. They use the unchanged v3 full-material projection,
Gemini 3.7 Flash, low, max output 32768.

- Qwen epoch-3 candidate pack SHA-256:
  `c803e97312ad1ad741c263098e7e5b3894e67690efde92f93253a1306e4cec43`;
  19,897 retained raw actions.
- Gemini 3.1 Pro Agent candidate pack SHA-256:
  `bcc68b6b261054c6cfebecc2a3d288f4f8e1b1b98f06ca75f162666c9d3e2834`;
  11,276 retained raw actions.

Input/journal directory:
`/volume/ybo/wza/evaluation/sft3084-gemini31pro-v3-judge-20260915`.
Collector output:
`/volume/ybo/wza/runs/eval/sft3084-gemini31pro-batch-judge-v3-low32k-20260915`.

Most images are submitted inline. The shared 17,581,496-byte GIF cannot fit the
inline envelope; an unchanged-file upload returned explicit FileStorageBytes
quota 429 (20 GiB). Its two judge requests are isolated and deferred, not dropped.
No shared provider files were deleted. Later Batch creates also returned 429:
28 accepted jobs covered Qwen 74 and Pro 75 cases at that point, not all 3052.
The submitter now backs off on explicit 429 and resumes only unsubmitted shards;
uncertain creates stop for reconciliation. The collector independently retains
completed jobs. Current progress must be read from the journal, not inferred from
a launched process. The exact account-level Batch quota dimension is unconfirmed.

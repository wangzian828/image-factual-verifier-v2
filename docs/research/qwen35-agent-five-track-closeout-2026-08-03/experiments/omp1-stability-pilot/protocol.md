# H8 protocol: OMP1 20-step stability and winner-driven pilot

Status: locked before execution; hardware execution is gated by H1.

## Question

After H1 selects workers0 or workers4 under OMP1, does that exact production
profile remain stable over 20 optimizer steps and provide a safe launch profile
for the formal pilot?

## Fixed conditions

- physical GPUs 4-7 only, after ownership and idleness checks;
- `OMP_NUM_THREADS=1` and `IFV_OMP_NUM_THREADS=1`;
- ZeRO-3 optimizer CPU offload with parameters resident on GPU;
- FlashAttention, language/VIT activation checkpointing, max length 32768;
- global batch 8 and unchanged sample order;
- content-addressed processor cache in readonly mode with 100% hits;
- full optimizer/scheduler/RNG checkpoint state retained.

## Prepared profiles

Prepare equivalent 20-step workers0 and workers4 profiles. They are not a new
worker sweep: only the H1 winner may execute. Prepare a formal pilot profile for
each worker count and let the runner select the matching one from the signed H1
closeout summary.

## Measurements

- treat steps 1-5 as startup and steps 6-20 as the steady window;
- report steady step-wall mean, median, p90, maximum, coefficient of variation,
  and stall count;
- report useful samples/s, per-GPU mean/quantile utilization and peak memory;
- report process-tree CPU/RSS pressure, encoded-cache hits/misses, scheduler
  audit, evaluation time, checkpoint phase time, and resume integrity;
- distinguish train, eval, save, and finalize wall time.

## Acceptance

No OOM, NaN/Inf, NCCL error, cache miss, scheduler-gate failure, checkpoint
integrity failure, or unexplained steady-state stall. The pilot launcher must
fail closed when H1 is missing, failed, or names an unknown winner. Historical
legacy-20 artifacts cannot select a profile or checkpoint.

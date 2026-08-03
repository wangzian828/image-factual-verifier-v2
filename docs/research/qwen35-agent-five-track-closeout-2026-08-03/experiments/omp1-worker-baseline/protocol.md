# H1 protocol: OMP1 worker selection and production gate

Status: locked before execution.

## Prediction

Workers0 may remain simplest and fastest, but workers4 can improve lazy image
processing now that CPUAdam is restricted to one OpenMP thread. No winner is
assumed.

## Fixed conditions

- physical GPUs 4-7 only, after ownership and idleness checks;
- ZeRO-3 optimizer CPU offload, parameters on GPU;
- global batch 8, max length 32768, FlashAttention;
- language and visual activation checkpointing;
- cached train/validation rows and identical ordering;
- `IFV_OMP_NUM_THREADS=1`.

## Runs

1. two-step workers0;
2. two-step workers4 with persistence and prefetch2;
3. promote the faster stable candidate to ten steps with validation/save;
4. resume checkpoint 10 to step 11 and audit optimizer/scheduler/RNG.

## Acceptance

No OOM, NaN, NCCL error, cache drift, save failure, or resume failure. Select the
faster useful-sample throughput; choose workers0 when the difference is within
run variance.

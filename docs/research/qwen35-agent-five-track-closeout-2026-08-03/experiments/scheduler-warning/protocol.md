# H3 protocol: scheduler-order warning

Status: locked before execution.

## Prediction

The warning is likely caused by wrapper bookkeeping around DeepSpeed CPUAdam, but
this must be demonstrated by observed optimizer parameter change, optimizer step
count, scheduler epoch, and learning-rate sequence.

## Tests

1. inspect the frozen Transformers/Accelerate/DeepSpeed call path;
2. run a minimal deterministic optimizer/scheduler probe;
3. capture the real SFT logging sequence for steps 1 and 2;
4. add a fail-closed profile audit when the observed LR sequence differs from the
   configured schedule.

## Acceptance

Either remove the warning through a correct integration change, or document it as
cosmetic with a deterministic test proving that optimizer state and LR advancement
are correct. Never suppress the warning without proof.

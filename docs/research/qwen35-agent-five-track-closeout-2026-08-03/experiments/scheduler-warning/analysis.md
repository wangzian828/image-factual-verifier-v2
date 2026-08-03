# H3 analysis: DeepSpeed scheduler warning

Status: confirmed wrapper false positive.

Accelerate invokes the real `DeepSpeedEngine.step()` at the accumulation
boundary. `DeepSpeedOptimizerWrapper.step()` is intentionally a no-op, but
Transformers observes that wrapper and emits the PyTorch scheduler-order warning.

The audit verified:

- one-step run: DeepSpeed global step 1, scheduler `last_epoch=1`, no skipped
  optimizer step;
- ten-step run: DeepSpeed global step 10, scheduler `last_epoch=10`, expected LR
  progression from `1e-5` to zero, no skipped step;
- structural source audit: the wrapper no-op and real engine step are both
  present in the expected locations.

The warning remains visible. Production gates require a passing
`wrapper_false_positive` sidecar whenever it appears.

Artifact:

- `/gsdata/home/wza/image-factual-verifier-v2-data/training/logs/scheduler-audit-411a8f1-20260803/`


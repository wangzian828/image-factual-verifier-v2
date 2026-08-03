# H1 analysis: OMP1 worker and production closeout

Status: hardware execution pending.

Implemented:

- dedicated OMP1 workers0 and workers4 two-step profiles;
- readonly content-addressed encode cache in both profiles;
- workers0/workers4 ten-step and resume profiles;
- a fail-closed closeout runner that:
  1. stops only the managed Qwen3.5 vLLM service;
  2. compares workers0 and workers4;
  3. selects workers4 only when it is more than 5% faster;
  4. runs 10-step train/eval/save;
  5. resumes checkpoint 10 to step 11;
  6. requires 100% encode-cache hits and passing production sidecars;
  7. restores the managed service on success or failure.

A detached waiter is active. It requires the frozen reviewed-52 batch to be
complete and physical GPUs 6/7 to be below 1 GiB before invoking the runner.
The launcher then repeats the full 4–7 idle check. Physical GPU 6 remains owned
by another user's process and is not touched.

Pending artifact root:

- `/gsdata/home/wza/image-factual-verifier-v2-data/training/logs/omp1-closeout-five-track-0e45675-20260803/`


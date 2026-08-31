# H1: 1Cat-vLLM SM70 preflight

## Hypothesis

The released 1Cat-vLLM V100/SM70 wheel can start the local Qwen3.5-9B multimodal checkpoint in an isolated Python 3.12 environment and satisfy the minimum IFV API contract.

## Prediction

The service starts on a reserved loopback port, lists the model, handles a short text request, accepts one image request, emits parsable structured output and tool calls, and remains stable under a bounded concurrent smoke test.

## Fixed initial configuration

- Engine: 1Cat-vLLM v1.3.0 prebuilt wheel.
- Hardware: GPUs 0-3 first, then full eight-GPU layout only after the contract gate passes.
- Precision: FP16 for the initial dense checkpoint.
- No speculative decoding, FP8 cache, or AWQ assumptions in the first smoke.
- First memory target: 0.88 GPU-memory utilization; sweep later only after correctness and stability pass.

## Measurements

Record startup result, dependency versions, model support outcome, API smoke results, GPU memory allocation, and any concrete incompatibility trace.

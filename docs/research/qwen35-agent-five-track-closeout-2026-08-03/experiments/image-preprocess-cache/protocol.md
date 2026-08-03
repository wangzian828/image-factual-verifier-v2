# H2 protocol: content-addressed image preprocessing cache

Status: locked before execution.

## Prediction

Caching deterministic image/processor outputs on the training data root will
reduce repeated open/decode/resize/processor work, especially across epochs and
resume, without changing model inputs.

## Cache key

The key must include original image SHA-256, processor/model revision, template
revision, max length, image-token or pixel limits, non-thinking-prefix setting,
and the cache schema version.

## Correctness gate

- compare uncached and cached processor outputs for representative image rows;
- require identical tensor names, shapes, dtypes, values, labels, and loss mask;
- fail closed on missing/corrupt artifacts or key mismatch;
- keep artifacts outside Git and write a manifest with hashes.

## Performance gate

Measure cold build time, warm hit rate, item latency, CPU RSS, GPU utilization,
and two-step training throughput. Promotion requires a material warm-run gain and
zero semantic mismatch.

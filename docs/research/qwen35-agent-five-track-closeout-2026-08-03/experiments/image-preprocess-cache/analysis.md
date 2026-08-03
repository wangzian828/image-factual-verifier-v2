# H2 analysis: content-addressed processor cache

Status: confirmed.

The cache is installed at the ms-swift `LazyLLMDataset.__getitem__` boundary and
stores the complete encoded payload rather than only Arrow rows or image paths.
Its key includes image SHA-256, processor/model/template identity, model config
asset hashes, length and image-token limits, non-thinking-prefix state, framework
versions, and schema version.

Server validation:

- cold process: 526 rows, 343 unique misses/writes, 183 same-run hits;
- exact comparison: 12 real rows, including multimodal tensors, passed;
- warm fresh process: 526/526 hits, zero misses, zero encodes, zero writes;
- contract digest:
  `d3d721394c77856218db995ad9ef4d844dffd8cf026705de938c662126098e6f`.

Production profiles use namespace
`qwen35-pilot30-v3-img1024-ml32768-v1` in `readonly` mode. A cache miss therefore
fails training instead of silently rebuilding a different contract.

Artifacts:

- `/gsdata/home/wza/image-factual-verifier-v2-data/training/logs/encode-cache-prewarm-c73d0c1-cold/`
- `/gsdata/home/wza/image-factual-verifier-v2-data/training/logs/encode-cache-prewarm-c73d0c1-warm/`


# Primary-source survey

## Local primary sources

- ms-swift 4.4.2 installed source: `LazyLLMDataset` performs item-time template
  encoding over cached rows.
- Transformers Trainer installed source: warning originates from scheduler state
  observing no optimizer step before `lr_scheduler.step()`.
- DeepSpeed 0.19.2 installed source and ZeRO checkpoint files: resumable optimizer
  state is partitioned by rank and dominates checkpoint size.
- IFV reviewed-52 historical snapshots and current reducers: mechanism validation
  is available without rerunning retrieval.

## Public primary sources

- ModelScope ms-swift repository and documentation for cached datasets and SFT.
- Hugging Face Transformers repository for Trainer scheduler ordering.
- DeepSpeed documentation for ZeRO checkpointing and optimizer offload.
- Google Gemini API documentation for retryable failures and structured outputs.

Public sources provide compatibility context. Decisions in this project are gated
by the frozen local versions and measured artifacts.

# Algorithm and curriculum lessons

Sources:

- Search-R1: https://arxiv.org/abs/2503.09516
- OpenSearch-VL: https://arxiv.org/abs/2604.15838
- LiteResearcher: https://arxiv.org/abs/2604.17931
- WebAgent-R1: https://arxiv.org/abs/2505.16421
- MM-DeepResearch: https://arxiv.org/abs/2601.14567

Cross-project patterns:

1. Cold-start SFT remains common. Pure online RL often fails to discover tool
   protocols reliably.
2. Plain sparse GRPO is not a safe default. Recent projects use RLOO/REINFORCE
   baselines, DAPO, GSPO, DUPO, adaptive rollout allocation, dynamic filtering, or
   staged environments.
3. Search infrastructure failures must be separated from policy failures.
   OpenSearch-VL's fatal-aware masking matches IFV's existing engineering-failure
   records.
4. Stable cached or local-web environments precede live web RL.
5. Search count, format, and efficiency are usually shaped separately from answer
   correctness.

Recommended first IFV reward surface:

- outcome correctness;
- construction evidence-chain recovery;
- stop calibration and bounded cost;
- protocol/engineering validity.

Keep these components separately logged. Only the optimizer-facing scalar combines
them.


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
5. Released training code is dominated by terminal outcome reward. Format is often
   auxiliary; search-count, retrieval, and efficiency rewards are frequently disabled
   or absent in the public recipes.
6. The strongest relevant multi-reward precedent is R-Search: score an explicitly
   selected Evidence bundle by asking a frozen verifier to answer from that bundle,
   then compare the verifier answer with gold. It does not score hidden reasoning.

Recommended first IFV reward surface:

- outcome correctness;
- selected-basis recoverability from raw Evidence only;
- protocol/engineering validity and fatal masking.

Keep search count, stopping, claim-level labels, and cost as diagnostics until a real
same-policy ranking calibration shows that they improve ordering without reward
hacking.

Keep these components separately logged. Only the optimizer-facing scalar combines
them.

# Research Log

| # | Date | Type | Summary |
|---|---|---|---|
| 1 | 2026-07-17 | bootstrap | Locked a comparative study of mature search-agent training stacks. Primary evidence is restricted to official repositories, documentation, released code, and papers. The comparison separates SFT support from online agent RL instead of assuming one framework must own both. |
| 2 | 2026-07-17 | inner-loop | H1/H2: surveyed text and multimodal search agents. Vision-DeepResearch, OpenSearch-VL, and LiteResearcher all split cold-start SFT from online RL. No single framework had both the strongest Qwen3-VL SFT path and the best external-runtime integration. |
| 3 | 2026-07-17 | inner-loop | H3: inspected IFV provider/tool boundaries and rLLM/OpenRLHF gateways. IFV policy calls are text-only independent stages; image perception and VLM comparison are environment calls. rLLM can record independent OpenAI-compatible calls as per-session Steps without rewriting the runtime. |
| 4 | 2026-07-17 | outer-loop | Direction DEEPEN. Select ms-swift for full multimodal SFT. Select rLLM with veRL backend as the primary RL candidate, OpenRLHF as fallback, and AReaL 2.0 as a later scale-up option. Before implementation, require tool/schema passthrough and a four-GPU memory smoke. |

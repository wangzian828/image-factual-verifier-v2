# Multimodal search-agent precedents

Sources:

- https://github.com/EvolvingLMMs-Lab/multimodal-search-r1
- https://github.com/Osilly/Vision-DeepResearch
- https://github.com/shawn0728/OpenSearch-VL
- https://github.com/HJYao00/MM-DeepResearch
- https://github.com/AQ-MedAI/SimpleSearch-VL

## MMSearch-R1

MMSearch-R1 extends Search-R1/veRL with both image and text search. The released
Qwen2.5-VL-7B recipe uses eight GPUs, GRPO, multi-turn response masks, bounded search
counts, and explicit search/format penalties. It is the closest older precedent for
IFV's mixed visual and textual tools.

## Vision-DeepResearch

Vision-DeepResearch separates training stacks:

- Qwen3-VL-8B full-parameter SFT uses Megatron-SWIFT;
- RL uses rLLM, veRL, Megatron-LM, and SGLang;
- the public RL demo is designed around eight H20-class GPUs.

Its 8B SFT script is true full-parameter training with tensor, sequence, and
activation-recompute parallelism. It uses 64K context and disables optimizer-state
saving in the published demo, so it is not directly suitable as our resume gate.

## OpenSearch-VL

OpenSearch-VL is the strongest direct Qwen3-VL-8B search-agent precedent:

- full SFT uses LLaMA-Factory, Ray, and DeepSpeed ZeRO-3;
- RL uses rLLM plus veRL, Megatron-LM, and SGLang;
- the project introduces Factorized Adaptive Rollout and fatal-aware masking so
  infrastructure/search failures do not become negative policy examples;
- the documented 8B SFT setup asks for eight A100-80GB GPUs and RL uses sixteen
  H100-80GB GPUs.

The framework split and failure masking are directly relevant. The published
hardware recipe is not.

## MM-DeepResearch

MM-DeepResearch uses Qwen2.5-VL, GSPO-style optimization, a custom asynchronous
search loop, and sixteen GPUs for 7B training. Its useful lesson is reward separation:
answer correctness, format validity, and search-efficiency shaping are distinct.

## SimpleSearch-VL

SimpleSearch-VL reports a Qwen3-VL-8B end-to-end deep-search model and a factorized
rollout method, but the repository currently contains only a notice that code and
models are forthcoming. It informs algorithm design but cannot be selected as
infrastructure.


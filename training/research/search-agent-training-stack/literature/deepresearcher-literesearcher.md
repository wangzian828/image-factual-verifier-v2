# DeepResearcher and LiteResearcher

Sources:

- https://github.com/GAIR-NLP/DeepResearcher
- https://arxiv.org/abs/2504.03160
- https://github.com/simplexai-labs/LiteResearcher
- https://arxiv.org/abs/2604.17931

DeepResearcher uses a modified veRL/Search-R1 stack and trains a text agent directly
against live web browsing. Its public implementation is useful historically, but it
depends on an older pinned stack and does not address VLM training.

LiteResearcher is more relevant architecturally:

- cold-start SFT is implemented with LLaMA-Factory;
- online RL is implemented with veRL;
- environments progress from stable knowledge APIs to a local virtual web and then
  live search;
- the authors explicitly report that pure RL from an unprepared model does not
  reliably acquire tool-use behavior.

Its released 8B recipe uses sixteen H100-class GPUs. It supports the split-stack
hypothesis and cached/local-web curriculum, but not the current four-A100 resource
envelope.


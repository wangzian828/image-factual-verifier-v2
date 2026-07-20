# Search-R1 and text-search RL

Sources:

- https://github.com/PeterGriffinJin/Search-R1
- https://arxiv.org/abs/2503.09516
- https://github.com/Agent-RL/ReSearch
- https://github.com/Agent-RL/ReCall

Search-R1 established the common open-source pattern of a policy model, a search
endpoint, rule-based answer reward, and veRL-based policy optimization. The released
system is text-only and treats the interaction as a growing token trajectory.

The later Search-R1 update is important for algorithm choice: it reports that
zero-variance groups give GRPO no learning signal and that lower-bias estimators such
as REINFORCE and leave-one-out baselines can be more reliable. This weakens any plan
that assumes plain GRPO is the default for a small, difficult search dataset.

ReSearch and ReCall stay in the same broad veRL family. They are useful precedents
for sparse outcome rewards and search behavior, but they do not establish
multimodal, evidence-chain, or external-runtime maturity.


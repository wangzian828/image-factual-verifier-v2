# Algorithm and curriculum lessons

## Corrected field view

The field has two visible generations rather than one uniform reward recipe.

1. Early Search-RL systems such as Search-R1, ReCall, DeepResearcher and the
   built-in rLLM SearchReward mostly optimize terminal answer or environment
   success.
2. Recent systems treat process credit as a primary long-horizon optimization
   problem. Released implementations now cover support-document gain, answerability
   potentials, counterfactual turn attribution, state graphs, semantic judges,
   process reward models and outcome-credit redistribution.

The previous statement that released search-agent training is simply dominated by
terminal reward was too broad. It describes the first group, not the current process-
credit literature.

## What the implementations teach

- **Gold-structured rewards are strong but narrow.** StepSearch and SearchEyes/HaPO
  obtain precise credit from support documents, search keys or entity chains. They
  fit synthetic or curated training worlds better than unrestricted live-web IFV.
- **State improvement is more informative than action appearance.** TIPS, OASES and
  TRACE ask whether a new state makes the answer more recoverable. This avoids
  rewarding a query merely because it sounds reasonable, but requires a reliable
  gold-conditioned probe.
- **Counterfactual attribution can expose misleading turns.** LOTAPO and TACO compare
  the answer with a turn removed, masked or bypassed. This is useful for image
  reinterpretation and evidence regressions, but intervention validity must be
  checked.
- **Outcome should remain the direction anchor.** CW-GRPO's safest idea is to use a
  step judge to redistribute the positive advantage of correct trajectories instead
  of letting subjective process scores replace verifiable success.
- **Failed trajectories contain useful prefixes.** TRACE explicitly shows useful
  early actions followed by a harmful late turn. Uniformly negative credit is too
  coarse, so IFV should localize blame while initially preventing a failed episode's
  process signal from reversing the overall outcome direction.
- **Context management and process evaluation are coupled.** PRInTS recursively
  summarizes the trajectory and evaluates the current step from the summary, latest
  observation and action. A process judge should not repeatedly consume the entire
  raw long context.
- **A teacher judge needs grounding and calibration.** FaithMed and CW-GRPO prove
  that LLM step judges can participate in RL. They do not prove that an unrestricted
  scalar over hidden reasoning is reliable. Evidence/turn citations, confidence,
  human ranking calibration and a second attribution channel remain necessary.
- **Framework support is not credit assignment.** Agent Lightning can capture
  intermediate reward spans, while its current official veRL path still propagates
  the final reward. rLLM likewise needs a custom transformation or estimator before
  IFV turn rewards become distinct token advantages.

## Recommended IFV reward curriculum

### Stage 0: immutable outcome and runtime gates

- verdict correctness;
- selected-basis recoverability from raw selected Evidence;
- strict protocol audit;
- fatal provider/search failure masking.

### Stage 1: offline process annotation

For every investigation turn, build a bounded packet containing the pre-turn state,
current action, returned observation and post-turn state delta. Ask a frozen Gemini
teacher for short categorical labels covering evidence gain, direction quality,
belief update and regression. Independently estimate a counterfactual or state-delta
contribution. Preserve raw scores, citations, confidence and token usage.

### Stage 2: same-policy ranking calibration

Sample K=4-8 trajectories per question and compare teacher ordering with blind human
turn labels and trajectory rankings. Reject reward dimensions that correlate mostly
with length, eloquence, search count or explicit reflection language.

### Stage 3: conservative online use

Begin with outcome-gated contribution reweighting:

- distribute correct trajectories' positive outcome advantage across useful turns;
- keep failed trajectories negative overall, but concentrate negative credit on
  misleading/regressive turns and reduce blame on useful prefixes;
- apply process credit only where the semantic teacher and attribution channel agree;
- keep format as a gate and cost as a diagnostic.

Only after this estimator is calibrated should IFV test locally positive credit for
useful turns inside failed trajectories.

## Stack implication

The SFT decision remains ms-swift plus DeepSpeed. The RL backend must now satisfy an
additional non-negotiable requirement: it must accept turn/segment-level advantages,
not merely one episode scalar. rLLM remains useful as the external-runtime gateway,
but its default veRL transformation is insufficient. The integration must either add
an IFV credit-aware transform/advantage estimator or move the rollout records into a
veRL-derived trainer where corresponding action-token spans can receive distinct
advantages.

Full implementation evidence and source links are recorded in
`reward-implementations.md`.

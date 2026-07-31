# H1 Protocol: Unmodified v4 Evidence-to-Vision Behavior

**Type:** confirmatory mechanism diagnostic  
**Locked:** 2026-07-31, before candidate rollouts  
**Runtime baseline:** `discrepancy-first-v4` at `4729301`  

## Hypothesis

On at least some high-value diagnostic cases, the existing Agent can complete:

```text
new external knowledge
-> newly recognized image-checkable distinction
-> focused visual reinspection
-> new pixel Evidence
-> changed subsequent Decision or investigation action
```

without a prompt, schema, reducer, or budget modification.

## Sample selection

Select 5-10 development cases before running them. A candidate must have:

1. a reasonable coarse or potentially mistaken initial visual interpretation;
2. retrievable external knowledge that reveals a fine-grained distinction;
3. a distinction that is in principle visible in the supplied pixels;
4. adequate or plausibly adequate image resolution;
5. an offline human-auditable expected distinction and target region.

Selection may use evaluator-private development annotations offline. Runtime input remains exactly `case_id`, `image_path`, and `image_sha256`.

## Measurements

For each canonical trace, label the first failing stage:

1. `knowledge_retrieval`
2. `distinction_recognition`
3. `reinspection_trigger`
4. `question_or_region`
5. `visual_observation`
6. `state_consumption`
7. `causal_decision_impact`
8. `complete`

A case is `complete` only if the new visual observation changes a later canonical Claim assessment, SearchHypothesis, MaterialDiscrepancy, next investigation action, stopping decision, or final verdict in a direction judged relevant to the distinction.

## Pre-registered interpretation

- No relevant knowledge retrieved: investigate search strategy first.
- Knowledge retrieved but no reinspection: investigate the Decision bridge first.
- Reinspection generated but non-discriminative: investigate request/region representation.
- Good question but observation fails: investigate image resolution, crop, or visual model.
- Correct observation but old interpretation persists: only then consider a minimal policy-consumed interpretation state.
- One or more complete cases: preserve the current architecture and study trigger reliability before adding new state.

## Exclusions

- Do not tune prompts between H1 cases.
- Do not manually supply queries or discriminative features to the Agent.
- Do not count an attractive narrative without downstream behavioral impact.
- Do not treat inaccessible details or honest `ambiguous` observations as hallucination.
- Record exploratory observations separately from the pre-registered labels.

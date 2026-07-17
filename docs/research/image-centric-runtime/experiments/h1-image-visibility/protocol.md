# H1 Protocol — Persistent Image Visibility

## Protocol amendment before real-provider execution

The original wording proposed attaching the image independently to every policy
stage. API and architecture review showed that this would repeatedly encode the same
static image and still leave isolated conversations. The implemented treatment is:

- attach the original image once to Target Planning;
- preserve one stored main Interactions chain across ReAct, Evidence Decision,
  Reflection, and Judgment;
- keep Query Concept Extraction/Replan and other auxiliary calls independent;
- use focused visual tools only when the investigation introduces a new visual
  question.

## Change

Keep the original image available to the main policy through one stateful
Interactions chain without re-uploading it at every stage.

Do not change:

- prompts beyond wording needed to acknowledge image availability;
- search providers;
- evidence extraction and provenance;
- source policy;
- action, Reflection, reinspection, or query budgets;
- target-revision ownership.

## Prediction

Gemini will retain visually salient event, relation, and scene qualifiers in Target
Planning and choose routes more consistent with the image. If later evidence reveals
a better target but the run still cannot adopt it, that supports H2.

## Measurements

- initial core target statement;
- whether salient visual relation/event qualifiers are retained;
- first three tool routes;
- final verdict;
- post-determination and low-value actions;
- strict trace audit;
- tool and LLM call counts.

## Interpretation

- Positive: target fidelity improves across heterogeneous cases without regressions.
- Partial: initial target improves, but target revision remains blocked after new
  evidence; proceed to H2.
- Negative: image access does not change target fidelity; inspect prompt/context
  composition or model transport before changing target semantics.

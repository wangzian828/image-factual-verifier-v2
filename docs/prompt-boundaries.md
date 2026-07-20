# Prompt Boundaries

Prompts express semantic judgment. The runtime owns mechanical constraints.

| Prompt responsibility | Runtime responsibility |
|---|---|
| Select an exact relevant webpage passage and judge its factual relation to a goal. | Validate the passage id, recover the original text and offsets, and enforce source policy. |
| Choose a useful image-grounded core relation. | Validate fact ids, pixel/OCR grounding, atomicity, query grounding, stable core ownership, and task budgets. |
| Choose a next investigative action from supplied state. | Enforce active tasks, one action per turn, route deduplication, source access, inspection-before-repeat, and stopping. |
| Explain a compiled result. | Compile verdicts and evidence bases from immutable state. |

Do not add a prompt rule merely because a deterministic guard is missing. Add or
repair the guard in code, then keep the prompt focused on the semantic decision the
model is uniquely suited to make.

The image defines the account under review and supplies initial clues, not the
investigation's vocabulary or search boundary. SearchHypotheses and queries may
independently seek the underlying real-world fact and may use model knowledge to
propose unverified leads. That knowledge is never Evidence: only recorded tool
observations can support a ClaimAssessment, MaterialDiscrepancy, or verdict basis.

For webpages, lack of mention is insufficient to refute a goal. A model may mark a
source as refuting only when the selected passage states a proposition incompatible
with the positive goal. The runtime records the original span rather than a model
paraphrase.

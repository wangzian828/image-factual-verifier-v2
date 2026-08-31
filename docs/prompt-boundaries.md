# Prompt Boundaries

This page describes the active `unified-react-v1` boundary. Older Claim/Task
rules remain only in dated historical plans and legacy replay code.

Prompts express semantic judgment. The runtime owns mechanical constraints.

| Prompt responsibility | Runtime responsibility |
|---|---|
| Choose the next useful action from the image, task, and compact memory. | Expose only public tool arguments, inject the image, enforce one action per turn, deduplication, source policy, budgets, and stopping. |
| Interpret inspected page passages and visual observations in thought. | Record the original tool result, classify discoveries/evidence/failures, and persist state deltas. |
| Write the final reader-facing report and bounded binary judgment. | Compile the final basis from immutable runtime state and preserve trace provenance. |

Do not add a prompt rule merely because a deterministic guard is missing. Add or
repair the guard in code, then keep the prompt focused on the semantic decision the
model is uniquely suited to make.

The image and fixed task define what is being checked. The policy may change its
question or search direction as observations arrive; it does not first create a
Claim/route/task graph. Search results and reverse-image matches are unverified
leads until a page, image, or visual observation is independently inspected.
Only the reducer's recorded Evidence can support the final report.

Visual tools are ordinary ReAct actions. Their order is not fixed, and the
original image is attached again to each direct-multimodal request without being
duplicated in the text history.

For webpages, lack of mention is insufficient to refute a goal. A model may mark a
source as refuting only when the selected passage states a proposition incompatible
with the positive goal. The runtime records the original span rather than a model
paraphrase.

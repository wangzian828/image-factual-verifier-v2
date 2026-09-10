# Prompt Boundaries

The active `unified-react-v1` path separates semantic interpretation from
mechanical enforcement without a reducer or evidence ledger.

| Prompt responsibility | Runtime responsibility |
| --- | --- |
| Interpret the image and retained raw tool results in their exact scope. | Preserve native calls, raw results, provider history, and artifact provenance. |
| Choose one useful public tool action or finish. | Hide execution-only fields and enforce schema, source policy, budgets, and one action per turn. |
| Write a bounded binary report and cite used observations. | Supply valid successful observation IDs and reject invalid citations. |

The image and fixed objective define what is being checked. The policy changes
direction through its next ReAct action; it does not create a Claim/route/task
graph. Search results and reverse-image candidates remain leads until returned
content or a comparison directly addresses the question.

Errors and access failures remain limitations. Successful empty searches remain
observations about their submitted queries. Neither is automatically evidence
for a verdict. Runtime guards should enforce deterministic protocol rules while
the prompt stays focused on interpretation and action choice.

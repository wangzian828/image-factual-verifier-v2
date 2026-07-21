# Image-Centric Runtime Findings

## Current understanding

The Qwen3.5 short-request runtime now reaches a complete bounded binary Judgment
without engineering errors on the first heterogeneous canary. The compiled basis
retains supported high-salience Claims, unresolved medium Claims, Evidence, and an
explicit gap. The remaining strict-audit failure is request-lineage bookkeeping:
Qwen corrections are real independent Chat Completions requests and therefore need
durable context-request ancestry instead of Gemini Interaction IDs.

The baseline runtime was a deterministic coordinator of isolated Gemini stages.
Initial perception saw the image, while Target Planning, ReAct, Evidence Decision,
Reflection, Query Replan, and Judgment normally consumed text-only state.

The highest-risk boundary is Target Planning: it freezes one core proposition from
a lossy visual summary. Downstream search may then be internally correct while
investigating an underspecified or wrong relation.

The first structural repair is implemented: Target Planning receives the original
image once and creates a stored main Interactions chain. ReAct, Evidence Decision,
Reflection, and Judgment inherit it through `previous_interaction_id`. Query Replan
and auxiliary extraction/tool calls remain independent. The deterministic suite is
green; real-provider target fidelity remains unmeasured.

## Lessons and constraints

- Do not add case-, person-, domain-, or keyword-specific prompt rules.
- Do not treat deterministic tests as real trajectory acceptance.
- Preserve exact Evidence provenance and engineering-failure visibility.
- Re-uploading the same image to every call is unnecessary; persistent main-chain
  visual context is the current ablation.
- Image persistence alone is not assumed to solve frozen target ownership.
- Stop only after reviewing target fidelity, evidence sufficiency, and remaining
  image-grounded uncertainty together.
- A rejected output is recoverable only when an explicit same-stage correction
  chain reaches an accepted output. Provider-specific IDs must not be fabricated;
  local Chat Completions uses context request IDs and parent request IDs.
- A provider profile must bind one endpoint for both policy and vision. Falling back
  to a shared legacy local URL can fail before Planning or silently split the model
  roles across services.
- A forced structured boundary must not advertise tools. Exact rejected arguments
  are more useful correction feedback than a generic duplicate warning.

## Open questions

- Is persistent image access sufficient to preserve target meaning?
- When should a target revision be allowed without expanding into irrelevant
  creator, title, date, or source metadata?
- Can one multimodal checkpoint replace three overlapping policy stages while
  reducing calls and improving trace clarity?

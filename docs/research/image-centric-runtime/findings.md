# Image-Centric Runtime Findings

## Current understanding

The current runtime is a deterministic coordinator of isolated Gemini stages, not
one continuous multimodal investigator. Initial perception sees the image, while
Target Planning, ReAct, Evidence Decision, Reflection, Query Replan, and Judgment
normally consume text-only state.

The highest-risk boundary is Target Planning: it freezes one core proposition from
a lossy visual summary. Downstream search may then be internally correct while
investigating an underspecified or wrong relation.

## Lessons and constraints

- Do not add case-, person-, domain-, or keyword-specific prompt rules.
- Do not treat deterministic tests as real trajectory acceptance.
- Preserve exact Evidence provenance and engineering-failure visibility.
- Image attachment alone is an ablation, not assumed to be the final solution.
- Stop only after reviewing target fidelity, evidence sufficiency, and remaining
  image-grounded uncertainty together.

## Open questions

- Is persistent image access sufficient to preserve target meaning?
- When should a target revision be allowed without expanding into irrelevant
  creator, title, date, or source metadata?
- Can one multimodal checkpoint replace three overlapping policy stages while
  reducing calls and improving trace clarity?


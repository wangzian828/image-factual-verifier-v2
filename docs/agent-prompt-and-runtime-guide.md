# Agent Prompt and Runtime Guide

The active production path is `unified-react-v1`. Runtime policy prompts are
English and are defined in `src/orchestrator/unified_prompts.py`. The exact
generated copy is [Active Agent System Prompts](active-agent-system-prompts.md).
The Chinese files are reading translations, not runtime inputs.

## 1. One ReAct loop

Each turn is:

```text
original image + fixed task + compact memory
  -> thought
  -> exactly one public native tool call
  -> tool observation and reducer state delta
```

`perceive_scene` and `ocr_with_position` are ordinary tools. The model chooses
their order and may use visual tools again later. There is no mandatory
bootstrap sequence, Planning output, Query Replan output, or separate route
graph in the active runtime.

Changing a query, page, reverse-image candidate, visual question, or stopping is
another action in the same loop. The runtime owns IDs, internal tool parameters,
deduplication, retries, budgets, failures, and state updates.

The search tools have distinct roles:

- `text_search` finds web-page candidates from a factual text query.
- `text_image_search` finds image and image-page candidates from a factual text
  query, when a named person, event, place, object, or visible text can locate
  relevant images.
- `reverse_image_search` uploads the current image for Lens or semantic image
  correspondence.

Results from all three search tools are unverified Discovery. A
`text_image_search` result must be inspected with `visit` or
`compare_with_reference` before it can contribute to Evidence.

## 2. Context and image handling

The root request of each Gemini Interaction gets one temporary controlled image
attachment. Follow-up requests reuse that image through the same provider
session; it is not uploaded again. The image is not appended to text history
and its base64 is not stored inside canonical step text. Each next request
continues from:

- the fixed objective;
- visual memory;
- unverified discoveries;
- successful Evidence;
- external/access and engineering failures;
- attempted queries, visited URLs, recent actions, open questions and budget.

Each ReAct tool call also carries the model-maintained
`investigation_progress` state. It remains `investigating` while a material
factual question is open and changes to `decision_capable_support` or
`decision_capable_refute` only when the model itself judges that the accumulated
material directly supports or contradicts the factual situation. The runtime
preserves this state and checks it on an explicit finish action; it does not
infer this status from a tool name, result field, or evidence class, and it
does not add or remove tools based on that state. Existing evidence annotations
remain available to reporting and audit. The context exposes both the global
action budget and each tool's remaining budget.

The full request/response archive remains available for audit, but is not
replayed into every policy turn. The immediately completed canonical function
result is passed to the next policy request in full textual form; only binary
transport fields and raw HTML are excluded from model text.

All images entering a vision API use JPEG with longest edge 1024 and quality 95.
This includes the original image, crops, downloaded candidate images and
single-image contact sheets. `perceive_scene` keeps its local output bounds and
normalizes the returned observation after the provider response. A recoverable
provider failure may make one additional schema-light attempt; both attempts
remain observable and a failed recovery remains `status=error`.

The active Gemini Interactions session is persistent for the whole episode.
Dynamic tool schemas may be rebuilt between actions, but the next request keeps
the previous interaction ID and submits the previous function result before the
new compact context. A newly-created session per action would lose this
continuation boundary.

`reverse_image_search`, `text_image_search`, and `crop_and_search` keep at most
three new reference-image candidates in the next multimodal request. Crops and
focused visual views are attached at the same immediate observation boundary.
They are observations or unverified candidates, not automatically Evidence.

`visit` uses a retrieve-then-bounded-extract path: Jina/direct content is cleaned
and ranked into passages, a summary/extraction model reads a bounded selection,
and exact source spans are retained for the trace. The summary input is capped
by runtime code at 24,000 characters (18,000 by default); the full page is not
placed in the Agent context.

Gemini vision structured-output requests send an API-compatible schema projection:
provider-side validation-only bounds such as `maxLength` and `maxItems` are
removed, while the local tool still enforces its own normalization and limits.

## 3. Evidence boundary

Search results, snippets, titles, URLs, source labels and guesses are leads.
Only successful visual/OCR observations, valid image comparisons, or inspected
page passages can support the final report. A similar image or background page
does not by itself establish the complete image fact.

`finish_investigation` ends the ReAct loop only after the model declares
`investigation_progress.status=decision_capable_support` or
`decision_capable_refute`. Otherwise the model continues using the same
available tools. If the global action budget is reached, the existing runtime
path ends ReAct and sends the case to Judgment directly.
Judgment then writes the binary label and reader-facing report; SFT audit remains
responsible for judging evidence quality.

## 4. Training export

The complete episode is exported in Qwen-compatible form:

```text
system
user: task + compact context
assistant: <think>...</think>
          <tool_call>...</tool_call>
tool: result + state delta
...
assistant: final report
```

The exporter keeps one complete episode and does not create one training row per
action. It reconstructs the exact request images from the runtime artifact
archive, writes portable `data:image/...;base64,...` entries, attaches each new
image at the corresponding message boundary, and deduplicates identical bytes
by SHA-256. Missing ReAct thought is classified as action-only/RL material
rather than filled with fabricated text; a thought-free structured Judgment
does not invalidate otherwise complete ReAct supervision.

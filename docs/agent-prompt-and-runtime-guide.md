# Agent Prompt and Runtime Guide

The production path is `unified-react-v1`. Runtime prompts are defined in
`src/orchestrator/unified_prompts.py`; the generated English copy is
[Active Agent System Prompts](active-agent-system-prompts.md).

## One retained-history loop

```text
original image + fixed objective + retained provider history
  -> thought
  -> exactly one public native tool call
  -> raw function result
  -> next interaction in the same session
```

There is no mandatory visual bootstrap and no Planning, route graph, Replan,
Reflection, or Discrepancy Decision stage. The model changes direction by
choosing another tool in its next turn. It may finish when another available
action is unlikely to materially change the bounded judgment.

The runtime exposes public tool arguments, injects execution-only fields,
checks source policy and budgets, and records the result. It does not summarize
results or classify them into discoveries, evidence, and failures. Successful,
empty, malformed, and error results all remain observable in raw history.

## Provider history and images

One `InteractionSession` spans the full investigation and Judgment. The root
request receives the controlled original image in `direct_multimodal` mode.
Each next request sends the immediately completed `function_result` once and
uses `previous_interaction_id`; earlier observations remain in provider-side
history rather than a reconstructed workspace.

In `separate_vlm` mode, visual tools receive the image and policy reads their
raw results. Candidate images, crops, and focused views are attached only at
the observation boundary that produced them. Canonical text never embeds image
base64; artifact references and SHA-256 preserve provenance.

The mechanical context contains only the objective, image availability, and
remaining global action budget. The persisted control state contains case ID,
image SHA-256, action count, stop reason, and finish rationale.

## Interpretation boundary

Search titles, snippets, URLs, and reverse-image candidates are leads until a
returned page or comparison addresses the question. Tool errors and access
failures are limitations. A successful directed no-match is an observation
about that query, not proof of `fake`. The policy and final Judgment must
interpret each raw result within its returned scope.

`visit` retrieves content, selects bounded passages, and keeps exact source
spans in trace artifacts. The full raw page is not injected into Agent text.
Provider-facing image schemas may omit unsupported validation-only keywords;
local tools still enforce their normalization and bounds.

## Judgment and export

Judgment continues in the same session after finish, budget exhaustion, or a
bounded protocol stop. It receives a mechanical locator for observations and
must cite only successful raw observation IDs. It cannot call another tool or
invent an unrecorded fact.

The exporter writes one Qwen-compatible row per complete episode:

```text
system
user: task + image
assistant: <think>...</think>
tool_call: native call
tool_response: raw public result
...
assistant: final answer
```

It reconstructs the images actually seen at each request boundary, writes
portable data URIs, and deduplicates identical bytes by SHA-256. Missing ReAct
thought is marked action-only/RL material rather than fabricated.

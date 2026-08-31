# Agent Prompt and Runtime Guide

The active policy is `unified-react-v1`. Runtime prompts are English and are
defined only in `src/orchestrator/unified_prompts.py`; the exact generated
copy is [Active Agent System Prompts](active-agent-system-prompts.md).
The Chinese files are documentation translations, not runtime inputs.

## 1. One unified ReAct loop

Each policy turn is:

```text
thought -> exactly one native tool call -> tool observation/state delta
```

The model chooses the order of `perceive_scene` and `ocr_with_position`. No
external investigation tool is exposed until both bootstrap observations are
complete. The first investigation action carries `investigation_intent`: a
positive image-grounded `target_fact` and the route information needed by that
action. Runtime creates the canonical route/task objects and IDs.

Changing a query, candidate page, or visual direction is another ReAct action.
There is no standalone Planning, Query Replan, or Route Replan request. If
`route_local_replan` is exposed, it is a normal runtime control tool inside the
same loop.

## 2. Sparse checkpoints

- `unified_reflection` summarizes global gaps and strategy; it does not choose
  the next tool.
- `unified_discrepancy_decision` interprets recorded Evidence, Findings, and
  image anchors; it does not create a new investigation route.
- `unified_judgment` writes the reader-facing report from the runtime-compiled
  verdict basis.

## 3. State and context

The runtime owns state, IDs, budgets, deduplication, Evidence writes, and
termination. Each turn receives a compact workspace projection, recent
observations, active routes, open gaps, and state deltas. The visual workspace
also carries bounded entity attributes, visible relations, scene details, and
pixel-level uncertainties. The complete request/response/state archive remains
available for audit, but is not copied in full into every policy turn.

The current default output cap is 8,192 tokens for ReAct, Reflection,
Discrepancy Decision, and Judgment. It is a completion cap, not a tool timeout.

## 4. Image API boundaries

- `direct_multimodal`: every request of unified ReAct, Reflection, Discrepancy
  Decision, and Judgment receives one temporary controlled image attachment.
  The image is compressed with `IFV_IMAGE_MAX_LONG_EDGE` (default `1280`) and
  `IFV_IMAGE_JPEG_QUALITY` (default `88`). It is not appended to text history
  or persisted as base64; the runtime ledger externalizes it as media.
- `separate_vlm`: visual tools/VLM receive the image; the policy receives
  structured observations and state deltas.

Both modes share the same dynamic tool schema, reducer, and trace contract.
Tool-internal prompts and visual tool contracts remain with the mature tool
implementations.

In `direct_multimodal`, the structured perception report is a compact index,
not a replacement for the original pixels. The final Judgment remains a
binary synthesis step: a non-empty runtime verdict is reproduced, while an
empty one is resolved from the available target, Evidence, visual observations,
image, and explicit uncertainties.

## 5. Trace files

The canonical trace is one JSON object per completed episode in
`traces/*.json`. A Qwen SFT export is a derived file:
`trajectory_sft.jsonl` contains one JSON line per complete episode, with the
messages and compacted training context. `selection-manifest.jsonl` and
`manifest.json` describe the export; they are not trajectories.

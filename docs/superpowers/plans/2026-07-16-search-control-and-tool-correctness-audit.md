# Search Control Simplification and Tool Correctness Audit

**Date:** 2026-07-16

**Status:** Audit pass complete; implementation pending

**Repository:** `D:\image-factual-verifier-v2-worktrees\visual-fact-search-agent`

**Branch:** `codex/image-factual-verifier-v3`

**Data repository:** read-only; do not modify

## 1. Why this plan exists

Phase J solved several real retrieval defects but also allowed the investigation target
to expand during search:

```text
visible scene relation
  -> source attribution
  -> creator/title/date/platform
  -> composite or AI mechanism
  -> new decisive facts
```

This makes optional metadata block the verdict, resets low-gain stopping through weak
new records, and can turn a directly decidable case into `unverifiable`.

The correction is architectural. It must not be implemented as another prompt patch.

## 2. Target runtime model

The runtime has one stable `CoreVerdictFact` and a bounded set of evidence obligations:

```text
CoreVerdictFact
  -> image/source binding gap
  -> direct support/refute gap
  -> conflict-resolution gap, only when a real conflict exists
```

Gemini or Qwen may interpret sources, compare images, extract atomic statements, and
propose the next route. Deterministic runtime code owns:

- the identity of the core verdict fact;
- which evidence gaps are currently open;
- whether a proposed action can affect one of those gaps;
- whether new evidence is qualified progress;
- whether the investigation continues or stops;
- the final verdict and minimal basis.

Optional title, creator, date, platform, asset ID, software, and second-source metadata
remain supporting records. They do not become verdict obligations merely because they
were discovered.

## 3. Decisive-fact ownership

Exactly one runtime component may mutate verdict ownership:

```text
reconcile_core_verdict_fact(...)
```

Target Planning, Attribution, Reflection, and tools may submit proposals but cannot
directly edit `decisive_fact_ids`.

A replacement is allowed at most once and only when all conditions hold:

1. the proposed fact is atomic;
2. it stays inside the semantic scope of the current fact;
3. it is tied to the same salient image subject;
4. it has already reached `supported` or `refuted`;
5. it is more direct than the current broad fact;
6. replacing the fact removes ambiguity rather than adding metadata obligations.

An unresolved web-discovered attribution, fabrication mechanism, or compound metadata
bundle can never own the verdict.

## 4. Search admission and progress

Every tool action must own one open evidence gap. A proposed action is rejected when
its successful result could only add optional context.

Qualified progress is limited to:

- changing the adjudicated status of the core verdict fact;
- creating a new qualified direct support/refute edge for that fact;
- establishing the input-image/source binding needed by that edge;
- resolving a material support/refute conflict;
- converting a previously inaccessible decisive route into an executable one.

The following do not reset low-gain stopping:

- another SERP result or unvisited candidate;
- another weak, indirect, duplicated, or off-target source;
- a new supporting fact or metadata field;
- a new Evidence ID with no stronger binding or directness;
- task creation, recommendation, or priority churn;
- author/title/date/platform/asset-ID completion by itself.

## 5. Stopping contract

Coverage is deterministic over structured state, while semantic extraction may use an
LLM:

```text
qualified refutation of CoreVerdictFact
  -> fake

qualified support of CoreVerdictFact with required image/source binding
and no unresolved material conflict
  -> real

no executable route for an open decisive gap
  -> unverifiable

two consecutive decision checkpoints without qualified progress
  -> unverifiable

hard action/time/provider budget exhausted
  -> incomplete engineering-bounded result
```

Coverage is checked after every accepted evidence-producing action. Reflection selects
routes but cannot delay an already determined verdict.

No second independent website is required when one qualified source provides both:

```text
same_capture or near_duplicate binding
+ direct source_assertion for the atomic fact
```

## 6. Structural simplification tasks

- [ ] Introduce an explicit `CoreVerdictFact`/evidence-gap state.
- [ ] Centralize every decisive-set mutation.
- [ ] Remove decisive promotion from Reflection.
- [ ] Make Attribution supporting by default.
- [ ] Remove Target Refresh as a mechanism for changing the factual question.
- [ ] Permit one resolved, scope-preserving core-fact refinement.
- [ ] Replace evidence-ID gain with qualified decision gain.
- [ ] Run Coverage after every accepted evidence update.
- [ ] Stop immediately when the core fact is adjudicated.
- [ ] Preserve optional attribution metadata in traces and process metrics.
- [ ] Keep Judgment constrained to the compiled verdict and basis.
- [ ] Add migration tests for old traces without preserving old runtime behavior.

## 7. Tool audit method

Tests alone are not acceptance. Each active tool is audited in this order:

1. **Schema:** arguments, result contract, error semantics, source policy.
2. **Implementation:** provider calls, hidden work, retries, caching, redirects,
   truncation, extraction, provenance, and security boundaries.
3. **Reducer compatibility:** whether the tool output becomes Discovery, Evidence,
   Finding, or Failure correctly.
4. **External comparison:** official provider documentation and mature open-source
   search/fact-check agents with an equivalent tool.
5. **Real probe:** one normal case, one empty/blocked case, and one adversarial or
   ambiguous case where practical.
6. **Trace inspection:** model-visible output, latency, LLM-call count, provenance,
   and whether the result can incorrectly keep the search loop alive.

Audit ratings:

```text
correct
correct with bounded limitations
needs repair
unsafe for verdict use
remove or merge
```

## 8. Audit order

High-risk external evidence chain:

- [x] `text_search`
- [x] `visit`
- [x] `reverse_image_search`
- [x] `compare_with_reference`
- [x] `crop_and_search`

Image understanding:

- [x] `perceive_scene`
- [x] `ocr_with_position`
- [x] `crop_and_inspect`
- [x] `count_objects`

Integrity and utility:

- [x] `check_consistency`
- [x] `analyze_visual_anomalies`
- [x] `current_time`

Shared integrations are audited with their first consuming tool:

- [x] Serper text/image/Lens clients
- [x] Jina/direct page reader and goal-conditioned extractor
- [x] visual-search upload and semantic fallback
- [x] reference-image download/page extraction fallback
- [x] Gemini/Qwen VLM adapters
- [x] source identity, policy filtering, and tool-result serialization

## 9. External comparison set

Use primary sources and repository code where available:

- DEFAME for fixed-claim planning, source deduplication, per-source relevance
  filtering, OCR, reverse/image search, geolocation, and manipulation checks;
- Search-R1/R1-Searcher style search environments for explicit search/answer actions
  and hard interaction boundaries;
- Tongyi DeepResearch/AgentFold and LangGraph Open Deep Research for search-result
  inspection, research-complete actions, and iteration limits;
- official Serper, Jina Reader, Gemini, EasyOCR, and model/tool documentation;
- visual forensics implementations used by DEFAME, including TruFor, for anomaly
  output semantics.

Do not copy a mechanism solely because another agent uses it. Record the task
assumptions under which it is valid.

## 10. Audit artifacts

Maintain:

```text
docs/tool-audit/2026-07-16-tool-correctness-audit.md
```

For each tool record:

- current contract and actual behavior;
- hidden provider/LLM calls;
- correctness failures and misleading success states;
- comparison findings;
- proposed disposition;
- focused tests and real probes required;
- whether it affects classification, efficiency, or only diagnostics.

## 11. Implementation sequence after audit

Do not patch tools while their shared assumptions are still under review.

1. [x] Finish the external evidence-chain audit.
2. [ ] Freeze shared Discovery/Evidence tool contracts.
3. [ ] Implement the simplified core-fact and stopping controller.
4. [ ] Repair evidence-chain tools in dependency order.
5. [ ] Repair local image tools.
6. [ ] Run focused deterministic tests.
7. [ ] Run the full local suite.
8. [ ] Commit and fast-forward gpu-13.
9. [ ] Run fixed real probes for every repaired tool.
10. [ ] Rerun the same supported/refuted v4 canaries.
11. [ ] Manually inspect traces before any 20-case run.

## 12. Acceptance

The work is complete only when:

- a search-discovered metadata bundle cannot expand verdict scope;
- an already adjudicated core fact stops immediately;
- low-value evidence cannot reset saturation;
- every tool has a written audit and a real or reproducible probe;
- tools expose no undocumented hidden search, visit, or LLM fan-out;
- blocked/empty/malformed results have unambiguous error semantics;
- Discovery never becomes verdict Evidence;
- same-source image binding plus direct assertion can close an atomic fact;
- supported and refuted v4 canaries classify correctly with materially shorter,
  manually reasonable traces;
- the full 20-case run starts only after the fixed canaries pass.

## 13. Audit pass result

Detailed results are frozen in:

```text
docs/tool-audit/2026-07-16-tool-correctness-audit.md
```

Confirmed implementation defects:

```text
Qwen structured-vision signature incompatibility
tool-health false availability
top-level error payloads can still create Discovery
crop_and_inspect accepts invalid/reversed bbox coordinates
count_objects has no reducer path
model-visible and reducer-visible tool payloads differ
```

Primary structural findings:

```text
one action is not one bounded external operation
crop_and_search is a hidden compound sub-agent
visit searches only a page prefix and has no automatic fetch fallback
same_capture is currently one uncalibrated VLM judgment
general VLM anomaly tools are not equivalent to forensic detectors
optional metadata can still expand verdict scope
```

External repositories were inspected at fixed commits:

```text
DEFAME        0d5c2eb5e07cfa8a673351e765c9c576070cdd6c
Search-R1     598e61bd1d36895726d28a8d06b3a15bed19f5d3
Open Research b764481fca7f0dbf00b2c70239bd97cea59d1059
STORM         fb951af7744dab086e34962e9bc6fe878e145f83
```

The comparison supports:

- one stable factual target;
- explicit search and completion actions;
- bounded search top-k and query fan-out;
- accounting by actual query/subcall count;
- separating retrieval candidates from verdict Evidence;
- specialized forensic signals for manipulation claims.

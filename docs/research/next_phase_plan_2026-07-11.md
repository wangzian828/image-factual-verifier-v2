# Next Phase Plan: Evidence-Conditioned Active Visual Investigation

Date: 2026-07-11

## Implementation Status

The runtime items in P0 and P1 are implemented on the active agent branch:

- `VerificationCase` freezes claim mode, image SHA-256, claim surface/region, and
  decision-policy version without exposing benchmark gold.
- Claim, source, evidence, discovery, and failure records have stable IDs and are
  rebuilt from immutable successful/failed function calls.
- Web evidence is selected by passage ID, then restored by code as an exact span
  with artifact hash and retrieval time. Jina metadata, image markup, and dense
  navigation blocks are excluded from the evidence document before passage selection.
- `InvestigationReducer` records observation assessments, belief deltas, source-family
  novelty, visual questions, regional observations, and stopping assessments after
  every native tool result.
- Search-driven visual questions are linked to one discovery/evidence ID and can only
  be resolved by a matching real OCR/crop/count/reference call.
- Coverage closes decisive claim slots rather than counting questions; Ledger Judgment
  accepts only exact claim/evidence IDs and emits deterministic typed insufficiency.
- `test_full_native_agent_trace.py` exercises the complete controlled reference path:
  Planning, two native ReAct iterations, premature-output rejection, two audits,
  Replanning, Interaction/call IDs, ledger Judgment, and ReInspect resolution.

The remaining work in this document is experimental and data-oriented: run the
four-system baseline, build and audit Investigation World v0, then decide whether
action-policy learning is justified. It is not a reason to add another runtime agent
or weaken the current provenance and failure gates.

## Decision

Keep the current production control flow:

```text
Perception
-> Planning
-> native ReAct Verification
-> deterministic Coverage Audit / Replanning
-> Judgment
```

Do not add more top-level agents or replace it with a fixed tool sequence. The next
phase should strengthen the state inside Verification so that search results can
create a targeted visual question, trigger a real crop/OCR/comparison observation,
and produce an auditable hypothesis update.

## What The Current Refactor Establishes

- Gemini Interactions is the only active Gemini protocol.
- Native function calls and results remain linked by interaction IDs.
- Planning and replanning are model-driven and schema constrained.
- Coverage is audited between bounded verification iterations.
- Tool failures are explicit and cannot become evidence.
- Evidence and visual anomalies must be bound to successful recorded tool calls.
- Priority coverage gates Judgment.
- Face and biometric functionality is absent.
- JSON and HTML traces expose the multi-stage trajectory.

These are runtime invariants. Future training or prompting must not weaken them.

## P0: Freeze The Case And Claim Contract

Define one versioned `VerificationCase` contract before creating a larger benchmark:

```yaml
case_id: string
image_path: string
image_sha256: string
claim_mode: external_claim | embedded_claim
user_claim: string | null
claim_surface: string | null
claim_source_region: bbox | null
decision_policy_version: string
```

- `external_claim`: the claim is part of the runtime input.
- `embedded_claim`: the Agent must recover the claim from visible pixels/OCR; benchmark
  `primary_claim` remains hidden gold and must not leak into runtime context.
- Replace the ambiguous notion of image intent with atomic, decision-relevant claims.

## P0: Introduce Machine-Verifiable Ledgers

Add stable IDs and provenance for:

1. Claim ledger: atomic claim, criticality, status, unresolved distinction.
2. Evidence ledger: tool-call ID, exact page span or image region, artifact hash,
   stance, quality, and retrieval time.
3. Source ledger: canonical URL/domain, source family, primary/secondary/UGC class,
   and dependency edges.
4. Failure ledger: tool-call ID, typed failure, criticality, and recovery link.

Search snippets, reverse-image-search titles, and generated summaries are discovery
candidates, not verdict evidence. Web evidence only becomes eligible after a page visit
records the exact span, URL, content hash, retrieval time, and source family.

## P1: Incremental Belief And Active Reinspection

Update state after every observation rather than producing only an end-of-iteration
world summary:

```text
old hypotheses
-> selected investigation action
-> real tool observation
-> evidence validation
-> support / refute / split / create / unknown update
-> new hypotheses and remaining distinctions
```

Add search-conditioned visual actions:

- positioned OCR on a selected region;
- crop and inspect for a named hypothesis distinction;
- crop reverse-image search;
- reference comparison tied to a candidate source;
- explicit target region and expected discriminative property on every visual revisit.

The decisive constraint is that a web/RIS observation cannot directly change visual
belief. It first creates a visual question, then a real visual tool observation resolves
or preserves that question.

## P1: Claim-Slot Coverage And Stopping

Replace question-count coverage with decisive claim/evidence slots. A slot closes only
when validated evidence directly supports or refutes its atomic claim. Track source
independence, conflicts, tool failures, repeated observations, and the expected value of
another action.

Use typed `unverifiable` reasons such as:

- decisive evidence absent;
- sources conflict;
- evidence depends on one upstream source family;
- relevant visual region is unreadable;
- selected tool/provider cannot access the needed evidence;
- budget exhausted without sufficient coverage.

Engineering failures remain errors and must not be relabeled as `unverifiable`.

## P1: Baseline Experiment

Under the same tool and token budget, compare:

1. VLM only.
2. Fixed OCR + RIS + web pipeline.
3. Current question-level ReAct.
4. Ledger + incremental belief + active visual revisit.

Primary metrics:

- decisive-region hit rate;
- valid evidence recall and citation entailment precision;
- supported-verdict accuracy and selective accuracy;
- independent source-family coverage;
- premature stop, over-search, and silent false-success rates;
- latency and cost per supported verdict.

## Data Work In Parallel

Treat the existing 34 examples as an engineering smoke set. Build Investigation World v0
in parallel with the ledger implementation, starting from world facts and evidence
topology before generating images.

Pilot with 10-20 worlds, then expand only after schema and leakage audits pass. A mature
version can target 100-200 worlds with supported, refuted, and insufficient evidence
variants; multiple visual assets; frozen documents/search; source-family structure; and
real counterfactual action branches.

Required audits include image-only, OCR-only, metadata-only, no-tool, random-search,
oracle-evidence, gold-action, source-shuffled, and grouped leakage tests.

## Deferred Work

Do not start these until the ledger baseline and Investigation World pilot are stable:

- multi-agent architecture;
- graph database infrastructure;
- an independent LLM auditor as a correctness guarantee;
- end-to-end RL;
- live-web training rewards;
- conformal calibration.

If action selection remains the bottleneck, compare SFT, pairwise ranking, DPO, and a
contextual bandit before long-horizon RL. RL may optimize `Belief State -> Next Action`;
it must never learn or override provenance, failure propagation, or judgment gates.

## Paper Boundary

- Paper 1: task definition, real/frozen benchmark, process metrics, construction engine,
  and a non-RL ReInspect reference Agent.
- Paper 2: Internet-grounded synthetic training worlds and learned active-investigation
  policy using Paper 1's task and metrics.

This separation keeps the current runtime useful as the Paper 1 reference system while
leaving large-scale action learning as a distinct research contribution.

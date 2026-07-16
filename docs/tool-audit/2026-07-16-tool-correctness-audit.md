# Tool Correctness Audit

**Date:** 2026-07-16

**Runtime:** VisualFact Search Agent v3

**Plan:** `docs/superpowers/plans/2026-07-16-search-control-and-tool-correctness-audit.md`

**Status:** Local audit and repairs complete; gpu-13 real-canary validation pending

## 1. Audit rules

Each rating considers the public schema, provider implementation, model-visible
result, state reducer, source policy, hidden calls, real failure semantics, and whether
the tool can incorrectly affect verdict or stopping.

Tests are supporting evidence only. A passing test means the implemented contract is
stable; it does not establish that the contract is the right one.

## 2. Cross-tool findings

### 2.1 One action does not currently mean one bounded external operation

The runtime counts one accepted tool call as one action, but several tools contain
unbounded or multi-provider fan-out:

- `text_search`: one request per supplied query;
- `visit`: one page fetch and one evidence-extraction LLM call per supplied URL;
- `reverse_image_search`: upload + Lens + image-to-query VLM + semantic image search;
- `crop_and_search`: repeated crop analysis, upload/search, page visits, and extraction.

Action count therefore cannot currently serve as a cost, latency, or information-gain
unit. Every tool needs a hard internal fan-out bound and explicit subcall accounting.

### 2.2 Model-visible output and reducer-visible output can differ

`StageRunner` shows at most two search queries and three visits to the policy, while
the reducer processes all results returned by the tool. A single action may therefore
change state using discoveries or evidence that the policy never saw.

The same bounded normalized result must feed both model context and state reduction.

### 2.3 Partial-success semantics are inconsistent

Several tools return `status=error` while retaining useful rows from another branch.
The reducer may still create Discovery from those rows, while the action is also
recorded as a Failure. Partial success needs an explicit contract instead of combining
error status with usable payloads.

Proposed statuses:

```text
success
partial
error
```

If the global tool contract remains binary, a tool must either return a bounded
successful subset or fail without state-changing rows.

## 3. `text_search`

**Rating:** correct with bounded limitations; needs a schema repair before relying on
action-level efficiency metrics.

### Pre-repair behavior

- Uses Serper Search only.
- Produces titles, URLs, snippets, provider aggregates, locale, and timing.
- Performs no hidden page visit and no evidence-extraction LLM call.
- Source-policy filtering removes blocked result rows and provider aggregates.
- Search output is reduced to Discovery only.
- Empty SERP results are successful empty output and later become a recorded
  `no_results` failure for the owning task.

### Correct aspects retained

- Discovery/Evidence separation is correct.
- Provider exceptions become explicit tool errors.
- Sync and async paths share the same result contract.
- Exact and semantic duplicate route checks exist outside the tool.
- Existing focused validation passed:

  ```text
  text search / browse / source policy / failure contracts: 54 passed
  ```

### Problems found

1. `queries` has no `maxItems`, item schema, length limit, or non-empty-string
   validation. One action can launch arbitrarily many Serper requests.
2. The async path launches every query concurrently without an internal concurrency
   bound.
3. `StageRunner` exposes only the first two query groups, while the reducer records all
   result groups.
4. If one query fails and another succeeds, the whole tool reports `error` while
   retaining successful results. Those rows may still become Discovery.
5. Default locale is inferred as China for any CJK query and United States otherwise.
   Language is not equivalent to evidence geography; this can bias retrieval without
   an explicit task requirement.
6. Result deduplication is deferred to later URL/route logic. The tool can return
   repeated canonical URLs across queries.
7. Provider answer-box and knowledge-graph values are returned when no source-access
   policy is active even though they lack a stable evidence URL. They are not used by
   the reducer, but add model-visible ungrounded aggregates unless compaction removes
   them.

### Implemented disposition

- Limit one action to one query, or at most two explicitly schema-bounded queries.
- Canonicalize and deduplicate returned URLs before both model display and reduction.
- Remove answer-box and knowledge-graph output unconditionally from the runtime tool.
- Make locale an explicit optional route decision; do not infer geography solely from
  script.
- Define partial success or make a multi-query action atomically successful only when
  all bounded queries succeed.

### External comparison

DEFAME also treats web search as source retrieval rather than evidence and removes
already known source URLs. It uses an exact action dedupe and a small result limit.
The v3 semantic route dedupe is stronger, while DEFAME's source-level dedupe should be
copied into the normalized output boundary.

## 4. `visit`

**Rating:** needs repair. Its evidence contract is strong, but its retrieval,
fan-out, and document-selection behavior can miss decisive content or distort the
action budget.

### Current behavior

- Fetches through either Jina Reader or direct HTTP, selected globally.
- Cleans a page into a paragraph document.
- Truncates the cleaned document.
- Runs one goal-conditioned LLM passage selector per URL.
- Recovers selected text and offsets from the cleaned document.
- Emits stance, directness, temporal alignment, source URL, timestamp, document hash,
  injection flags, and evidence eligibility.
- A multi-URL visit runs URLs concurrently and selects one best visit.

### Correct aspects

- The LLM selects a passage ID; runtime code recovers the exact passage instead of
  trusting generated evidence text.
- Unknown passage IDs and invalid enum values fail closed.
- Missing mention is explicitly not treated as refutation.
- All-blocked and all-failed multi-visits become tool errors.
- Source-policy checks apply before fetching and across redirects.
- Web evidence is created only from eligible exact-span records.
- Prompt-injection flags prevent affected records from entering verdict evidence.

### Problems

1. The URL array has no hard maximum. One action can trigger arbitrary page fetches
   and one LLM extraction per page.
2. The model sees at most three visit records while the reducer sees all records.
3. Default `max_chars=12000` means only the first 12,000 characters of the cleaned
   page can be selected, despite `extract_max_chars=60000`. Relevant captions,
   tables, or later sections can be silently unreachable.
4. Passage selection is not retrieval over the whole page; it is selection over a
   prefix. This explains failures where the page contains the fact outside the prefix
   or in metadata removed by cleaning.
5. Fetch mode is exclusive. `fetch_provider=jina` does not automatically fall back to
   direct fetch, and direct fetch does not fall back to Jina.
6. Blocked-page and prompt-injection detection happen after the content has already
   been sent to the extraction LLM. Evidence is later rejected, but cost and exposure
   already occurred.
7. `_pick_best_visit` prioritizes relevance, temporal alignment, and passage length,
   but not directness, stance usefulness, source class, or exact core-gap ownership.
8. The stored `artifact_sha256` hashes the clipped cleaned evidence document, not the
   raw fetched artifact. The field name is stronger than its actual semantics.
9. The direct HTML-to-text conversion is regex-based and cannot reliably handle
   rendered JavaScript, structured data, PDFs, complex tables, or captions carried
   only in attributes.
10. Cache identity includes exact goal text. Semantically equivalent goals repeat the
    extraction LLM call.

### Required disposition

- Restrict a visit action to one URL. Let scheduling choose another candidate in a
  later action.
- Detect blocked/security/injection patterns before LLM extraction.
- Implement a deterministic fetch fallback chain with recorded attempts.
- Build chunks across the full bounded page, retrieve candidate chunks against the
  fixed core fact/gap, then let the LLM select from those candidates.
- Preserve and name both raw-fetch hash and normalized-document hash.
- Rank candidates by directness and gap relevance before passage length.
- Ensure the identical normalized visit payload is shown to the model and reducer.

### External comparison

DEFAME fetches several SERP pages inside search and applies an LLM source-relevance
filter that returns `NONE` for irrelevant pages. Its fixed Claim keeps that filter
stable, but its hidden fan-out is expensive. V3 is correct to expose `visit` as a
separate action; it should keep that separation and make one visit equal one page.

## 5. `reverse_image_search`

**Rating:** needs repair for accounting, lifecycle, and result semantics; useful as
Discovery only.

### Current behavior

One tool action executes two branches:

```text
local image
  -> upload to selected host
  -> visual reverse search / Lens

local image
  -> VLM-generated semantic query
  -> Serper image search
```

Both branches return candidate pages and candidate image URLs as Discovery.

### Correct aspects

- Visual matches and semantic matches are explicitly separated.
- Results remain Discovery until a later page visit or image comparison.
- Provider failures are surfaced instead of silently switching providers.
- Candidate page and image lists are deduplicated.
- Source-policy filtering applies before candidates enter output.
- Existing focused reverse/reference tests passed:

  ```text
  reverse/reference focused tests: 15 passed
  ```

### Problems

1. A single action contains upload, visual provider search, VLM inference, and semantic
   image search. Latency, cost, and progress cannot be attributed to one route.
2. The tool name implies reverse search, while half of the output is semantic image
   search generated by a VLM. These have different evidentiary meaning.
3. A visual-branch failure causes global `status=error` even when semantic results are
   usable; the reducer can still record the partial rows as Discovery.
4. `_is_image_url` accepts only known extensions or a short host allowlist. Many CDN,
   signed, resizing, and API image URLs have no image extension and are discarded.
5. Uploaded OSS objects are not deleted. Signed URL expiry limits access but does not
   delete the stored input image.
6. The upload metadata does not include a cleanup deadline/object identifier usable
   by a cleanup process.
7. A VLM-generated semantic query can introduce a mistaken identity. It is Discovery
   only, but can redirect the entire search unless the fixed core-fact/gap controller
   rejects off-scope routes.
8. Candidate quality is not calibrated. Serper Lens rows, Zhipu rows, and semantic
   image rows share one downstream Discovery shape despite different match semantics.

### Required disposition

- Split visual reverse search and semantic image search into separately accounted
  subroutes, even if retained under one high-level policy action.
- Define explicit partial-success semantics.
- Validate candidate image URLs by bounded HEAD/GET content type during comparison,
  not by URL extension during discovery.
- Add upload deletion or lifecycle cleanup and record it in the trace.
- Preserve provider-specific match type and confidence instead of flattening rows.
- Prevent the semantic-query branch from creating or replacing the core verdict fact.

### External comparison

DEFAME uses Google Cloud Vision Web Detection for web entities, best-guess labels, and
pages containing matching images. It also keeps reverse-search output as retrieval
results. DEFAME benefits from provider-native matching categories but does not perform
the later explicit same-capture comparison that v3 has. V3 should retain its two-step
`candidate discovery -> explicit comparison` design.

## 6. `compare_with_reference`

**Rating:** correct with a serious verdict-risk limitation. The transport and output
validation are strong; same-capture truth currently rests on one VLM judgment.

### Current behavior

- Downloads one reference image with URL variants, redirects, referer, and source-page
  image extraction fallbacks.
- Sends the reference and current image to Gemini Interactions.
- Produces same-subject, same-capture, different-capture, visible-edit, difference,
  confidence, and observation fields.
- Runtime validation enforces schema and logical consistency.

### Correct aspects

- It clearly separates same subject from same capture.
- Different original capture cannot simultaneously be marked same capture.
- Edit evidence must be grounded in explicit addition/removal/modification items.
- Crop, compression, watermark, color, and lighting differences are not automatically
  treated as manipulation.
- Unrelated images are valid neutral comparison results rather than tool failures.
- Download redirects and page fallbacks respect source policy.
- The exact provider interaction and token metrics are recorded.

### Problems

1. `same_capture_or_near_duplicate` is a VLM boolean with no deterministic perceptual
   similarity, feature matching, or second-pass consistency check.
2. Downstream code can treat this boolean as high-strength visual binding even when
   confidence is low. The comparison confidence is stored but not used as an evidence
   admission threshold.
3. “Near duplicate” combines benign transformations and potentially materially edited
   copies. A result may be both near-duplicate and contain strong factual edits, which
   needs separate downstream semantics.
4. Page fallback extracts only `og:image`, `twitter:image`, and `img src`; it misses
   `srcset`, lazy-load attributes, JSON-LD, CSS backgrounds, and dynamic galleries.
5. SVG is accepted by download headers but MIME sniffing does not recognize it when
   content type is missing.
6. Download catches broad exceptions and returns an empty attempt result, which can
   hide the exact terminal error class.
7. No image dimensions, decode validation, perceptual hash, or EXIF information are
   recorded before the VLM call.

### Required disposition

- Add deterministic preflight: decode both images, dimensions, MIME, hashes, and a
  perceptual/feature similarity signal.
- Use deterministic similarity to detect obvious exact/near duplicates and obvious
  non-matches; reserve VLM comparison for ambiguous cases and semantic edits.
- Require a calibrated confidence/similarity condition before `same_capture` becomes
  decisive binding.
- Keep edit evidence independent from same-capture binding.
- Preserve exact download failure reasons and expand page image extraction.

### External comparison

DEFAME's reverse search trusts provider-native “pages with matching images” and does
not provide an equivalent explicit pairwise same-capture adjudicator. V3's comparison
stage is therefore a useful addition, but it needs a non-LLM visual check before its
output can safely own verdict evidence.

## 7. `crop_and_search`

**Rating:** remove or split. The current implementation is a hidden sub-agent rather
than one bounded tool action.

### Current behavior

One call performs:

```text
crop and persist region
  -> upload crop
  -> visual reverse search
  -> VLM-generated crop query
  -> semantic image search
  -> collect candidate pages/images
  -> visit up to three pages
  -> one evidence-extraction LLM call per visited page
  -> select one best page and emit Evidence
```

The schema currently accepts one bounding box, despite descriptions and internal
structures referring to multiple regions.

### Correct aspects

- Bounding boxes are normalized and small crops are expanded.
- Temporary crop files are removed after execution.
- Saved diagnostic crops are separated from temporary provider input.
- Search findings remain explicit in nested result records.
- Source-policy filtering is applied to search rows and page visits.
- A selected page can emit exact-span web evidence through the shared visit contract.

### Problems

1. One action can contain two searches, one upload, one VLM query, three page fetches,
   and three evidence-extraction LLM calls.
2. It recreates the old hidden `text_search -> auto-visit -> LLM extraction` behavior
   that was previously removed for cost and trajectory-quality reasons.
3. Search Discovery and final Evidence are produced by the same tool action, making it
   impossible for the policy to choose or inspect the candidate before evidence enters
   state.
4. The model-visible compact result omits most Lens/semantic rows, visits, provenance,
   and candidate images, while the reducer processes the complete nested payload.
5. There are no focused tool tests for crop search behavior, fan-out, provider
   failures, source-policy propagation, or reducer semantics.
6. A visual search provider error can be carried in `visual_search_errors` while the
   region still succeeds through semantic search and page extraction.
7. Candidate image URL filtering repeats the extension/host heuristic already found
   unsafe in `reverse_image_search`.
8. `_pick_best_region` ranks by LLM relevance, evidence length, and the presence of
   Lens results; it does not require directness, source binding, or ownership of a
   decisive evidence gap.
9. Persisted crops use random filenames and have no retention or cleanup policy.
10. The low-value-query filter is an English lexical heuristic. It is not multilingual
    and does not determine whether a query can affect the core verdict fact.

### Implemented disposition

The active Agent no longer exposes this compound tool. A future reintroduction must
use explicit actions:

```text
crop_image
  -> reverse_image_search(crop artifact)
  -> visit(selected candidate)
  -> compare_with_reference(selected image)
```

The retained standalone implementation is diagnostic only. It cannot appear in the
Agent registry or produce runtime verdict Evidence.

### External comparison

DEFAME keeps reverse search and source summarization inside its search stack, but its
fixed Claim and three-round outer loop bound the expansion. V3 has explicit actions
and canonical traces, so preserving a hidden multi-stage crop sub-agent provides no
benefit and defeats its stronger audit model.

## 8. `perceive_scene`

**Rating:** correct with bounded limitations as bootstrap perception; unsafe as
identity evidence.

### Current behavior

- One VLM call inventories up to eight entities, scene type, scene description, model
  confidence, and optional boxes.
- Runtime accepts normalized XYXY boxes or converts Gemini-native 0..1000 YXYX boxes.
- Perception produces initial image-grounded facts and retrieval anchors.

### Correct aspects

- Output is tightly schema-bounded.
- Non-pixel biography/history reasoning is explicitly excluded.
- OCR is separated from visual perception.
- Invalid or unordered boxes fail closed.
- Raw model attributes are discarded.

### Problems

1. Entity names may contain specific real-world identities even though the pixels only
   support an appearance-level description. The earlier unexpected person recognition
   demonstrated this behavior.
2. Model-reported confidence is not calibrated and is later reused to rank retrieval
   anchors.
3. A single scene sentence becomes a broad `appears_to_depict` fact, which can combine
   several entities, places, and relations into one non-atomic proposition.
4. The 0..1000 coordinate conversion assumes Gemini-native YXYX whenever any value is
   greater than one. Other providers or a model emitting absolute pixels can be
   silently misinterpreted.
5. No image dimensions or provider coordinate convention accompany the model output.

### Required disposition

- Treat names as appearance hypotheses unless grounded by visible text or later
  source/image binding.
- Do not use VLM confidence as evidence strength.
- Build the core verdict candidate from atomic visible relations rather than copying
  the full scene sentence.
- Make coordinate convention explicit in the provider adapter.
- Keep perception outside verdict Evidence; it defines what may be investigated.

### External comparison

DEFAME separates object detection into a dedicated DETR model with explicit boxes and
uses the LLM planner afterward. Its detector is less semantically expressive but does
not invent named identities. V3 should keep VLM scene understanding but apply the same
separation between visible class/geometry and external identity.

## 9. `ocr_with_position`

**Rating:** correct with performance and evidence-linking repairs needed.

### Current behavior

- Runs EasyOCR with simplified Chinese and English.
- Supports full-image or regional OCR.
- Returns normalized quadrilaterals, axis-aligned boxes, confidence, text, and a
  heuristic language label.
- Bootstrap converts OCR regions above confidence 0.5 into visible-text facts and
  retrieval anchors.

### Correct aspects

- OCR is local and does not rely on an LLM.
- Empty output is a valid success.
- Regional coordinates are mapped back to the full image.
- Low-confidence bootstrap strings are filtered.
- OCR text remains a pixel observation, not direct proof of external-world claims.

### Problems

1. The EasyOCR reader is destroyed and garbage-collected after every call. Repeated
   regional OCR reloads model weights and can make a cheap action expensive.
2. `gpu=False` is hard-coded even on gpu-13.
3. Language detection only checks CJK characters and labels everything else English.
4. `full_text` concatenates detector order without explicit reading-order sorting.
5. Regional OCR evidence uses the whole image SHA rather than a crop artifact hash.
6. Reducer linkage attaches the returned `full_text` to all task fact IDs; it does not
   verify that the expected property or target text was actually found.
7. Bootstrap confidence threshold 0.5 is global and uncalibrated across text size,
   language, and image quality.

### Required disposition

- Cache one reader per process/device and make CPU/GPU configuration explicit.
- Keep low-confidence candidates outside canonical `full_text`.
- Store the actual crop hash and coordinates for regional OCR.
- Reject reversed or empty regional boxes.
- Allow a PP-OCR/PP-Structure-compatible service and record fallback to EasyOCR.
- Keep OCR positive matches as pixel evidence and absences as inconclusive unless the
  region/visibility obligation makes absence meaningful.

### External comparison

DEFAME also uses EasyOCR, but its checked-in implementation is incomplete and does not
provide normalized positional evidence. V3's positional contract is stronger. The
main improvements are runtime reuse and disciplined linkage, not replacing EasyOCR
because DEFAME uses it.

## 10. `crop_and_inspect`

**Rating:** correct as a visual observation helper; currently low utility for Coverage.

### Current behavior

- Crops one region and asks a VLM for description, findings, anomalies, and a direct
  answer to a focus question.
- Reducer stores its answer as neutral `pixel_observation`.

### Correct aspects

- One crop and one VLM call make the action cost understandable.
- Temporary crop cleanup is explicit.
- The output is schema-bounded and does not directly decide the final verdict.

### Problems

1. Bounding-box schema has no `minItems`, `maxItems`, range, or ordering constraints;
   validation happens only after execution starts.
2. It accepts normalized coordinates, Gemini-style 0..1000 values, or absolute pixels
   based only on magnitude, unlike its public description.
3. The reducer ignores the structured findings/anomalies and keeps only free-text
   answer/description.
4. Its observation is always neutral, so it cannot resolve even an explicit visual
   yes/no gap. It often consumes an action without changing Coverage.
5. The focus question can ask for an external identity that pixels alone cannot
   establish.
6. No confidence or uncertainty field is returned.

### Required disposition

- Restrict it to a registered visual evidence gap with an expected observable
  property.
- Normalize coordinates before the tool call using one shared bbox contract.
- Return structured `observed | not_observed | uncertain` plus literal evidence.
- Permit it to resolve only pixel-visible properties, never external attribution.
- Remove generic anomaly output; use the dedicated integrity path when needed.

## 11. `count_objects`

**Rating:** remove from the active verification registry or repair its state contract.

### Current behavior

- Uses one VLM call to count a named target in a full image or normalized crop.
- Returns count, model confidence, explanation, and text locations.

### Correct aspects

- Bounding boxes are strictly normalized and validated.
- Count is constrained to a non-negative integer.
- Invalid structured output fails closed.

### Problems

1. `count_objects` has no branch in `_visual_evidence_record`; its successful output
   creates neither Evidence nor Finding.
2. It therefore consumes an action but cannot produce qualified Coverage progress.
3. VLM-reported confidence is uncalibrated.
4. Textual locations are not machine-checkable boxes or instance IDs.
5. There is no deterministic cross-check against object detections or marked regions.
6. Object count is commonly an incidental fact and can expand the search target
   without helping real/fake adjudication.

### Required disposition

- Remove it from active ReAct until a registered count gap exists.
- If retained, return instance boxes/points and link the result to an explicit atomic
  visual count fact.
- Treat count as pixel evidence only; it must not become a general provenance or
  authenticity signal.

### External comparison

DEFAME uses a dedicated DETR object detector and returns detected classes plus boxes.
That gives a reproducible basis for counts but only over a fixed ontology. V3's open
vocabulary VLM count is useful for targeted questions, but without boxes or a reducer
path it is currently less auditable.

## 12. `check_consistency`

**Rating:** remove or merge into `analyze_visual_anomalies`.

### Current behavior

- Uses a VLM to check one of shadow, perspective, scale, lighting, edges, physics, or
  all.
- Returns a boolean `consistent`, details, and free-form inconsistency records.
- Reducer allows it to support/refute only a `visual_integrity` fact.

### Correct aspects

- Runtime correctly blocks it from adjudicating identity, location, event, habitat,
  or other external facts.
- Structured output and tool errors are bounded.

### Problems

1. It duplicates `analyze_visual_anomalies` with a weaker schema.
2. A boolean “consistent” is not evidence that an image is authentic. Many manipulated
   images are visually consistent, and many authentic images contain apparent
   inconsistencies.
3. No confidence, calibration, localization boxes, or specialized forensic signal is
   present.
4. Reducer maps `consistent=true` to support for visual integrity and
   `consistent=false` to refutation, which overstates what a generic VLM can establish.
5. No focused implementation tests exercise this tool.

### Required disposition

- Remove from the active registry, or merge its aspect focus into one anomaly tool.
- Never allow absence of visible inconsistency to support authenticity.
- At most, a clearly localized inconsistency may create a diagnostic requiring
  corroboration.

## 13. `analyze_visual_anomalies`

**Rating:** correct as a diagnostic VLM tool; unsafe as independent authenticity
Evidence.

### Current behavior

- Runs one Gemini Interactions request over the image.
- Produces localized textual anomalies, severity, type, involved entities, overall
  authenticity, confidence, and notes.
- Strict Pydantic validation rejects malformed output.

### Correct aspects

- Schema and native interaction handling are strong.
- The prompt is conservative and separates anomaly families.
- Runtime only accepts it for a dedicated visual-integrity task.
- Tool tests cover provider protocol and schema failures.

### Problems

1. It is a general VLM assessment, not a forensic detector.
2. `overall_authenticity=authentic` with no anomalies currently supports a visual
   integrity fact. Absence of VLM-detected anomalies is not positive authenticity
   evidence.
3. `likely_ai` or `likely_manipulated` plus any anomaly currently refutes integrity
   without a minimum severity, confidence, localization verification, or independent
   forensic signal.
4. Model-reported confidence and severity are not calibrated.
5. “Logical inconsistency” can confuse an implausible depicted world with pixel
   manipulation, recreating the earlier world-fact/integrity conflation.
6. Text-only regions are not returned as inspectable masks or boxes.

### Required disposition

- Keep only as route discovery/diagnostic support.
- Do not allow a clean scan to support authenticity.
- Require a strong, localized anomaly plus a second visual/forensic check before it
  may refute pixel integrity.
- Separate physical-world implausibility from image-generation/manipulation evidence.
- Consider a specialized forensic model for manipulation claims.

### External comparison

DEFAME uses TruFor and Noiseprint-derived outputs: a manipulation score, localization
map, confidence map, and noiseprint map, then uses an LLM only to summarize those
signals. This is a better evidence architecture for pixel manipulation than asking a
general VLM to generate the forensic signal itself.

## 14. `current_time`

**Rating:** correct implementation, unnecessary in the active image-only verification
registry.

### Current behavior

- Returns system date, datetime, and timezone.
- The pipeline currently excludes it from verification-stage tools.

### Findings

- The implementation is deterministic and simple.
- Image-only release cases contain no runtime claim date or as-of cutoff.
- Current time cannot adjudicate a visual fact by itself.
- Keeping it registered but unavailable adds configuration surface without current
  utility.

### Required disposition

- Remove from `STAGE_TOOLS` and optionally retain as a general utility for future
  dated-claim protocols.
- Temporal evidence should use an explicit case cutoff supplied by a future contract,
  not the server clock.

## 15. Shared VLM adapters and tool health

**Rating:** needs repair; current health reporting can produce false availability.

### Confirmed compatibility failure

`PerceiveSceneTool`, `CropAndInspectTool`, `CropAndSearchTool`,
`CountObjectsTool`, and `CheckConsistencyTool` pass `response_schema` to
`create_image_json`.

`QwenVLClient.create_image_json` does not accept `response_schema`. A direct probe
reproduced:

```text
TypeError: create_image_json() got an unexpected keyword argument 'response_schema'
```

Despite this, `build_all_tools_with_health(vlm_provider="qwen")` marks every tool
available because health registration only constructs objects and never checks
capabilities or runs a schema smoke test.

This is a P0 issue for the planned Qwen takeover.

### Additional findings

1. The Gemini adapter honors JSON Schema and checks required paths, which matches the
   official structured-output contract.
2. The Qwen adapter only parses a JSON object and does not validate a supplied schema.
3. Different VLM providers therefore do not implement one shared protocol despite
   being returned by the same factory.
4. Gemini forces a minimum of 8,192 output tokens even when a visual tool requests
   200-2,000 tokens. This inflates declared generation capacity and can increase
   latency/cost.
5. Tool health does not check credentials, selected provider configuration, upload
   configuration, structured-output support, or a minimal provider interaction.
6. `analyze_visual_anomalies` and `compare_with_reference` require a backend exposing
   `create_interaction`, while other visual tools require `create_image_json`.
   Capability requirements are implicit rather than declared.

### Required disposition

- Define one `StructuredVisionClient` protocol including `response_schema`.
- Make every adapter validate and return the same result contract.
- Add declared capabilities such as:

  ```text
  structured_image_json
  native_interactions
  remote_image_uri
  local_image_data
  ```

- Health-check required capabilities, configuration, and a cheap schema smoke test.
- Do not register a tool as available when its selected provider cannot execute its
  call signature.
- Respect per-tool output-token bounds unless a documented provider minimum is
  actually required.

## 16. Source identity and policy

**Rating:** correct for benchmark contamination control; source-quality taxonomy is
too narrow for evidence weighting.

### Correct aspects

- Canonical URLs remove common tracking fields.
- Redirected archive and WordPress proxy origins are inspected by source policy.
- Excluded fact-check domains and URLs are filtered before model-visible output.
- Cached results are re-sanitized against the active policy.
- Lookalike official domains receive a risk flag.

### Problems

1. Official and news domain lists are very small. Many museums, universities,
   scientific organizations, agencies, publishers, archives, and reputable news
   domains become `unknown`.
2. Any `.edu`, `.gov`, or `.int` host is classified as official without an ownership
   or jurisdiction distinction.
3. `source_family` becomes `content:<hash>` whenever content exists. Two mirrors of the
   same text become one family, but two page versions from the same publisher become
   different families. This is unsuitable as a complete independence measure.
4. The content hash may represent normalized/clipped text rather than the raw source
   artifact, depending on the producing tool.
5. URL canonicalization does not normalize host-specific equivalences, archive
   snapshots, AMP/mobile variants, or default index pages.

### Required disposition

- Separate publisher identity, URL identity, and content identity.
- Use source class as a descriptive feature, not a universal truth score.
- Preserve both raw and normalized content hashes.
- Keep benchmark source blocking independent from general evidence-quality policy.

## 17. Tool result, reducer, and context boundary

**Rating:** binary result validation is strong; state reduction is incomplete and can
consume information the policy did not see.

### Correct aspects

- Every tool must return an object with `status=success|error`.
- Malformed results become engineering errors rather than Evidence.
- Source-policy sanitization occurs again after tool execution and cache retrieval.
- Tool-internal LLM calls and token use are recorded when tools expose runtime metrics.

### Problems

1. Binary status cannot represent useful partial output cleanly.
2. Discovery is recorded from nested rows even when the top-level tool status is
   `error`. This is confirmed by reducer order: `_record_discoveries(...)` runs before
   the `succeeded` branch.
3. Full raw output drives the reducer, while a separately compacted subset is shown to
   the policy.
4. Tool schemas do not consistently express the validation later enforced by Python.
5. `count_objects` has no reducer path.
6. `crop_and_inspect` and regional OCR generally create neutral observations without a
   deterministic expected-property adjudicator.
7. General VLM integrity outputs can currently become support/refute Findings too
   easily.
8. An Evidence record's `quality` is inferred from source class, relevance, and
   directness, but LLM-generated relevance/directness are not independently calibrated.

### Required disposition

- Normalize one bounded tool payload before both model context and reduction.
- Add explicit partial-result handling or prohibit state-changing rows on error.
- Define a reducer contract for every active tool; remove tools without one.
- Separate observation extraction from deterministic comparison to the registered
  evidence gap.
- Record subcalls, provider attempts, and per-subcall cost in canonical traces.

## 18. Comparison with mature agents and primary implementations

The comparisons support several consistent design choices:

- DEFAME keeps one fixed Claim, removes repeated actions and known source URLs, and
  filters each source for relevance to that Claim. Its stopping and retrieval controls
  are simpler, though its search tool hides page fetch and summarization fan-out.
- Search-R1 exposes search as an explicit model action rather than mixing search with
  verdict evidence. This supports keeping SERP retrieval separate from visits.
- STORM retrieval modules expose a configurable top-k and track usage by query. This
  supports counting internal query fan-out rather than treating a query list as one
  action.
- Google Cloud Vision Web Detection distinguishes full matches, partial matches,
  visually similar images, matching pages, and web entities. V3 should preserve
  provider-native match categories rather than flattening all visual rows.
- EasyOCR supports reusable readers over configured compatible languages. Rebuilding
  the reader on every regional call is an avoidable local implementation cost.
- TruFor produces a whole-image score, localization map, and reliability/confidence
  map from forensic signals. This supports treating general-VLM anomaly judgments as
  diagnostics rather than equivalent forensic Evidence.
- Gemini structured output provides schema-constrained JSON but does not make semantic
  confidence calibrated. Runtime validation is still required after schema validation.

## 19. Repair priority

### P0: correctness and provider compatibility

1. Fix the Qwen structured-vision protocol and health false positives.
2. Centralize core verdict ownership and remove decisive mutation from Reflection.
3. Make model-visible and reducer-visible tool payloads identical.
4. Define partial-success semantics.
5. Prevent clean VLM anomaly scans from supporting authenticity.
6. Prevent unresolved metadata/attribution from owning the verdict.

### P1: action semantics and hidden fan-out

1. One `text_search` action = one bounded query.
2. One `visit` action = one page and one extraction.
3. Split or demote `crop_and_search`.
4. Account reverse visual search and semantic image search separately.
5. Add upload cleanup.
6. Replace evidence-ID gain with qualified core-gap gain.

### P2: evidence quality

1. Full-page chunk retrieval before visit passage selection.
2. Jina/direct fallback with recorded attempts.
3. Deterministic image decode/hash/similarity before VLM comparison.
4. Expected-property adjudication for regional OCR and crop inspection.
5. Remove or repair `count_objects`.
6. Add a specialized forensic path if pixel manipulation remains a released target.

### P3: efficiency and diagnostics

1. Reuse EasyOCR readers and configure device.
2. Refine source identity/family representation.
3. Reduce inflated visual output-token limits.
4. Add artifact retention and cleanup policies for crops/uploads.

## 20. Validation performed during this audit

Focused deterministic suites:

```text
text search / visit / policy / failure contracts: 54 passed
reverse search / reference comparison: 15 passed
perception / OCR / crop / count: 10 passed
integrity tools: 13 passed
shared adapters / source policy / native protocol: 82 passed
```

These results confirm existing contracts, not factual correctness.

Live provider probes were not run locally because API credentials were not loaded in
the current shell. Real probes remain required on gpu-13 after repairs.

## 21. Confirmed bugs versus design risks

Confirmed reproducible bugs:

1. Qwen structured vision rejects the `response_schema` argument used by active
   tools.
2. Tool health reports Qwen-backed structured vision tools as available despite that
   incompatible call signature.
3. Reducer records Discovery from top-level error results.
4. `crop_and_inspect` accepts reversed and negative bounding boxes; later clipping can
   turn them into a different tiny crop instead of rejecting the call.
5. `count_objects` successful output has no Evidence/Finding reducer path.
6. Model-visible search/visit output is a strict subset of reducer-visible output.

Design risks requiring real calibration rather than only a code fix:

- VLM same-capture confidence and threshold;
- VLM anomaly/authenticity semantics;
- OCR confidence threshold;
- source-class and independence weighting;
- full-page evidence chunk retrieval strategy;
- maximum search/visit fan-out per policy action.

## 22. gpu-13 real-provider probes on 2026-07-16

The committed audit head `cc9b16d2b780f9314153ce6783239c60bf6e733d`
was fast-forwarded to the clean gpu-13 checkout before these probes. All project
commands ran through the `ifv-agent` kernel with `OMP_NUM_THREADS=1`. Credentials
were loaded from the existing untracked project environment and were never printed
or persisted.

### `text_search`

Observed provider accounting:

```text
one normal query       -> 1 Serper POST, 1 credit
one nonsense query     -> 1 Serper POST, 1 credit, 5 rewritten-looking results
two queries in action  -> 2 Serper POSTs, 2 credits
empty query list       -> 0 requests, status=error
```

The nonsense query did not produce an empty result. Google/Serper returned unrelated
GitHub, Codex, Verizon, IBM, and Cisco rows. Therefore result count is not progress,
and action count is not search cost. Query and retry counts must be explicit.

### `visit`

Three real behaviors were reproduced:

1. A USDA official page that had fetched successfully moments earlier failed through
   Jina with an SSL record-layer error. The tool returned `status=error` and made no
   direct-fetch fallback attempt.
2. A NOAA PDF contained the requested monarch/Mexico terms at about character 62,000
   in the cleaned document. The extractor received only the first 12,000 characters
   and returned `low/unclear/none` with a summary that the document did not mention
   monarch butterflies or Mexico.
3. A nonexistent domain produced one Jina request and a typed top-level error without
   an extraction call.

Additional static position probes found relevant terms at about 34,000 to 66,000
characters in several NOAA documents. `extract_max_chars=60000` does not currently
help because `max_chars=12000` is applied first.

### `reverse_image_search`

One Apple Tysons Corner image action expanded into:

```text
1 OSS upload
1 Serper Lens request
1 Gemini structured-vision query generation
1 Serper semantic image search
```

Lens returned five unrelated clothing/product rows. The semantic branch generated
`Apple Tysons Corner reopening first customer` and found the official Apple page.
The combined action reported success. Lens and semantic branches therefore require
separate outcomes and accounting.

### `compare_with_reference`

Using one backend in the normal asynchronous execution path:

```text
exact same image -> same_capture_or_near_duplicate=true, confidence=1.0
unrelated Apple/monarch images -> same_subject=false, unrelated_content, confidence=1.0
```

The visual decisions were correct for these two easy probes. Repeated direct use of
the synchronous compatibility wrapper with a shared async backend reproduced
`Event loop is closed`; the orchestrator uses `call_async`, so this is a development
API lifecycle defect rather than the current trajectory failure.

### Scene perception and local visual tools

Gemini scene perception correctly described both the Apple event photo and the
synthetic monarch/Antarctica image. The monarch scene description explicitly
identified the polar setting, penguin, iceberg, pine tree, and monarch butterflies.

`crop_and_inspect` correctly recovered `Tysons Corner` from a focused Apple-shirt
crop. It also accepted the invalid box `[0.8, 0.8, -0.2, -0.2]`, silently converted
it into a different one-pixel crop, spent one Gemini call, and returned a successful
description of a solid color.

`count_objects` returned a plausible count of 12 people but produced 50 malformed or
duplicated free-text location strings. Its stricter bbox validator correctly rejected
a reversed crop. Successful count output still has no reducer landing.

### OCR

Whole-image EasyOCR on the Apple photo returned:

```text
U          confidence=0.037
TyU(TI T   confidence=0.081
```

The tool exposed both strings through `full_text`, while focused Gemini crop
inspection correctly read `Tysons Corner`. Bootstrap filters OCR regions below 0.5,
but the later reducer consumes unfiltered `full_text`; the two runtime paths disagree.

DEFAME also names EasyOCR, but its reader initialization is commented out and its
implementation contains a PaddleOCR TODO. It is not evidence that the current
EasyOCR-only path is mature.

OpenSearch-VL at commit `236e0e07ded730e66cf6e85ad39d5a34e403dbca`
uses a PP-StructureV3-compatible layout-parsing service and exposes perspective
correction, super-resolution, and sharpening before document text extraction.
MMSearch, DeepEyes, and UI-TARS did not expose a comparable standalone traditional
OCR tool in the inspected commits.

Disposition: retain a cheap detector only as one layer. Separate document/layout OCR
from natural-image scene text, exclude low-confidence strings from canonical text,
reuse the reader, and use focused Gemini/Qwen-VL verification for decisive small text.

### Integrity tools

`check_consistency` judged the Apple photo consistent and the monarch/Antarctica
scene inconsistent. Its refutation was based on ecological knowledge and scene
semantics rather than pixel forensics.

The general visual-anomaly tool falsely labeled the real Apple official photo
`likely_ai` with confidence 0.7 because of allegedly distorted shirt text and hands.
It labeled the generated monarch image `likely_ai` with confidence 0.8. This confirms
that the tool is not safe as standalone authenticity Evidence. Clean scans also cannot
support authenticity.

### `crop_and_search`

One focused Apple-shirt action expanded into:

```text
1 crop and persistent artifact copy
1 OSS upload
1 Serper Lens request
1 Gemini query-generation call
1 Serper semantic image-search request
1 Jina page fetch
1 Gemini page-extraction call
```

It generated the useful query `Apple Store Tysons Corner shirt`, but selected an eBay
page, returned `low/unclear/none`, and consumed two model calls. This confirms that
`crop_and_search` is a hidden compound sub-agent and must not remain one atomic
verification action.

## 23. Real-probe disposition

The pre-repair probes change the repair priority as follows:

1. Prevent top-level errors and low-confidence OCR garbage from mutating state.
2. Make provider/tool health capability-aware, especially for Qwen structured vision.
3. Use one canonical bounded payload for policy context and reduction.
4. Record actual query, fetch, upload, extraction, and model subcalls.
5. Split or demote `crop_and_search`.
6. Repair full-document visit selection and Jina/direct fallback.
7. Treat VLM consistency/anomaly output as diagnostic unless a registered
   authenticity gap and qualified forensic signal exist.
8. Replace EasyOCR-only decisive text extraction with layered OCR plus focused VLM
   verification.

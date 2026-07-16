# Tool Runtime Contract

**Date:** 2026-07-16

**Status:** frozen for the search-control repair

## Purpose

Every verification action must be bounded, auditable, and tied to one open evidence
gap. Tool output is not verdict evidence merely because a provider returned data.

## Canonical result

Each tool returns exactly one bounded JSON object:

```json
{
  "status": "success",
  "completion": "complete",
  "result_kind": "discovery",
  "subcalls": [],
  "data": {}
}
```

The transition implementation may preserve existing top-level tool fields while
moving toward this envelope, but the semantics are fixed:

- `status` is `success` or `error`;
- `completion` is `complete` or `partial` for successful results;
- `result_kind` is `discovery`, `evidence`, `observation`, `diagnostic`, or `failure`;
- `subcalls` records every real provider operation;
- one normalized payload is used both for model context and state reduction.

An error result cannot create Discovery, Evidence, Finding, or positive progress.
A partial result may create records only from explicitly successful subresults.

## Subcall record

Each real operation is counted separately:

```json
{
  "kind": "search_query",
  "provider": "serper",
  "status": "success",
  "request_count": 1,
  "result_count": 5,
  "duration_ms": 1234.5
}
```

Allowed kinds include:

```text
search_query
image_search_query
reverse_image_query
upload
page_fetch
page_extract
vision_extract
image_compare
ocr
forensic_scan
```

Retries are request counts, not additional policy actions. Budgets and traces expose
both action count and subcall count.

## Discovery versus Evidence

Search rows, image matches, generated queries, titles, snippets, and candidate URLs
are Discovery only.

Evidence requires one of:

- an eligible exact webpage passage with source provenance;
- a validated OCR/text observation tied to a registered visible-text gap;
- a qualified reference comparison tied to a registered source-binding gap;
- a specialized forensic signal tied to an image-authenticity gap.

General VLM anomaly opinions are diagnostics. A clean anomaly scan never supports
authenticity.

## Action boundary

One policy action performs one bounded semantic operation:

```text
text_search            -> one query
visit                  -> one page fetch plus one extraction
reverse_image_search   -> one explicitly selected lens or semantic branch
compare_with_reference -> one reference pair comparison
ocr_with_position      -> one image or one crop
```

`crop_and_search` violates this boundary. The active runtime therefore does not expose
it or `count_objects` to the Agent. Their standalone implementations remain available
for offline diagnostics and future explicitly scoped visual-question work; any future
crop-search flow must use separate crop, reverse/semantic search, visit, and comparison
actions.

## Observation adjudication

Visual and OCR tools report observations. Deterministic runtime code compares an
observation with the `expected_property` registered on the open evidence gap.

Unscoped observations stay neutral. Tools cannot attach their output to every fact on
a task merely because those facts share a task ID.

## OCR

Canonical OCR text contains only accepted regions. Low-confidence candidates remain
in diagnostics and cannot enter `full_text`, Evidence, retrieval anchors, or verdict
bases.

Document/layout text and natural-image scene text are separate routes. Set
`PPOCR_SERVICE_URL` to use a PP-OCR/PP-Structure-compatible positioned-OCR service;
otherwise the runtime uses EasyOCR. A configured service failure falls back to
EasyOCR and records both attempts. Decisive small
or stylized text requires focused crop verification when the primary OCR result is
missing, low-confidence, or conflicts with a VLM reading.

## Stopping

Only qualified progress on the stable `CoreVerdictFact` or one of its open evidence
gaps resets saturation. New rows, IDs, metadata, diagnostics, or failed subcalls do
not reset it.

Coverage runs after every accepted evidence update and owns stopping:

```text
qualified refutation -> fake
qualified support plus required binding, no material conflict -> real
no executable decisive route or two no-gain checkpoints -> unverifiable
hard engineering budget exhausted -> incomplete
```

## Compatibility

This contract does not preserve old runtime behavior. Old traces may be parsed for
migration tests, but current execution must follow the contract above.

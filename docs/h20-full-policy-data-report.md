# Full-trajectory H20 data report — 2026-09-11

Scope: complete policy episodes only. Legacy standalone perception rows and the
28 action-only cases are not training inputs. Images and perception-tool responses
inside complete episodes remain intact. No training was started.

## Processing and integrity

The frozen delivery contains 2,578 policy train rows and one validation row, with
no test rows. All 2,579 rows passed the actual Qwen3.5 processor with the H20
contract: max_length 131072, raise on overflow, max_pixels 262144,
image_max_token_num 1024, padding-free, SP4, ignore_empty_think, and no fabricated
thinking prefix. Model weights were not loaded for this data audit.

Original messages, tool schemas, row order and split assignments are unchanged;
only image filesystem paths were rebound. No row was truncated, split, deduplicated
or discarded. File hashes and per-row features bind the result to its source.
Both strict structural audit and the existing launch data gate passed. Case IDs,
frozen groups, source images, referenced images and exact rows have no train/validation
overlap. This is integrity/processor verification, not a new semantic judge run.

## Train split characteristics

| Feature | Total | Mean per episode | Median | P95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Input tokens | 62,673,659 | 24,311 | 23,466 | 38,306 | 56,100 |
| Supervised tokens | 13,004,044 | 5,044 | 4,797 | 8,090 | 15,224 |
| Image references | 11,274 | 4.37 | 4 | 11 | 18 |
| Tool calls | 32,776 | 12.71 | 12 | 21 | 24 |
| Messages | 106,062 | 41.14 | 39 | 66 | 75 |

Input token minimum is 7,328. Supervised tokens represent 20.75% of input tokens.
All splits together retain 32,784 tool calls and 32,784 tool responses. The train
split has 2,578 unique cases and no exact duplicate rows. There are 1,348 `main`
cases and 1,230 route-aware/other cases.

| Actual input length | Train episodes |
| --- | ---: |
| ≤8,192 | 1 |
| 8,193–16,384 | 373 |
| 16,385–32,768 | 1,865 |
| 32,769–65,536 | 339 |
| >65,536, up to 131,072 | 0 |

The old token estimate uses UTF-8 bytes, not this tokenizer's token count. This
dataset has no real single-episode 64K–128K training examples. Keeping the 128K
context ceiling and packing episodes to longer sequences does not establish
learning of long-range dependencies across 128K within one episode.

Teacher final verdicts are 1,651 fake / 927 real; these are teacher outputs, **not
independently checked ground-truth labels**. Tools are used as follows:

| Tool | Calls in train |
| --- | ---: |
| text_search | 7,598 |
| visit | 5,032 |
| text_image_search | 3,885 |
| reverse_image_search | 2,658 |
| finish_investigation | 2,570 |
| compare_with_reference | 2,328 |
| perceive_scene | 2,326 |
| ocr_with_position | 2,270 |
| analyze_visual_anomalies | 1,757 |
| check_consistency | 913 |
| crop_and_inspect | 776 |
| focused_visual_inspection | 599 |
| count_objects | 51 |
| current_time | 13 |

## Images, packing and limitations

The complete train/validation episodes reference 11,253 unique image files, totaling
2,223,063,123 bytes. Both dimensions are at most 1,024 pixels, consistent with the
runtime-media projection rather than using all original-image files in the delivery.
Three referenced files have a dimension below 28 pixels (28×24, 69×23, 1×1).
Their row indices are retained for inspection, and no tool-return image was silently
deleted. They passed processor encoding; that does not establish useful visual content.

An index-only first-fit-decreasing 120K packing plan produces 531 packs, 98.36%
token-budget fill, averaging 118,029 tokens and 21.23 images; the largest image
count is 40 per pack. Each train row appears exactly once. This is not a claim
that ms-swift will choose the same packing order or that these fuller/more-image
packs fit GPU memory. Worst-case packed-shape testing remains necessary before
production; the earlier seven-row benchmark did not cover this distribution.

The single validation episode contains 16,854 input tokens, 2,988 supervised tokens,
three images and eight tool calls. The original frozen split had eight validation
candidates, only one of which appears in this accepted policy package. Do not
resplit accepted rows or pretend this one example can support reliable checkpoint
selection. No independent test split was created.

## Reproduction and storage

Run `training/scripts/h20/prepare_full_data.py` with a new output directory in the
authorized server root, then `finalize_full_data.py --processed <output>`, using
`environment.sh` and the isolated H20 environment. The latter verifies content
equivalence and binds the existing launch gate, and writes `RELEASE.json`.
All images remain referenced in the original delivery; do not remove that source
directory. Generated JSON/metadata consume approximately 236 MiB; no image copies
or model checkpoints are created. Superseded temporary processing directories were
removed, while diagnostic logs were retained.

Final train SHA-256:
`f9b7bbe3f833eaf19c73459ca86ea288946c8b4ba347fd2d152439968c9112fd`.

All code commits/pushes are performed locally; the server only fast-forwards the
committed checkout. The JSONL Unicode-separator regression, existing processor
probe, data-contract and policy-adapter tests passed (17 tests).

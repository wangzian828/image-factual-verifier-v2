# Filtered test-set recovery notes — 2026-09-11

The existing comparison uses 1,527 cases (377 real, 1,150 fake), as documented
in `evaluation-comparison-20260909.md`. Do not create a replacement split or use
the legacy full archive as if it were this filtered release.

## Historical location, not yet recovered on H20

The previous task history records this directory on the old server:

`/gsdata/home/wza/image-factual-verifier-v2-data/datasets/route-aware-hrc-stage2-test1527-agent-qa-filtered-20260909-r1`

Recorded contents:

- `test-manifest.jsonl` (1,527 rows)
- `evaluator_private/private-gold-v1/private-gold.jsonl` (1,527 rows)
- `removed-index.jsonl` (157 rows), `removed-case-ids.txt`
- `selection-summary.json`, `README.md`, `.complete`
- `images` symlink to the original dataset image directory

These names and counts were recovered from historical task records, not freshly
validated against the old server. The current H20 data directory and repository
do not yet contain the recovered authoritative filtered manifest.

History describes filtering 157 from an original 1,684: 79 capability-balancing
removals (38 source_grounded_mutation, 41 no_prototype_fabrication), plus 70
faithful_real_event and eight web_refuted cases where Agent errors intersected
second-pass QA rejection. Verify these details against selection-summary before
publishing methodology. This is a model-result-informed filtered benchmark, not
an untouched blind test; freeze IDs for before/after evaluation and disclose the
selection. Do not filter again using the new model's results.

## Downloaded legacy archive

Only `factcheck_test-1682-20260907.tar.gz` was downloaded from the user-specified
ModelScope dataset `jiashuhong/factcheck_test`; no loose images were downloaded.

H20 path: `/volume/ybo/wza/data/downloads/factcheck_test-1682-20260907.tar.gz`

Size: 3,074,465,877 bytes (about 2.86 GiB).

Verified SHA-256:
`485dba3b8b3947913f372b57f85dbe30b456894022b3fa467c9460dcb8847ea5`

It matches the repository's published archive digest. No images were extracted.
The archive names 1,682 cases, whereas historical filtering starts from 1,684:
do not assume all 1,527 retained cases are present. Reconcile by case ID and image
hash after retrieving the frozen manifest; investigate missing cases explicitly.

## Evaluation boundary

Keep all test data out of teacher rollout, SFT and RL. Runtime receives public
case/image inputs only; private gold remains evaluator-only. Reuse the frozen
1,527 denominator and BAcc, per-class recall, and SESR definitions in the existing
comparison report. Existing larger-model scores are reference results, not a
substitute for measuring the actual untrained Qwen3.5-9B baseline. This task has
not launched paid API evaluation or any new training.

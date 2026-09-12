# Filtered test-set recovery notes — 2026-09-11

The existing comparison uses 1,527 cases (377 real, 1,150 fake), as documented
in `evaluation-comparison-20260909.md`. Do not create a replacement split or use
the legacy full archive as if it were this filtered release.

## Recovered frozen release

The previous task history records this directory on the old server:

`/gsdata/home/wza/image-factual-verifier-v2-data/datasets/route-aware-hrc-stage2-test1527-agent-qa-filtered-20260909-r1`

Recorded contents:

- `test-manifest.jsonl` (1,527 rows)
- `evaluator_private/private-gold-v1/private-gold.jsonl` (1,527 rows)
- `removed-index.jsonl` (157 rows), `removed-case-ids.txt`
- `selection-summary.json`, `README.md`, `.complete`
- `images` symlink to the original dataset image directory

The directory was re-opened and validated on gpu-13 on 2026-09-12. The manifest,
private gold, selection summary and removed-case list were transferred to H20 as
metadata only; no test images were copied and no Judge API was called.

H20 metadata root:

`/volume/ybo/wza/data/factcheck-test-1527-filtered-frozen-20260909`

Verified SHA-256 values:

- `test-manifest.jsonl`: `9df1b0f6f8b6285f411d600fa230a70fdce4cefe9c2be264f7bd8b925358d1f0`
- `private-gold.jsonl`: `49a5785522b982dc4f97b28270a4b5bf5f4da0d7247f8dc7fd30cdf0de5ea671`
- `selection-summary.json`: `733e32dc09c8e8e9f620d43631811e22d4ead27da8cfdab6d5cecf513616310f`
- `removed-case-ids.txt`: `b9479c69ae0e76f4c7a991ed6aa4ea86412a581987a098f1b5e8028a869adf6b`

Exact case-ID reconciliation against the completed old-1,682 baseline found
1,526 shared cases, one formal case absent from the old run, and 156 old-run
cases outside the formal release. The formal summary counts the missing real
case as a miss rather than shrinking the denominator.

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

# Qwen3.5-9B Agent pre-training baseline — formal 1,527 denominator

## Scope

This is the frozen pre-training baseline for the first H20 full-parameter SFT
experiment. The denominator is the historical filtered benchmark of **1,527
cases**: 377 `real` and 1,150 `fake`. The frozen split was recovered from the old
gpu-13 server; no new filtering used the Qwen3.5-9B results.

No Agent inference or Judge API call was repeated for this report. Exact case-ID
matching reused 1,526 completed records from the already-finished old-1,682 run
and its corrected evidence audit. The one formal case absent from that run is
counted as a class miss and as not `strong`, preserving the historical fixed-
denominator rule.

All 1,526 reused rows were independently reconciled against the recovered formal
private-gold sidecar; gold-verdict mismatches: **0**.

## Binary result

| Metric | Result |
| --- | ---: |
| Formal denominator | 1,527 |
| Reused completed Agent records | 1,526 |
| Missing/invalid formal records | 1 |
| Correct | 969 |
| Accuracy | 63.4578% |
| Balanced accuracy | 65.6654% |
| Real recall | 70.0265% (264/377) |
| Fake recall | 61.3043% (705/1,150) |

Confusion matrix, with gold labels as rows:

| Gold | Predicted real | Predicted fake | Missing/invalid |
| --- | ---: | ---: | ---: |
| Real | 264 | 112 | 1 |
| Fake | 445 | 705 | 0 |

The missing real case is
`route-aware-hrc-final-3000-20260816-input:2428:web_r021-web_crawled_supported-web-backlog-00004`.
It was not silently removed from the denominator.

## Evidence result

The reused audit was produced with the comparison-standard configuration:
`gemini-3.7-flash`, `thinking_level=low`, the shared private-gold prompt/schema,
8,192 maximum output tokens, and zero final API or parsing errors.

| Quality bucket | Count | Formal-denominator rate |
| --- | ---: | ---: |
| `strong` | 44 | 2.8815% |
| `usable` | 156 | 10.2161% |
| `rejected` | 1,326 | 86.8369% |
| Missing/unjudged | 1 | 0.0655% |

The formal strict evidence sufficiency rate (SESR) is therefore
**44/1,527 = 2.8815%**. The missing case contributes zero. The obsolete
`gemini-3.1-pro-preview`, high-thinking audit is not used anywhere in these
statistics.

## Provenance

Recovered frozen release on H20:

`/volume/ybo/wza/data/factcheck-test-1527-filtered-frozen-20260909`

| Input | SHA-256 |
| --- | --- |
| `test-manifest.jsonl` | `9df1b0f6f8b6285f411d600fa230a70fdce4cefe9c2be264f7bd8b925358d1f0` |
| `private-gold.jsonl` | `49a5785522b982dc4f97b28270a4b5bf5f4da0d7247f8dc7fd30cdf0de5ea671` |
| `selection-summary.json` | `733e32dc09c8e8e9f620d43631811e22d4ead27da8cfdab6d5cecf513616310f` |
| `removed-case-ids.txt` | `b9479c69ae0e76f4c7a991ed6aa4ea86412a581987a098f1b5e8028a869adf6b` |

Formal summary root:

`/volume/ybo/wza/runs/eval/qwen35-base-agent-formal1527-pretrain-from-old1682-20260912`

The root contains `summary.json`, 1,526 selected audit records and the explicit
missing-case record. Its source corrected-audit SHA-256 is
`a89ebf168939e1a50abed39f4f71d50a27741ec898398eae6252405e37a917ac`;
the selected-audit SHA-256 is
`b123552f3b7c32231bddb254ed59067f3c9b827ab62d706d85fac1d05725c02e`.

## Interpretation boundary

This baseline is valid for the historical 1,527 denominator, but one case lacks
a Qwen3.5-9B Agent attempt. It is conservatively counted as failure. The post-
training run should execute all 1,527 cases directly if the missing image can be
recovered, then use the same Gemini 3.7 Flash/low Judge. No test result may be
used for hyperparameter selection or additional training.

# Gemini 图像直接事实核查实验记录

**记录范围：** 测试集上的 Gemini direct QA 对照实验，以及与其对应的 100 条
Agent 对照。本文记录截至 **2026-08-30 20:47 UTC** 已落盘的结果；Gemini 3.7
全量 direct QA 尚在自动补跑，因此其全量指标只是进行中快照，不是最终结论。

## 1. 数据、输入与指标口径

| 项目 | 记录 |
|---|---|
| 测试集 | 统一数据集 `test-manifest.jsonl`，共 1,684 条 |
| 标签分布 | real（supported）447；fake（refuted）1,237 |
| 模型可见输入 | 仅一张图片 + 共用 direct-QA prompt |
| 模型不可见输入 | 标签、private gold、图片来源、URL、生成 prompt、能力格子和任何 Agent 轨迹 |
| 输出 | `core_fact`、二分类 `verdict`、`reason`；模型原生 thought 单独保存 |
| 共用 prompt | 两个全量 run 的 `prompt.txt` SHA-256 均为 `6a814810ee32101a557f498e49bacaabb7b65c12e71c383f2f4bf1cc801dd211` |
| 二分类指标 | Accuracy；BACC = (real 召回率 + fake 召回率) / 2 |

private gold 只在模型输出后由 judge 使用。三分类的定义是：

| 分类 | 规则 |
|---|---|
| 判断正确且理由充分 | `verdict_matches_gold=true` 且 `quality_bucket=strong` |
| 判断正确但理由不足 | `verdict_matches_gold=true` 且质量桶不是 `strong` |
| 判断错误 | `verdict_matches_gold=false` |

direct QA 没有搜索或外部 Evidence；这里的“理由充分”仅评价它输出的
`core_fact` 与 `reason` 是否足以支持该结论，不表示它检索到了外部证据。

## 2. 100 条分层对照

样本从五个构造子路线各抽 20 条；三种方法都完成了 100 条最终候选。

| 方法 | Accuracy | BACC | real 召回 | fake 召回 | 首轮工程错误 |
|---|---:|---:|---:|---:|---:|
| Gemini 3.7 Agent | 71.00% | 75.00% | 95.00% | 55.00% | 29，补跑后替换 |
| Gemini 3.7 direct QA | 73.00% | 71.25% | 62.50% | 80.00% | 0 |
| Gemini 3.1 Pro direct QA | 73.00% | 67.92% | 42.50% | 93.33% | 0 |

100 条的 Agent 审计以当前认可的“正确且有决定性、落地的依据”口径重新读取为：

| 方法 | 判断正确且理由/依据充分 | 判断正确但理由/依据不足 | 判断错误 |
|---|---:|---:|---:|
| Gemini 3.7 Agent | 32 | 39 | 29 |
| Gemini 3.7 direct QA | 25 | 48 | 27 |
| Gemini 3.1 Pro direct QA | 25 | 48 | 27 |

Agent 的第一类要求实际 trace Evidence 与最终 report 均能支撑结论；direct QA
的第一类只能说明其图像直答理由足够。这两者可以共同报告三分类，但不能把
direct QA 的第一类解释成“找到了外部证据”。

100 条历史对照的原始过程、首次运行错误和旧审计产物保留在
[`2026-08-30-gemini37-test100-comparison.md`](2026-08-30-gemini37-test100-comparison.md)。
该旧报告中的 Agent `quality_bucket=strong` 计数 20 是已废弃的严格覆盖度读法，
不作为当前主指标。

## 3. 全量测试集 direct QA

### 3.1 Gemini 3.1 Pro：已完成

运行配置：`thinking_level=high`、`max_output_tokens=4096`、单次超时 240 秒、
最多重试 2 次、并发 10。

| 项目 | 数值 |
|---|---:|
| 已完成 QA | 1,682 / 1,684 |
| 原始 QA 工程错误 | 2（1 SSL、1 Gemini HTTP 错误） |
| Accuracy（仅完成样本） | 77.47%（1,303 / 1,682） |
| BACC（仅完成样本） | 60.82% |
| real 召回 | 25.28%（113 / 447） |
| fake 召回 | 96.36%（1,190 / 1,235） |

混淆矩阵（行为 gold，列为预测）：

| gold \ prediction | real | fake |
|---|---:|---:|
| real | 113 | 334 |
| fake | 45 | 1,190 |

private-gold judge 已完成 1,682 条；2 条原始 QA 工程错误保留为错误，不强行审计。

| 三分类 | 条数 | 占全部 1,684 条 | 占已完成 1,682 条 |
|---|---:|---:|---:|
| 判断正确且理由充分 | 366 | 21.73% | 21.76% |
| 判断正确但理由不足 | 937 | 55.64% | 55.71% |
| 判断错误 | 379 | 22.51% | 22.53% |
| 工程错误 | 2 | 0.12% | — |

审计附加诊断（均基于 1,682 条已完成 QA）：

| `reason_quality` | 条数 | `fact_match` | 条数 |
|---|---:|---|---:|
| `decisive_and_grounded` | 408 | `same_fact` | 389 |
| `artifact_based` | 690 | `compatible_subfact` | 531 |
| `partially_grounded` | 149 | `overgeneralized_subfact` | 599 |
| `unsupported` | 236 | `different_fact` | 163 |
| `contradictory` | 199 |  |  |

结论：3.1 Pro 的 Accuracy 看起来较高，主要来自对 fake 的极强偏好；对 real 的
召回只有 25.28%，所以 BACC 只有 60.82%。不能仅以 Accuracy 将它描述为更好。

### 3.2 Gemini 3.7 Flash：自动补跑中

当前恢复配置：`thinking_level=high`、`max_output_tokens=4096`、单次超时 240 秒、
最多重试 3 次、并发 8。最初的高并发运行已写入同一结果目录；此处以最新 case
记录为准。

| 项目 | 当前快照 |
|---|---:|
| 已完成 QA | 1,638 / 1,684 |
| 待恢复工程错误 | 46 |
| 当前 Accuracy（仅已完成） | 73.20%（1,199 / 1,638） |
| 当前 BACC（仅已完成） | 70.84% |
| 当前 real 召回 | 65.69%（291 / 443） |
| 当前 fake 召回 | 75.98%（908 / 1,195） |
| 当前错误构成 | 4 SSL、22 `ReadTimeout`、20 Gemini HTTP 错误 |

当前混淆矩阵（仅 1,638 条已完成样本）：

| gold \ prediction | real | fake |
|---|---:|---:|
| real | 291 | 152 |
| fake | 287 | 908 |

这组尚未启动全量 private-gold judge。必须等 46 条恢复完成、重写 run summary 后，
再对完整 1,684 条一次性审计；不能用当前 partial QA 的 BACC 和 3.1 的最终
private-gold 三分类做最终模型比较。

## 4. 可复现产物

| 用途 | 路径 |
|---|---|
| 测试 manifest | `/gsdata/home/wza/image-factual-verifier-v2-data/datasets/route-aware-hrc-stage2-10563-final-organized-20260824-r2/unified-dataset/test-manifest.jsonl` |
| private gold sidecar | `/gsdata/home/wza/image-factual-verifier-v2-data/datasets/route-aware-hrc-stage2-10563-final-organized-20260824-r2/unified-dataset/evaluator_private/private-gold-v1/test-private-gold.jsonl` |
| 3.7 全量 direct QA | `/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini37-preview-image-only-test1684-full-high-20260830-r1/` |
| 3.1 全量 direct QA | `/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini31pro-preview-image-only-test1684-full-high-20260830-r1/` |
| 3.1 全量 private-gold 审计 | `/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini31pro-preview-image-only-test1684-full-high-20260830-r1/private-gold-audit-20260830-r2/` |
| 100 条 Agent 报告回填与审计 | `/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini37-agent-test100-agent-private-gold-audit-with-reports-20260830-final/` |

代码入口：

- `scripts/run_direct_qa_baseline.py`：只构造图片 + prompt 请求并写入原始输出。
- `scripts/audit_direct_qa_baseline.py`：在 QA 完成后读取 private gold 进行审计。
- `src/eval/private_gold_metrics.py`：统一三分类汇总。

## 5. 后续更新规则

1. Gemini 3.7 补完 46 条后，先确认每个 `case_id` 只有一个最新 completed 结果；
   重新生成其 `summary.json` 和 Accuracy/BACC。
2. 对完整 1,684 条 3.7 QA 启动一次全量 private-gold judge，再补写其三分类与审计
   附加诊断。
3. 任何后续 prompt、模型、思考强度、输出上限、超时、并发或测试 manifest 改动，都
   建立新 run 目录，不能覆盖本记录中的产物。

## 6. 2026-08-31 Gemini 3.7 private-gold judge 最终结果

3.7 全量测试集共 1,684 条，已全部完成审计，0 条审计错误。三分类按全部
1,684 条计算：

| 分类 | 数量 | 比例 |
|---|---:|---:|
| 判断正确且证据充分 | 143 | 8.49% |
| 判断正确但证据不足 | 1,097 | 65.14% |
| 判断错误 | 444 | 26.37% |
| 判断正确合计 | 1,240 | 73.63% |

最终审计目录：

`/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini37-preview-image-only-test1684-full-high-20260830-r1/private-gold-audit-gemini37-20260831-low-resume-r2/`

本节结果覆盖第 3.2 节中的 partial 快照；后续比较应使用本节的完整结果。

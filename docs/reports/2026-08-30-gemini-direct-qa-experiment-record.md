# Gemini 图像直接事实核查实验记录

**记录范围：** 测试集上的 Gemini direct QA 对照实验，以及与其对应的历史 100 条
Agent 对照。本文记录截至 **2026-09-02** 已落盘的结果；Gemini 3.7
全量 direct QA 和 private-gold 审计均已完成。

本文中的 2026-08-30/31 对照数字是历史基线，不覆盖后续新版 Agent-100 诊断结果。
新版 Agent-100 的最终 34/40/25 分类和逐条问题，以
`reports/2026-09-02-agent100-newagent-rollout-and-review.md` 为准。Gemini 3.7
当前是否在线不影响这些已落盘结果。

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

### 3.2 Gemini 3.7 Flash：已完成

运行配置：`thinking_level=high`、`max_output_tokens=4096`、单次超时 240 秒、
最多重试 3 次。结果文件包含追加式重试记录，最终统计按 `case_id` 只保留最新一条
结果；1,684 个 case 全部完成，0 个 QA 工程错误。

| 项目 | 数值 |
|---|---:|
| 已完成 QA | 1,684 / 1,684 |
| 原始 QA 工程错误 | 0 |
| Accuracy | 73.63%（1,240 / 1,684） |
| BACC | 71.05% |
| real 召回 | 65.55%（293 / 447） |
| fake 召回 | 76.56%（947 / 1,237） |

混淆矩阵（行为 gold，列为预测）：

| gold \ prediction | real | fake |
|---|---:|---:|
| real | 293 | 154 |
| fake | 290 | 947 |

private-gold judge 同样完成 1,684 / 1,684，0 审计错误。三分类见第 6 节。

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

## 5. 结果解释与后续规则

1. 3.7 结果按 `case_id` 去重后统计；追加式重试的旧记录保留，但不重复计入最终指标。
2. private-gold 只在模型输出后使用，不进入 direct QA 或 Agent 的模型输入。
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

## 7. unified-react-v1 10 条真实 smoke

新版 ReAct smoke 运行目录：

`/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/unified-react-v1-gemini37-train-smoke-20260831-f88352a-p10/`

| 项目 | 结果 |
|---|---:|
| canonical trace | 10 / 10 |
| strict trace audit | 10 / 10，0 warning，0 protocol/scheduler/route rejection |
| 工程错误 | 0 |
| 平均耗时 | 247.288 秒 |
| 平均工具调用 | 7.4 |
| 平均 Gemini 请求 | 11.4 |
| provider-visible thought | 原始 trace 均有 thought 统计字段；9 / 10 条满足每个 ReAct action 都有可读 thought |

当前提交 `62f6b9c` 相对该 smoke 所用的 `f88352a` 只更新了 claimless unified-react
canary 的验收逻辑和测试，没有改 ReAct runtime、工具、prompt 或 SFT 导出逻辑。
因此已在服务器当前 checkout 上直接复核这批 trace 的 canary 合约，结果通过。

SFT 下游分桶复核：

- `reasoning_sft`：9 条；
- `action_only`：1 条（`main-02754`，没有 provider 可读 thought，不伪造 `<think>`）；
- `rl_candidate`：10 条；
- 10 条中有 1 条导出估算超过 128K，`main-02755` 约 199,805 tokens，应进入长轨迹
  holdout，不应直接进入短轨迹 SFT。

## 8. Agent-100 启动决策

以下是本节形成时的历史启动决策。之后按人工要求完成了新版 Agent-100
诊断性重跑；新版结果不覆盖本节的历史对照，见
`reports/2026-09-02-agent100-newagent-rollout-and-review.md`。

新版 10 条 smoke 的统一 private-gold 对照为：

| 版本 | 判断正确且证据充分 | 判断正确但证据不足 | 判断错误 |
|---|---:|---:|---:|
| 旧版 Agent | 1 | 6 | 3 |
| 新版 Agent | 4 | 1 | 5 |

新版的强证据桶有所增加，但总正确数由 7 / 10 降至 5 / 10，不能判定为调查质量整体
明显改善。因此本计划不启动新的 100 条 Agent rollout；现有旧版 Agent-100 只保留为
历史基线。大规模教师 rollout 同样保持未启动，等待人工复核。

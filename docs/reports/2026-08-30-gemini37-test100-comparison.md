# Gemini 测试集 100 条对比记录

## 实验设置

- 样本：测试集 100 条
- 抽样：5 个构造子路线各 20 条
- gold：100/100；其中 7 条使用历史 archive 回读补全
- Agent：Gemini 3.7 Flash，统一 ReAct，并发 24；首轮 71 条完成，29 条由补跑替换，最终审计候选为 100 条
- direct QA：Gemini 3.7 Flash / Gemini 3.1 Pro，使用同一中文事实核查 prompt

## 结果

| 方案 | 最终审计候选 | 工程错误（首轮） | private-gold 判断正确 | Accuracy | BACC |
|---|---:|---:|---:|---:|---:|
| 3.7 Agent | 100/100 | 29 | 71/100 | 71.00% | 75.89% |
| 3.7 direct QA | 100/100 | 0 | 73/100 | 73.00% | 71.25% |
| 3.1 direct QA | 100/100 | 0 | 73/100 | 73.00% | 67.92% |

3.7 Agent 首轮的 29 个工程错误中，22 个是 Gemini HTTP 429，7 个是整条
1800 秒预算耗尽。补跑后形成 100 条最终候选；下面的 private-gold 三分类按这
100 条候选统计，首轮工程错误不重复计入错误分类。若把首轮工程错误也保守地
计错，旧口径的整体 BACC 为 55.83%。

3.7 direct QA 平均单条耗时约 238 秒，3.1 direct QA 约 310 秒。两者均
100/100 返回 native thought。

## 统一 private-gold 三分类

三组结果现在使用完全相同的定义，唯一依据是审计 JSONL 中的
`verdict_matches_gold` 和 `quality_bucket`：

| 分类 | 规则 |
|---|---|
| 找对点且证据充分 | `verdict_matches_gold=true` 且 `quality_bucket=strong` |
| 判断对但证据不足 | `verdict_matches_gold=true` 且 `quality_bucket!=strong` |
| 判断错 | `verdict_matches_gold=false` |

工程错误、judge 失败、private gold 不可审计不进入这三类，单独计数。Agent 不再
使用 `evidence_chain_recovery` 作为“找对点”的替代定义；该字段是 runtime 过程
指标，不能与 direct QA 的 private-gold 质量桶直接比较。

| 方案 | 找对点且证据充分 | 判断对但证据不足 | 判断错 | 三类合计 |
|---|---:|---:|---:|---:|
| 3.7 Agent | 18 | 53 | 29 | 100 |
| 3.7 direct QA | 25 | 48 | 27 | 100 |
| 3.1 direct QA | 25 | 48 | 27 | 100 |

三分类由共享实现 `src/eval/private_gold_metrics.py` 计算。任意一组审计结果可用
`scripts/summarize_private_gold_audits.py` 重新生成汇总，避免不同实验手工采用
不同口径。

## Private-gold 质量审计

### 3.7 direct QA

- strong：25
- usable：24
- rejected：50
- not_auditable：1

唯一 `not_auditable` 是：

```text
route-aware-hrc-final-3000-20260816-input:0739:baseline_main_generated_r015-pipeline_generated_supported-0003
```

这不是 gold 缺失。gold 行存在，但 judge 认为其中的历史/年份叙述内部不可靠，
因此没有强行纳入理由质量判断。

### 3.1 direct QA

- strong：25
- usable：22
- rejected：53
- gold 可审计：100/100

3.1 的 7 条历史稀疏 gold 已由 archive recovery audit 覆盖旧的
`not_auditable` 结果；最终统计不再使用旧的 93 条局部结果。

## 产物

总对比报告：

```text
/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini37-test100-final-comparison-20260829.json
```

3.7 Agent 测试集 judge：

```text
/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini37-agent-test100-p24-20260829/agent-rollout/test-set-judge-summary.json
```

3.7 direct QA 最终审计：

```text
/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini37-preview-image-only-test100-stratified-high-20260829-r1/private-gold-audit-final/summary.json
```

3.1 direct QA 最终审计：

```text
/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini31pro-preview-image-only-test100-stratified-high-20260829-r3/private-gold-audit-final/summary.json
```

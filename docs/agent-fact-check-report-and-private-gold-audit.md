# Agent 终局事实核查报告与 Private-Gold 审计

## 终局输出

统一 ReAct 的最终 Judgment 仍由 runtime 固定 `verdict`、`verdict_basis` 与所选
Evidence ID；模型新增一个面向读者的 `fact_check_report`：

- `headline`：简短标题；
- `claim_under_review`：图片实际表达、此次核查的事实；
- `verdict_summary`：real/fake 结论和直接理由；
- `key_findings`：1–5 条关键发现；
- `evidence_summary`：这些证据怎样支撑或反驳目标事实；
- `remaining_uncertainties`：不影响当前结论的剩余不确定项。

报告不是证据本体。runtime 从 `verdict_basis.evidence_ids` 编译
`evidence_citations`，保留每条实际 Evidence 的 ID、URL、来源、关系立场和原文节选。
因此报告不能伪造引用，也不会替代现有结构化状态。

## 审计口径

所有方法都用相同 private gold 判断事实是否命中：

- `verdict_matches_gold`：二元结论是否正确；
- `private_gold_fact_match`：是否命中同一完整事实或可决定同一结论的兼容子事实。

Agent 额外按真实 trace 审计：

- `trace_evidence_grounded`：`decisive`、`partial`、`not_grounded` 或
  `contradictory`；
- `report_grounding`：报告是否忠实于实际 Evidence/basis，是否过度声称或虚构来源；
- `quality_bucket=strong`：结论正确、事实命中、选中 Evidence 具有决定性，且报告忠实。

Direct QA 没有搜索、访问或引用。它的审计仍记录相同的 verdict/事实命中字段，但
`trace_evidence_grounded=not_available`；其 `quality_bucket` 只描述图像直答的答案质量，
不能解释为“检索到外部证据”。

## 工具

新增 Agent 专用离线审计：

```powershell
python scripts/audit_agent_private_gold.py `
  --run-dir <agent-run-dir> `
  --manifest <test-manifest.jsonl> `
  --archive-root <private-gold-archive> `
  --output-dir <audit-output>
```

审计器只读取 `run_results.jsonl` 指向的 canonical trace。它会把最终报告、basis、
成功 Evidence、所选 Evidence、运行时引用清单和 private gold 交给 frozen judge；
不会把 private gold 传入 rollout。

历史 trace 没有 `fact_check_report` 时，审计会标记 `report_grounding=missing`，但仍可根据
原始 basis 和成功 Evidence 复审，便于与新 trace 对比。

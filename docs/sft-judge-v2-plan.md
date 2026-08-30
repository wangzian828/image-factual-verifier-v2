# SFT Judge v2 设计与实施计划

## 目标

SFT judge 只回答一个问题：

> 完成的 teacher trajectory 是否正确判断了这张图表达的事实，并找到了足以支撑该判断的证据？

它不要求 teacher 复现构造阶段的 Claim、URL、原图、relation slot 或唯一证据路径。

## Agent 与 Direct QA 的统一事实口径

Agent 和 Direct QA 都在输出完成后使用同一份 private gold，统一计算
`verdict_matches_gold` 与事实命中（同一完整事实或可决定同一结论的兼容子事实）。
三分类只对 `status=completed` 且 `private_gold_auditable=true` 的记录计算：

1. 找对点且证据充分：`verdict_matches_gold=true` 且 `quality_bucket=strong`；
2. 判断对但证据不足：`verdict_matches_gold=true` 且质量桶不是 `strong`；
3. 判断错：`verdict_matches_gold=false`。

工程错误、judge 失败、private gold 不可审计或 judge 输出缺少质量桶，单独计数，
不塞进以上三类。Agent 的 `strong` 必须由实际 selected Evidence、verdict basis 与终局
报告共同支撑；Direct QA 没有 trace Evidence，其 `strong` 只表示图像直答的答案质量，
不能解释为“找到了外部证据”。完整契约见
`docs/agent-fact-check-report-and-private-gold-audit.md`。

共享实现位于 `src/eval/private_gold_metrics.py`；离线统计入口为
`scripts/summarize_private_gold_audits.py`。

Runtime 中的 `ImageClaim` 保留为 lineage/bookkeeping：

- Task、Evidence、Finding、Discrepancy 的所有权；
- trace 审计与 basis 编译；
- 轨迹导出时的 ID 关联。

但 Claim 不再是 SFT judge 的唯一语义目标。SFT judge 判断的是通用的 `ImageFact`。

## 通用 private target

最终数据集行通过一个 adapter 投影成 `ifv-sft-target-v2`。不为不同生产 route
维护不同 judge；缺失字段保留为空，不作为失败条件。

```json
{
  "schema_version": "ifv-sft-target-v2",
  "case_id": "...",
  "expected_verdict": "real|fake",
  "image_fact": {
    "statement": "...",
    "visible_anchors": ["..."],
    "subject": "...",
    "event_or_context": "...",
    "relation": "...",
    "depicted_value": "..."
  },
  "reference_facts": [
    {
      "statement": "...",
      "verified_value": "...",
      "evidence_text": "...",
      "source_url": "...",
      "role": "source_anchor|image_context|contradiction"
    }
  ]
}
```

`target_claim`/`primary_claim` 是事实描述，不是要求候选复现的字符串。
`decisive_visual_atom`、`visible_scene_facts` 和图片上下文统一进入
`visible_anchors`。`source_exact_spans`、`evidence.binding` 和反驳 chain
统一展开为 `reference_facts`。

## Candidate packet

SFT judge 应看到所有成功调用产生的有效 Evidence，而不只是最终
`verdict_basis` 选择的 Evidence。每条候选 Evidence 保留：

- `evidence_id`、`exact_text`、`source_url`；
- `claim_binding`、`relation_scope`、`relation_stance`、`stance`；
- `successful_call`；
- `basis_selected`；
- 可选的 Claim/Finding/Fact ID。

失败工具调用、Discovery-only 结果和空页面不能作为 Evidence，但可以进入压缩后的
action/audit summary。

## LLM judge 输出

```json
{
  "fact_alignment": "same_image_fact|compatible_subfact|different_fact|unclear",
  "decision_support": "supports_real|supports_fake|supporting_only|insufficient|unclear",
  "decisive_evidence_ids": ["..."],
  "supporting_evidence_ids": ["..."],
  "overclaiming": "none|minor|major",
  "boundary_assessment": "respected|minor_issue|major_issue",
  "confidence": 0.0,
  "explanation": "..."
}
```

`compatible_subfact` 允许候选通过另一条确实能决定同一图像事实的合理路径，
不要求匹配原始 Claim 或 relation slot。

## Eligibility gate

### 致命硬门槛

只有以下情况直接拒绝：

- 最终 verdict 错误；
- 原图缺失、损坏或 hash 不匹配；
- trace 存在不可恢复 engineering/protocol error；
- judge 选择不存在的 Evidence ID；
- 选择的 Evidence 来自失败工具调用；
- Evidence 明确对应另一张图或另一事件；
- `fact_alignment == different_fact`；
- `overclaiming == major`；
- 没有 decisive Evidence；
- source-access policy 违规或存在未恢复的安全边界错误。

### 软指标

以下只记录为 warning/diagnostic，不单独拒绝：

- 没找到精确原图；
- Claim 没进入最终 basis；
- 有效 Evidence 没进入最终 basis；
- `relation_scope` 不是 `same_relation`；
- relation/slot 字段缺失；
- Evidence 由多条记录共同闭环；
- 多搜、重复搜索或 stop 稍晚；
- 可恢复的 schema correction；
- 非致命 strict-audit warning；
- 存在少量无关 Discovery。

最终 eligibility 由程序计算：

```python
eligible = (
    verdict_correct
    and engineering_valid
    and image_available
    and fact_alignment in {"same_image_fact", "compatible_subfact"}
    and decision_support == expected_decision_support
    and bool(decisive_evidence_ids)
    and overclaiming != "major"
    and not fatal_errors
)
```

`unclear` 只保留在 judge artifact 中，不写回 runtime，也不新增
`unverified` 状态。

## 实施顺序

1. 将当前 `sft_eligibility.py` 的 target 投影改为通用 `ImageFact` target。
2. 去除 `decision_paths`、`acceptable_decision_paths` 和人工 review queue。
3. 将 candidate packet 改为包含全部有效 Evidence，并标注 `basis_selected`。
4. 用上述最小 judge schema 替换 Claim/relation-slot 对齐 schema。
5. 将 strict audit 结果拆成 fatal errors 与 warnings。
6. 重写 eligibility 汇总与 artifact，保留 warning 供 teacher 排序。
7. 更新 `score_sft_eligibility.py`，不再输出人工 review 队列。
8. 增加 generated、web、视觉证据、不同合理路径和失败 Evidence 的测试。
9. 在同一批历史 trace 上做旧版/新版离线对比，再进行少量真实 Gemini judge 调用。

## 不改动范围

- 不改 runtime `ImageClaim` 数据结构；
- 不改 ReAct、Decision、Coverage 和 Judgment 主链；
- 不把 private target 注入 rollout；
- 不把 semantic reward 接入 RL scalar reward；
- 不引入人工环境或 `unverified` runtime 状态。

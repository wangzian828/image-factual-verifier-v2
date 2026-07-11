# 供另一个 Agent 学习的阅读与实现指南

本文件是机器友好的学习入口。目标不是逐篇摘要，而是帮助另一个 Agent快速建立正确的问题结构、找到原始来源，并将结论转成实现与实验。

完整机器可读元数据位于 [`../literature/references.json`](../literature/references.json)（含摘要）和 [`../literature/references.csv`](../literature/references.csv)（扁平索引），共 81 条 arXiv 原始记录。

## 0. 使用规则

1. 先读本文件，再读 `search_agent_paradigm_report_zh.md`，需要证据细节时查 `../literature/survey.md`。
2. 引用论文时使用下表中的 arXiv/官方链接，不要从本报告二次转述作者的定量结果。
3. 2026 年来源多为预印本，标为 `frontier signal`；实现前应核对最新版 PDF、代码和后续评审状态。
4. 将论文中的自动指标、作者自报结果、厂商内部评测与独立复现分开。
5. 不要把图片“像素真实”、图片“来源真实”、配文“上下文一致”和整体 claim“有证据支持”混成一个变量。

## 1. 最短阅读路径（15 篇）

### 基础范式

1. [ReAct](https://arxiv.org/abs/2210.03629) — reasoning 与 action 交错。
2. [Self-RAG](https://arxiv.org/abs/2310.11511) — 按需检索与自反思。
3. [STORM](https://arxiv.org/abs/2402.14207) — 多视角研究与先提纲后写作。
4. [MindSearch](https://arxiv.org/abs/2407.20183) — 动态子问题图与并行搜索。
5. [DeepResearcher](https://arxiv.org/abs/2504.03160) — 真实网页环境端到端 RL。

### 多模态事实核查

6. [Open-domain OOC fact-checking / CCN](https://arxiv.org/abs/2112.00061) — 在线文本/视觉证据与跨模态一致性。
7. [VERITE](https://arxiv.org/abs/2304.14133) — unimodal bias 与 modality balancing。
8. [Similarity over Factuality](https://arxiv.org/abs/2407.13488) — shortcut 可伪装成进步。
9. [DEFAME](https://arxiv.org/abs/2412.10510) — 动态工具与多模态证据式核查。
10. [AVerImaTeC](https://arxiv.org/abs/2505.17978) — 真实 image–text claims、网页 QA evidence、充分性检查。

### 可靠性、覆盖与引用

11. [ALCE](https://arxiv.org/abs/2305.14627) — 引用正确性与完整性分解。
12. [BrowseComp-Plus](https://arxiv.org/abs/2508.06600) — 固定语料、gold supporting docs、困难负例。
13. [SeekerGym](https://arxiv.org/abs/2604.17143) — completeness 与漏证不确定性。
14. [Cited but Not Verified](https://arxiv.org/abs/2605.06635) — 实际抓取引用后验证支持度。
15. [REFLECT](https://arxiv.org/abs/2605.19196) — LLM judge 的过程错误检测不可靠。

## 2. 按问题路由阅读

| 如果你要解决 | 必读来源 | 应提取的机制 |
|---|---|---|
| 何时搜、搜什么 | [FLARE](https://arxiv.org/abs/2305.06983), [R²-Searcher](https://arxiv.org/abs/2606.28566) | 不确定性触发、query-token 事实、检索后反思 |
| 覆盖与重规划 | [Argus](https://arxiv.org/abs/2605.16217), [DeepRubric](https://arxiv.org/abs/2606.17029), [ScaffoldAgent](https://arxiv.org/abs/2606.20122) | evidence graph/tree、缺口派发、outline utility |
| 停止与拒答 | [Mind-ParaWorld](https://arxiv.org/abs/2603.04751), [DeepSearchQA](https://arxiv.org/abs/2601.20975), [Don't Stop Early](https://arxiv.org/abs/2604.24978) | sufficiency、早停/过搜、evidence-aware termination |
| 长轨迹信用 | [IGPO](https://arxiv.org/abs/2510.14967), [IGRPO](https://arxiv.org/abs/2607.06223), [TAPO](https://arxiv.org/abs/2606.05784) | 信息增益、预算分配、失败轨迹中正确动作信用 |
| 视觉主动搜索 | [MMSearch](https://arxiv.org/abs/2409.12959), [OpenSearch-VL](https://arxiv.org/abs/2605.05185), [Visual-Seeker](https://arxiv.org/abs/2606.15231) | requery/rerank、crop/OCR/增强、active visual reasoning |
| 反向搜图 | [RIS audit](https://arxiv.org/abs/2603.09130), [AVerImaTeC dual retriever](https://arxiv.org/abs/2602.15190) | RIS 是 noisy sensor；保留 rank/time/query/crop；二次访问验证 |
| 图像生成/篡改 | [Sanity Check](https://arxiv.org/abs/2406.19435), [Deepfake detector role](https://arxiv.org/abs/2602.01854) | detector 泛化差；只能作弱特征，不能成为 claim verdict |
| 引用绑定 | [FActScore](https://arxiv.org/abs/2305.14251), [AAR](https://arxiv.org/abs/2602.13855), [Cited but Not Verified](https://arxiv.org/abs/2605.06635) | 原子 claim、exact span、coverage/soundness/audit effort |
| 评测可复现 | [Deep Research Bench](https://arxiv.org/abs/2506.06287), [DeepResearchGym](https://arxiv.org/abs/2505.19253), [BrowseComp-Plus](https://arxiv.org/abs/2508.06600) | frozen corpus、RetroSearch、模块可分离评测 |
| 搜索污染与攻击 | [AgentDojo](https://arxiv.org/abs/2406.13352), [UGC poisoning](https://arxiv.org/abs/2605.24245), [FORGE](https://arxiv.org/abs/2607.04718) | data/instruction isolation、来源过滤、root-query anchoring |
| 自动 judge | [REFLECT](https://arxiv.org/abs/2605.19196), [Citation verifier calibration](https://arxiv.org/abs/2607.08700) | 人工 gold 标定、方向性 FP/FN、集成与 abstention |

## 3. 实现时的强制 schema

```yaml
case:
  input_image_hash: sha256
  user_claim: string
  decision_policy_version: string

claims:
  - id: C1
    text: string
    type: content_authenticity | source | time | location | identity_non_biometric | event | context | other
    criticality: decisive | supporting
    status: open | supported | refuted | mixed | insufficient

evidence:
  - id: E1
    tool_call_id: T1
    source_id: S1
    url: string|null
    retrieved_at: ISO8601
    published_at: ISO8601|null
    exact_span_or_region: string
    artifact_hash: string
    stance: {C1: support|refute|unclear}
    authority: float
    directness: float
    timeliness: float
    accessible: boolean

sources:
  - id: S1
    canonical_domain: string
    source_family: F1
    primary_or_secondary: primary | secondary | ugc | unknown
    dependency_edges: [S2]

failures:
  - tool_call_id: T2
    type: timeout | quota | blocked | empty | unsupported | partial_parse | invalid_response
    critical: true|false
    recovered_by: T3|null

coverage:
  slots: []
  decisive_slot_recall: float
  independent_source_families: int
  unresolved_conflicts: []
  marginal_information_gain: float
  missing_evidence_risk: float

judgment:
  proposed: real | fake | unverifiable
  gate_passed: boolean
  public_reason_codes: []
  cited_claim_edges: [{claim_id: C1, evidence_ids: [E1]}]
```

## 4. 容易误学的结论

- “多 Agent 更强” → 只对可并行、宽搜索、价值足以覆盖 15× token 的任务成立；不是通用真理。
- “更多工具调用更可靠” → 错；重复和低质量来源会使引用支持度下降。
- “有 citation 就 grounded” → 错；必须区分 link works、relevance、fact support 和 coverage。
- “视觉检测器说 AI-generated，所以 claim 假” → 错；内容生成方式与声明真实性是不同变量。
- “反向搜图无结果，所以图是新/假” → 错；RIS 存在 data void、索引偏差和平台限制。
- “LLM judge 分数高，所以可做 reward” → 错；先用人工 gold 核对 FP/FN drift，尤其是证据验证。
- “规划问题都有答案，所以覆盖完整” → 错；planner 可能漏掉整类问题；答案可能来自同一来源族或只与问题主题相关。
- “unverifiable 是失败” → 错；对开放世界、高风险事实核查，良好校准的拒答是核心能力。

## 5. 第一个可发表实验

**题目草案**：Quality- and Independence-Aware Coverage Constraints for Evidence-Grounded Image Fact-Checking Agents

**基线**：

1. 固定工具流水线；
2. 普通 ReAct；
3. ReAct + question-level coverage；
4. ReAct + claim/evidence/source/failure ledgers + deterministic gate。

**数据**：AVerImaTeC + 自建 rolling claims；为每例加入重复转载、低质量 RIS、冲突来源、工具故障四类干预。

**主指标**：

- selective verdict accuracy；
- decisive claim evidence recall；
- citation entailment precision；
- independent source-family coverage；
- silent false-success rate；
- cost per supported verdict。

**核心预测**：ledger + quality/independence coverage 会增加有理由的 `unverifiable`，但显著提升非拒答样本的准确率、引用支持度和攻击鲁棒性；这比单纯提高总体三分类 accuracy 更有研究价值。

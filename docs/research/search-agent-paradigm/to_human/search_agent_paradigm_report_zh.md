# 当前 Search Agent 范式、关键问题与单图事实核查研究路线

**版本**：2026-07-10

**面向系统**：单张图片、开放域、多阶段、证据驱动的事实核查 Agent

**当前架构**：Perception → Planning → Verification/ReAct → Coverage Audit/Replanning → Judgment

**输出**：`real / fake / unverifiable`

## 执行摘要

你们的方向处在当前前沿的正确主线上：动态多阶段 Agent、开放域多模态检索、原生工具调用、证据溯源、Coverage Audit、失败显式暴露，均比固定工具流水线或只靠模型参数知识更接近可信事实核查。

但当前范式最危险的错觉是：**只要 Agent 搜得更深、规划问题都“有结果”、最后附了引用，就完成了事实核查。** 文献显示这三件事都不成立：

- 更多搜索可能引入重复、低质量和互相抄袭的来源；最新研究甚至观察到工具调用从 2 增加到 150 时引用事实支持准确率明显下降 [Onweller et al., 2026](https://arxiv.org/abs/2605.06635)。
- 形式覆盖会掩盖实质漏证。SeekerGym 的最佳方法仅找回预定义完整语料中 42.5% 的 Wikipedia passages、29.2% 的 ML survey passages [Kim et al., 2026](https://arxiv.org/abs/2604.17143)。
- 链接有效和主题相关并不等于支持句子；前沿 Agent 的引用事实支持仍可只有 39–77% [Onweller et al., 2026](https://arxiv.org/abs/2605.06635)。
- 反向搜图的高排名结果会重复传播误信息，辟谣结果不足 30%，且早期存在 data void [RIS audit, 2026](https://arxiv.org/abs/2603.09130)。
- LLM judge 本身不能可靠发现长轨迹错误：REFLECT 中最佳 judge 总体准确率仍低于 55% [Wang et al., 2026](https://arxiv.org/abs/2605.19196)。

因此，本报告的核心建议是：

> **将系统从“一条会调用工具的 ReAct 轨迹”升级为“声明—证据—来源—失败四本账驱动的受约束状态机”。规划与搜索只能提出候选；只有经过真实性、时效性、相关性、立场、独立性和可访问性检查的证据，才能改变 verdict。**

最值得优先解决的五个研究问题是：

1. **可验证的证据充分性与停止条件**：何时真的搜够了，何时应继续，何时必须 `unverifiable`？
2. **claim-level provenance 与证据绑定**：每个决定性原子声明究竟由哪个页面的哪一段、哪一图支持？
3. **来源独立性、冲突和时间建模**：十个互相转载的网页不应算十份证据；旧来源不能覆盖新事件。
4. **多模态原生的主动视觉检索**：crop/OCR/局部反搜/几何或时间地点线索应由信息增益驱动，而非固定流水线。
5. **工具失败与开放世界校准**：零结果、网页打不开、RIS 受限、视觉信号矛盾时，如何把“未知”保真地传到最终判断？

## 1. Search Agent 现在有哪些范式

### 范式 A：Agentic RAG / 自适应检索

代表工作包括 FLARE、Self-RAG、CRAG、Adaptive-RAG [Jiang et al., 2023](https://arxiv.org/abs/2305.06983), [Asai et al., 2023](https://arxiv.org/abs/2310.11511), [Yan et al., 2024](https://arxiv.org/abs/2401.15884), [Jeong et al., 2024](https://arxiv.org/abs/2403.14403)。核心是模型按需决定检索、评估检索质量并纠错。它们适合局部 QA，但通常没有覆盖整个开放问题空间的显式结构。

### 范式 B：ReAct / 单轨长时程搜索

ReAct 将 reasoning 与 tool action 交错 [Yao et al., 2022](https://arxiv.org/abs/2210.03629)；Search-o1、Search-R1、R1-Searcher、ReSearch、DeepResearcher 等进一步用 SFT/RL 学习搜索策略 [Li et al., 2025](https://arxiv.org/abs/2501.05366), [Jin et al., 2025](https://arxiv.org/abs/2503.09516), [Song et al., 2025](https://arxiv.org/abs/2503.05592), [Chen et al., 2025](https://arxiv.org/abs/2503.19470), [Zheng et al., 2025](https://arxiv.org/abs/2504.03160)。

优势是灵活、模型原生、可端到端优化；缺点是单轨 path dependence、错误累积、上下文膨胀和稀疏终局奖励。

### 范式 C：图规划 / 动态提纲 / 证据树

MindSearch 将子问题建成动态图 [Chen et al., 2024](https://arxiv.org/abs/2407.20183)；STORM 先多视角研究再构建提纲 [Shao et al., 2024](https://arxiv.org/abs/2402.14207)；Argus、DeepRubric、ScaffoldAgent、VeriTrace 等 2026 工作则把 evidence graph/tree、动态 outline 和认知图作为核心中间表示 [Argus](https://arxiv.org/abs/2605.16217), [DeepRubric](https://arxiv.org/abs/2606.17029), [ScaffoldAgent](https://arxiv.org/abs/2606.20122), [VeriTrace](https://arxiv.org/abs/2605.26081)。

这是与你们 Coverage Audit 最相近、也最值得深化的范式。

### 范式 D：Orchestrator–worker 多 Agent

规划 Agent 将可独立子问题分给并行 Searcher，再汇总与补搜。Anthropic 的生产系统公开报告内部评测相对单 Agent 提升 90.2%，但 token 使用约为普通 chat 的 15 倍，且复杂依赖任务不一定适用 [Anthropic, 2025](https://www.anthropic.com/engineering/multi-agent-research-system)。

多 Agent 本质上是 test-time compute scaling 与多个独立上下文窗口，不自动解决证据正确性。它只有在子任务真正独立、汇总有共享证据图、重复来源能去重时才值得使用。

### 范式 E：多模态原生 Search Agent

MMSearch 将多模态 search 拆为 query reformulation、rerank、summarization 和 end-to-end [Yu et al., 2024](https://arxiv.org/abs/2409.12959)。2026 前沿进一步把 OCR、crop、锐化、超分、透视校正、图片搜索与文本搜索统一成工具空间，并通过 process reward 或 tool-aware RL 训练主动感知和搜索 [OpenSearch-VL](https://arxiv.org/abs/2605.05185), [ProMMSearchAgent](https://arxiv.org/abs/2604.20486), [Visual-Seeker](https://arxiv.org/abs/2606.15231), [SearchEyes](https://arxiv.org/abs/2607.05943), [TAPO](https://arxiv.org/abs/2606.05784)。

这说明你们未来不应只做“整图描述 → 文本搜索”，而要支持根据当前缺口主动选择局部视觉操作。

## 2. 现在 Search Agent 的主要问题，以及通常怎么解决

| 问题 | 典型失效 | 当前解法 | 仍未解决 |
|---|---|---|---|
| 任务分解不完整 | 开始就漏掉关键解释；后续只在错误框架内深挖 | 多视角提问、动态 DAG/outline、evidence tree | planner 与 auditor 常由同一模型实现，共享盲点；缺口缺少客观定义 |
| 搜索—推理边界错位 | 该搜时凭记忆答；该推理时继续堆网页 | 按不确定性检索、post-retrieval reflection、R² boundary calibration | 模型置信度不校准；“不知道自己不知道” |
| 早期噪声造成 tunnel vision | 第一批错误网页决定后续所有 query | 多分支搜索、信息增益、反事实分支、root-query anchoring | 分支爆炸；多个分支仍可能来自同一信息生态 |
| 检索结果相关但不支持 | 标题/snippet 相关，正文不蕴含 claim | 页面访问、span extraction、NLI/LLM entailment、citation agent | 动态页面、付费墙、图片内证据；judge 偏差 |
| 形式覆盖不等于实质覆盖 | 每个问题各有一个 URL，但关键 claim 没有充分证据 | evidence slots、rubric leaves、coverage-driven objectives | 如何在开放世界估计未见证据与停止风险 |
| 来源重复/依赖 | 十个媒体转载同一条未经证实 post | canonical URL、内容 hash、引用图、来源族聚类 | 转载/共同上游常不可见；来源独立性很少进入 reward |
| 冲突证据处理差 | 选择符合先入假设的网页；隐去反证 | support/refute/unclear stance、contradiction ledger、多 Agent debate | 证据质量与冲突强度的联合校准仍弱 |
| 时间错误 | 用事件之后的辟谣/网页泄漏评测；把旧图片当新图 | evidence cutoff、时间线、动态或冻结 benchmark | 网页发布时间/更新时间不可靠；搜索引擎索引时序不可见 |
| 引用装饰化 | 链接能打开、主题相关，但不支持句子 | claim atomicization、citation entailment/coverage、原文 span | 自动 judge 不稳定；“一个引用支持半句话”的组合错误 |
| 上下文膨胀/遗忘 | 长轨迹丢约束、重复搜索、结论漂移 | 结构化 memory、evidence graph、局部上下文、压缩 | 摘要压缩会丢否定、限定词与 provenance |
| 稀疏奖励与错误信用 | 失败轨迹中的正确工具调用也被惩罚 | process reward、information gain、TAPO、hop anchors | 过程 proxy 可被 reward hacking；跨工具比较困难 |
| 工具失败被语言掩盖 | API 报错后模型仍给肯定结论 | typed error、fatal-aware training、no-fallback policy | 部分失败与 silent degradation 难检出 |
| 成本/延迟不可控 | 为边际信息反复搜索，Agent swarm 重复劳动 | 并行独立 slots、utility/cost reward、预算式 stopping | 准确率 benchmark 很少报告成本、失败成本和 tail latency |
| 网页攻击/污染 | 页面 prompt injection 或假信息改变规划 | 内容/指令通道隔离、来源过滤、root query anchoring、sandbox | 规划层 poisoning、UGC 集中攻击、跨页协同攻击仍有效 |
| 评测失真 | live web 漂移、benchmark answer 可被搜到 | frozen corpus、RetroSearch、ParaWorld、私有 rolling set | sandbox 真实性与 live web 可复现性之间仍有张力 |

## 3. 你们的单图核查任务需要重新定义

`real / fake / unverifiable` 可以保留为用户界面标签，但内部不能只有一个真假变量。至少要维护以下隐变量：

```text
content_authenticity  ∈ {camera-origin plausible, AI-generated, manipulated, unknown}
source_provenance     ∈ {identified, partially identified, conflicting, unknown}
context_consistency   ∈ {consistent, miscaptioned, conflicting, unknown}
claim_support         ∈ {supported, refuted, mixed, insufficient}
evidence_coverage     ∈ [0,1] + calibrated uncertainty
tool_integrity        ∈ {healthy, degraded, failed}
```

然后通过公开、版本化的 decision policy 映射到三分类。例如：

- `fake`：至少一个**决定性 claim** 被高质量、可追溯证据反驳；或可靠 provenance 证明图像来自不同事件/时间/地点；或有充分文件级取证证据证明生成/篡改且该事实直接违反 claim。
- `real`：所有决定性 claim 均有充分支持，未解决反证风险低，来源/时间线合理；这里的 `real` 应命名为“claim supported”，避免被理解为“像素绝对原生”。
- `unverifiable`：存在决定性证据缺口、同等级冲突、工具关键失败、开放世界覆盖不够或只能得到弱/依赖来源。

这能避免典型错误：**一张真实旧照片配上错误新标题，像素是真，claim 是假；一张 AI 图配上“这是艺术家概念图”的声明，像素生成但 claim 可为真。**

## 4. 推荐的目标架构：Evidence-Constrained Multimodal Search Agent

### 4.1 四本账

**A. Claim Ledger（声明账）**

每个原子 claim 包含：

```json
{
  "claim_id": "C3",
  "text": "图片拍摄于 2026 年 7 月的事件 X",
  "type": "time_event_alignment",
  "criticality": "decisive",
  "visual_dependencies": ["V2", "V5"],
  "status": "open | supported | refuted | mixed | insufficient"
}
```

**B. Evidence Ledger（证据账）**

每条证据必须是实际工具结果，不是模型记忆：

```json
{
  "evidence_id": "E17",
  "tool_call_id": "T44",
  "url": "...",
  "retrieved_at": "...",
  "published_at": "...",
  "content_hash": "...",
  "span_or_region": "正文第 3 段 / 图 2 / OCR bbox",
  "stance_to_claim": {"C3": "refute"},
  "quality": {"authority": 0.8, "directness": 1.0, "timeliness": 0.9},
  "source_family": "S4",
  "accessible": true
}
```

**C. Source Ledger（来源账）**

追踪 canonical domain、作者/机构、一次/二次来源、互相转载或共同上游、内容 hash、RIS rank 与查询变体。Coverage 计数应按独立来源族，而不是 URL 数量。

**D. Failure Ledger（失败账）**

对 timeout、quota、empty result、blocked page、unsupported format、parser partial、RIS unavailable 等设 typed errors。任何关键证据槽依赖失败工具时，必须阻断 `real/fake` 的确定性输出，除非有经规则证明的等价替代证据。

### 4.2 阶段设计

1. **Perception / Claim induction**
   - OCR、对象/标志/场景、可见文字、图像边界和显著区域。
   - 生成事实核查 claim，而不是自由描述。
   - 明确“可见事实”“模型推断”“需要外部核验”三层。

2. **Evidence-slot planning**
   - 将每个决定性 claim 转为必要证据槽：来源、事件、时间、地点、主体、文件真实性、传播上下文。
   - 记录依赖和成功条件；不要只保存自然语言问题列表。

3. **Active multimodal acquisition**
   - 工具策略：整图 RIS → 关键 crop RIS → OCR query → 组合实体/时间 query → 页面内图文匹配。
   - 每一步按预期信息增益、成本、失败率和当前 claim criticality 选择。

4. **Evidence verification**
   - 页面实际访问；定位 exact span/region。
   - 评估 relevance、stance、directness、authority、timeliness、independence。
   - snippet、模型摘要、搜索排名只能是发现线索，不能成为终局证据。

5. **Coverage / contradiction audit**
   - 检查每个 decisive slot 是否满足质量门槛。
   - 主动搜索反证与替代解释；检查来源族去重和时间线。
   - 独立 auditor 只读结构化账本，不读取 planner 的隐藏 CoT，以减少共享锚定。

6. **Budgeted replanning**
   - 仅针对 highest expected decision value 的缺口再规划。
   - 若连续查询只返回同一来源族或边际信息增益低，则停止扩张；转入 `unverifiable` 风险评估。

7. **Deterministic judgment gate**
   - LLM 提议 verdict；规则引擎根据 ledger 门槛批准/驳回。
   - 最终自然语言只能引用已批准的 claim–evidence edges。

### 4.3 证据充分性不应是单一布尔覆盖

对决定性 claim `c`，建议定义：

```text
support(c) = Σ over independent source families f:
             max_e∈f [stance(e,c) × authority(e) × directness(e)
                      × timeliness(e) × accessibility(e)]

contradiction(c) = 同理计算 refute evidence
```

然后 Coverage Audit 至少检查：

- 必要 evidence slot recall；
- 每个 slot 的质量下界；
- 独立来源族数而非 URL 数；
- support/refute margin 与同级冲突；
- 工具完整性；
- 最近若干查询的边际新证据率；
- 对漏证概率的校准估计。

这不是最终数学公式，而是需要通过数据学习和校准的设计框架。关键是防止“一个低质量 URL 填满一个格子”的 reward hacking。

## 5. 最应该做的研究课题（按优先级）

### P0：可验证的 Coverage + Stop + Abstain

**问题**：Coverage Audit 如何既避免早停，也避免无限搜索和低置信广撒网？

**建议方法**：

- Evidence-slot DAG + source-family-aware coverage；
- 边际信息增益与 decision value 驱动的下一步工具选择；
- conformal/risk-coverage calibration，为 `real/fake` 设置最大可接受错误风险；
- 明确 `tool-degraded unverifiable`、`evidence-insufficient unverifiable`、`conflict-unresolved unverifiable` 子类型。

**可检验假设**：在相同工具预算下，质量/独立性感知的 coverage 相比“问题是否有答案”覆盖，能显著降低 unsupported confident verdict，同时保持或提高 macro-F1。

### P0：Claim–Evidence Provenance Compiler

**问题**：如何保证每个最终句子只由真实工具结果生成？

**建议方法**：

- 生成前先形成 typed claims；
- URL + 页面 hash + exact span/region + tool_call_id；
- synthesis 使用白名单 evidence IDs；
- 编译后做 citation entailment、coverage、时间和可访问性检查；未通过则删除/降级 claim，而不是自由改写。

**可检验假设**：约束式 evidence compiler 会减少长报告的事实支持错误，即便表面流畅度略降。

### P0：来源独立性与证据冲突图

**问题**：多个网页究竟是多份证据还是一次谣言的多次复制？

**建议方法**：内容 hash/近重复、引用链接、作者/机构、发布时间、共同图片、相同引语和首发线索构建 provenance graph；按 connected source family 折扣。对 support/refute 建显式 conflict set。

**可检验假设**：source-family discount 能显著降低 UGC poisoning 和“共识幻觉”，代价是更多 `unverifiable`，但 selective accuracy 上升。

### P1：主动视觉信息增益策略

**问题**：何时整图反搜，何时 crop、OCR、局部增强或文本搜索？

**建议方法**：将工具调用视为 belief-state action，奖励为对关键 claim entropy 的降低，加入成本和失败惩罚。训练时使用缓存 search world 和可控 counterfactual，评测时迁移到 live APIs。

**可检验假设**：视觉区域—claim 对齐的 tool policy 比固定 `整图RIS→OCR→Web` 流水线以更少调用获得更高 decisive evidence recall。

### P1：失败保真（Failure-preserving Reasoning）

**问题**：如何保证工具失败不会在摘要/重规划中被“修复”为成功？

**建议方法**：typed tool observations；fatal/partial failure propagation；planner 与 judge 均不能修改 tool status；用故障注入训练和测试。

**可检验假设**：失败账 + deterministic gate 能把 silent false success 接近降为零，而不是只提高平均答案准确率。

### P1：时间与开放世界评测

**问题**：如何防止训练泄漏、事件后证据污染和 live web 漂移？

**建议方法**：

- rolling private claims；
- evidence publication cutoff；
- frozen snapshot + live web 双轨；
- synthetic/parallel-world causal cases；
- 每例保存 SERP、网页和图像快照 hash。

### P2：安全与对抗检索

**问题**：网页内容如何劫持计划、引用或推荐？

**建议方法**：指令/数据通道隔离、网页永不具备系统指令权限、root-query anchoring、来源信誉与 UGC 限额、攻击文档 canary、跨 Agent 最小权限和 query privacy lint。

## 6. 推荐评测体系

### 6.1 数据集组合

- **核心真实 image–text claim**：AVerImaTeC [Cao et al., 2025](https://arxiv.org/abs/2505.17978)；其 shared-task 指标将 verdict 与 evidence 质量绑定 [Cao et al., 2026](https://arxiv.org/abs/2602.11221)。
- **OOC 与 shortcut 检验**：NewsCLIPpings、VERITE [Luo et al., 2021](https://arxiv.org/abs/2104.05893), [Papadopoulos et al., 2023](https://arxiv.org/abs/2304.14133)。
- **近期、真实传播与开放网页**：ClaimReview2024+ / DEFAME、RW-Post、X-POSE [Braun et al., 2024](https://arxiv.org/abs/2412.10510), [Xu et al., 2026](https://arxiv.org/abs/2605.10357), [X-POSE, 2026](https://arxiv.org/abs/2606.31367)。
- **多模态搜索能力**：MMSearch [Yu et al., 2024](https://arxiv.org/abs/2409.12959)。
- **自建 rolling set**：最新谣言、后续澄清、RIS data void、跨语言来源、页面删除和来源冲突。

### 6.2 不要只报一个 accuracy

**Outcome**

- macro-F1 / class-wise precision-recall；
- selective accuracy、coverage–risk curve；
- `unverifiable` 的原因准确率；
- severe false-real / false-fake rate。

**Evidence**

- decisive claim evidence recall；
- citation entailment precision；
- provenance coverage；
- exact-span availability；
- source-family diversity；
- conflict recall 与 contradiction transparency；
- temporal validity。

**Process**

- plan slot recall；
- first harmful error position；
- redundant query/tool-call ratio；
- marginal novel evidence per call；
- premature stop / over-search rate；
- failure propagation integrity。

**Operational**

- 中位与 P95 latency；
- cost per correct-and-supported verdict；
- tool error rate；
- reproducibility across engines/dates；
- prompt-injection / poisoning attack success。

### 6.3 必做消融

1. 固定流水线 vs 动态 ReAct vs evidence-ledger Agent。
2. 有/无 Coverage Auditor。
3. URL 数覆盖 vs source-family quality coverage。
4. 同模型 self-audit vs 独立 auditor vs deterministic gate。
5. 整图 RIS vs 主动 crop/OCR/RIS policy。
6. 正常工具 vs 故障注入。
7. 干净网页 vs UGC poisoning / prompt injection。
8. frozen web vs live web。
9. 像素检测器作为硬 verdict、弱 feature、完全移除。
10. 单 Agent vs 并行 Agent（同时报告 token、重复率和 tail latency）。

## 7. 90 天落地路线

### 第 1–3 周：建立可审计基线

- 实现四本账与统一 schema；所有 Gemini Interactions API function call 生成不可变 `tool_call_id`。
- 禁止 snippet 成为 final evidence；保存 URL、时间、hash、span/region。
- 将 `real/fake/unverifiable` 的 gate 规则版本化。
- 在 AVerImaTeC 上复现简单“文本 retriever + RIS + 单次 MLLM”基线，建立成本/准确率下界。

### 第 4–6 周：Coverage 与停止

- 将 planning questions 改为 typed evidence slots 和依赖 DAG。
- 实现来源族去重、support/refute/unclear 与冲突账。
- 建立故障注入集与早停/过搜诊断。
- 对 `unverifiable` 做子类型校准。

### 第 7–9 周：主动多模态搜索

- 加入区域建议、crop RIS、OCR query、局部增强；先用规则/上下文 bandit，再决定是否 RL。
- 每步记录 belief/coverage delta，用离线轨迹评估 counterfactual tool value。
- 控制实验比较固定流水线与 information-gain policy。

### 第 10–12 周：安全、时间与发布

- 建 frozen snapshot + live twin evaluation。
- 注入 UGC poisoning、页面 prompt injection、重复来源、冲突来源和关键工具失败。
- 人工复核一组 claim–evidence edges，标定自动 judge 的 FP/FN；不要直接用未标定 judge 做 RL reward。
- 发布 failure cards：哪些图片类型、语言、RIS 状态、时间窗口和来源生态仍不可处理。

## 8. 设计原则清单

保留并强化：

- 原生函数调用与真实工具结果绑定；
- 动态规划、Coverage Audit、显式重规划；
- 无人脸识别；
- 不使用固定工具流水线；
- 不允许伪成功 fallback；
- `unverifiable` 为正式结果。

新增硬约束：

- 模型记忆永远不能作为外部事实证据；
- 搜索 snippet 永远不能直接支持 verdict；
- 单一 URL 不能自动满足决定性 claim；
- 来源独立性按 provenance family 计算；
- 未解决的同等级冲突强制拒答；
- 关键工具失败必须进入最终可见理由；
- 像素检测器/C2PA/RIS 均为带可靠度的传感器，缺失不等于反证；
- planner 不得自证 coverage；
- final writer 只能使用 ledger 中批准的 evidence edge；
- 所有 judge 在进入 reward 或 gate 前必须做人工标定。

## 9. 结论

当前 Search Agent 的竞争已从“有没有搜索工具”转移到四件事：

1. **是否能把开放问题转成可核验的信息需求结构；**
2. **是否能在噪声、冲突、动态和对抗网页中组装独立证据；**
3. **是否知道证据何时足够、何时不足；**
4. **是否能证明最终每个决定性判断确实来自真实工具结果。**

你们现有的 Perception → Planning → Verification/ReAct → Coverage Audit/Replanning → Judgment 已经有正确骨架。真正有研究价值的下一步，不是继续增加 Agent 阶段或工具数量，而是让阶段间交换**机器可验证的证据状态**，并用 provenance-aware coverage、冲突透明、失败保真和风险校准把 `real/fake/unverifiable` 变成可审计决策。

完整的论文逐条说明与更多来源见 [literature/survey.md](../literature/survey.md)。机器友好的学习入口见 [agent_learning_guide.md](agent_learning_guide.md)。

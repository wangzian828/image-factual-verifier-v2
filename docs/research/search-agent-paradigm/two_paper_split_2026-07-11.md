# 两篇论文的依赖式拆分建议

日期：2026-07-11

## 总原则

推荐按以下逻辑拆分：

```text
Paper 1：定义能力，并证明现有系统缺少这种能力
Paper 2：使用 Paper 1 的任务和指标，训练出这种能力
```

用户提出的“第一篇任务命题 + benchmark，第二篇训练策略 + Agent”是正确主线。关键是进一步把两种数据分开：

```text
Paper 1：真实评测数据（evaluation data）
Paper 2：Internet-grounded 合成训练数据（learning data）
```

不要让 Paper 1 同时承担大规模合成训练世界和 RL；否则 Paper 1 会变成任务、benchmark、生成管线、Agent、RL 五项贡献的混合体，Paper 2 也失去独立问题。

## Paper 1：单图发起的开放互联网调查任务、半自动构造引擎与参考 Agent

### 核心研究问题

> 在没有外部文字 claim、但可以访问开放互联网的条件下，Agent 能否从单张图片主动提出值得核查的事实目标，利用 RIS/Web 发现候选语境，让外部证据反过来指导对输入图片的下一次观察，并最终形成有来源、覆盖充分的事实档案或裁决？

这里的 `single image` 只是查询入口，不是封闭环境：

```text
Input：single image, no externally supplied claim
Environment：open Web + RIS + browsing + OCR/crop/compare
Output：Internet-grounded verified image account
```

因此第一篇的互联网成分不是一个附加工具，而是任务成立的必要条件、数据 gold 的来源和主要评测对象。

### 建议任务名

```text
Evidence-Conditioned Active Visual Investigation
```

任务不是普通 image fact-checking，也不是单次 RIS。任务状态变化是：

```text
initial visual belief
→ search/RIS observation
→ targeted visual reinspection
→ hypothesis update
→ evidence-grounded verdict or abstention
```

### Paper 1 必须贡献的内容

1. **任务定义**
   - 输入的 claim 语义；
   - `supported / refuted / insufficient` 与产品 `real/fake/unverifiable` 的映射；
   - 什么构成有效的 search-conditioned reinspection；
   - RIS 作为 visual-neighbor discovery，而非仅找原图；
   - 工具失败与无结果的语义。

2. **真实 Benchmark**
   - 未修改的真实 query images；
   - 真实/冻结网页及 RIS top-k observations；
   - claim atoms、evidence spans、source families；
   - image-region / visual-clue 标注；
   - 最小充分证据集和正确停止状态；
   - `R0–R5` 视觉邻居关系；
   - frozen snapshot + 小规模 hidden/live refresh。

3. **新评测协议**
   - verdict correctness；
   - citation support/sufficiency；
   - useful visual neighbor recall；
   - search 后正确重看区域；
   - hypothesis update correctness；
   - false visual bridge；
   - premature stop / over-search；
   - no-result/failure calibration。

4. **强基线与诊断**
   - closed-book VLM；
   - text/Web search only；
   - RIS only；
   - 一次性 perception + search；
   - ordinary ReAct；
   - prompt/scaffold 实现的 reinspection Agent；
   - oracle RIS/oracle region 上界。

5. **半自动 Benchmark 构造引擎**
   - 从事实核查文章、公开 image–text claims 或许可事件页面发现候选 case；
   - 抽取 claim atoms、页面 evidence spans 和 source families；
   - 实际执行 RIS，保存 top-k，并标注 R0–R5 候选关系；
   - 让模型提出“该搜索结果应触发什么新视觉问题”和候选区域；
   - 自动做近重复、事件、时间、来源族和答案泄漏检查；
   - 低置信或冲突样本交由人工确认。

   这里的自动化目标是**降低构建真实评测数据的人工成本**，不是在 Paper 1 中生成大规模训练世界。论文应报告候选接受率、自动标注准确率/召回率、人工修改率、单例时间和成本。

6. **一个简洁但完整的参考 Agent**
   - 建议称为 `Evidence-to-Vision Agent` 或 `ReInspect`；
   - 不要求 RL，可以是结构化 prompting/scaffold 或轻量 action ranker；
   - 显式维护 hypotheses、evidence ledger、visual questions 和 target regions；
   - 搜索结果必须先转成新的视觉检查任务，才允许影响视觉 belief；
   - 使用真实 RIS/Web/OCR/crop 工具；
   - 最终裁决受 evidence coverage 与 failure state 约束。

   它的论文角色是证明任务可解、提供可复现强基线，并验证过程指标有意义；不是声称已经解决最优训练问题。

### Paper 1 不应承担

- 大规模 Internet-grounded 合成**训练集**；
- 长时程 RL；
- 完整候选动作树；
- 宣称提出最优 Agent；
- 每个 case 的绝对首发原图；
- 大规模持续 live benchmark 运营。

Paper 1 可以使用 Gemini Agent 作为执行底座，但需要提出一个可描述、可消融的 `search → visual question → region action → belief update` 控制机制，不能只是“换了一个长 prompt”。其创新应位于结构化状态转移和证据门控，而非模型训练。

### Paper 1 的推荐统一故事

```text
Task：现有 benchmark 没有测量搜索如何改变后续视觉观察
Builder：真实数据很贵，因此提出 human-in-the-loop 半自动构造引擎
Benchmark：发布真实主轨道 + 受控诊断轨道
Method：提出无需 RL 的 ReInspect 参考 Agent
Finding：显式 evidence-to-vision 转换有效，但策略选择和停止仍是瓶颈
```

这里可以有一个小型**受控合成诊断子集**，用于验证区域、反事实和 stopping 指标。例如 50–100 个 internet-grounded worlds、每个 2–4 个视觉变体。但它在 Paper 1 中只承担 benchmark stress test，不用于大规模训练。Paper 2 才把同一构造思想扩展成 learning environment、候选动作分支和过程奖励。

### 半自动真实数据构造引擎的建议流程

```text
seed source / fact-check page
→ 提取 query image 与原始配文
→ 原子化 claim
→ 抓取并冻结支持/反驳网页
→ 聚类 source family
→ 实际执行 RIS 并保存 top-k
→ 提议 R0–R5 关系和证据页面
→ 提议 search-conditioned visual question
→ grounding 到 query image region
→ 自动一致性/泄漏检查
→ 人工接受、修改或拒绝
```

自动化不能直接宣布 gold。建议三档：

```text
auto-accept：多个检查器一致且通过硬规则
human-review：低置信、冲突或关键样本
reject：claim 不可恢复、证据不足、许可不明或无检索钩子
```

论文需要用 50–100 例双人 gold audit 评估 builder 本身，并报告：

- claim atom precision/recall；
- evidence entailment correctness；
- R0–R5 relation accuracy；
- target-region IoU/pointing accuracy；
- source-family purity；
- case acceptance yield；
- 人工分钟/case 相对纯人工下降比例。

### 参考 Agent：ReInspect

最小状态：

```text
Hypothesis Ledger
Evidence Ledger
Visual Question Queue
Region Observation Ledger
Coverage / Failure State
```

核心转移：

```text
web/RIS observation
→ evidence extractor
→ visual question generator
→ region selector + crop/OCR/compare
→ verifier
→ hypothesis update
→ coverage/stopping gate
```

最关键的结构约束是：外部搜索结果不能直接改写最终 belief；它必须明确指出 `(target hypothesis, discriminative visual property, target region/action)`，再由真实视觉工具 observation 完成更新。这样才能把“一边搜、一边看”变成可审计机制。

建议消融：

```text
普通 ReAct
+ structured ledgers
+ evidence-to-visual-question bridge
+ targeted reinspection
+ coverage/stopping gate
```

这足以构成第一篇的方法部分，同时把“如何通过大规模 SFT/RL 学到动作策略”留给第二篇。

### Benchmark 的范围控制

建议首版：

```text
主测试：200–300 个真实 cases
  以 Semantic/RIS investigation 为主
  不要求每例找到原图

Provenance 子集：30–60 cases
  标注原始或最早可确认来源

Hidden rolling：30–50 cases
  用于补充开放 Web 结论，可延后或只做一轮
```

如果资源不足，先做 150–200 个高质量 cases，也好于 1,000 个只有图片和标签的 cases。

### Paper 1 的中心发现应该是什么

预期不是“我们的基线最好”，而是系统性揭示：

- 现有 Agent 能搜到相关页面，却不会把结果转成新的视觉检查；
- 视觉相似被误当成事件相同，出现 false bridge；
- RIS 无结果被错误当作反证；
- 同图命中后不访问页面便直接下结论；
- evidence coverage 看似完整，但决定性视觉 claim 未被验证；
- 搜索更多不等于裁决更可靠。

这会给 Paper 2 建立清晰的优化目标。

## Paper 2：Internet-grounded 训练与主动调查 Agent

### 核心研究问题

> 如何利用互联网事实锚定的可控视觉世界，训练 Agent 在有限预算下选择最具裁决价值的搜索和重新观察动作，并迁移到未见真实图片与开放 Web？

### 建议方法贡献

1. **Internet-grounded synthetic visual world generator**
   - 真实网页定义事实和来源图；
   - 生成模型渲染支持、单事实反例和证据不足视觉观测；
   - region ↔ visual atom ↔ claim atom ↔ evidence span ↔ source family；
   - 真实运行 RIS 后将样本分为 RIS-actionable/text-search-only/data-void。

2. **显式 investigation belief state**
   - 候选假设；
   - claim coverage；
   - 视觉区域和待验证线索；
   - 来源独立性与冲突；
   - 工具健康和剩余预算。

3. **搜索—重观察策略**
   - 高层选择调查目标、工具类型、探索/验证/反证、继续/停止；
   - 低层执行 query/crop/OCR；
   - 可先 action ranking/SFT，再判断 RL 是否必要。

4. **证据与覆盖约束**
   - 决定性 judgment gate；
   - 工具结果真实绑定；
   - failure propagation；
   - source-family-aware evidence；
   - calibrated abstention。

### Paper 2 的关键实验

统一在 Paper 1 的隐藏真实 benchmark 上测：

```text
A  prompted/scaffold baseline
B  ordinary trajectory SFT
C  internet-grounded counterfactual SFT / action ranking
D  C + hierarchical RL（如果 RL 确有增益）
```

核心消融：

- 普通文生图 vs internet-grounded 合成；
- 无反事实配对 vs 单事实反事实；
- 无 region–claim binding vs 有绑定；
- text-only search vs RIS + Web；
- 无显式 belief update vs 显式更新；
- terminal reward vs process/coverage reward；
- 无 stopping objective vs 有 stopping objective；
- synthetic-only vs synthetic + 少量真实适配。

### Paper 2 必须证明的结论

不能只在合成环境高分。必须证明：

- 对 Paper 1 未见真实图片有迁移增益；
- 搜索后正确重看区域的比例上升；
- 决定性证据发现而非工具调用次数上升；
- false bridge 和过早停止下降；
- 在相同预算下优于 prompted/scaffold 与普通 SFT；
- 如果使用 RL，必须优于动作排序/SFT，否则不宣称 RL 必要。

## 两篇论文的接口契约

Paper 1 应冻结并发布以下接口，Paper 2 不改变定义：

```text
case schema
claim/evidence/source schema
RIS relation taxonomy
region/reinspection annotation
tool trace schema
evaluation metrics
frozen test version
```

Paper 2 可以扩充训练字段，但不能用训练时 privileged gold 作为测试输入。

数据隔离：

```text
Paper 1 public train/dev cases
  可供 Paper 2 调试，但不作为最终 test

Paper 1 hidden test cases
  不进入 world generation、prompt设计或训练

Paper 2 synthetic worlds
  按 event/entity/source/template/generator 分组切分
```

## 为什么这一拆分独立成立

| 问题 | Paper 1 | Paper 2 |
|---|---|---|
| 科学问题 | 如何定义和测量该能力？ | 如何学习该能力？ |
| 数据 | 少而精的真实 evaluation data | 大规模可控 learning data |
| 主要贡献 | task + benchmark + diagnosis | training environment + policy/Agent |
| 方法要求 | 只需强而透明的 baselines | 需要 SFT/ranking/RL 和系统方法 |
| 成功条件 | 揭示稳定能力缺口 | 在隐藏真实 benchmark 上补上缺口 |

即使 Paper 2 尚未完成，Paper 1 也可因新任务、真实 benchmark、指标和系统诊断独立成立；即使 Paper 1 已发表，Paper 2 也可因新训练数据机制和真实迁移提升独立成立。

## 风险与建议

### 风险 1：Paper 1 被认为只是新数据集

解决：任务必须包含普通 image fact-checking 没有的交互状态和过程标注，尤其是 `search result → new visual question → target region → belief update`，并证明现有指标会掩盖 false bridge 和无效搜索。

### 风险 2：Paper 1 benchmark 太贵

解决：不要求每例绝对原图；主做 Semantic/RIS，provenance 只做小子集；优先从现有真实 image–text claims 和事实核查案例适配。

### 风险 3：Paper 2 与已有 Search Agent 相似

解决：核心不能只是“加 RL”，而是 internet-grounded visual counterfactual supervision + search-conditioned reinspection policy + real-image transfer。

### 风险 4：Paper 2 使用自己的 benchmark 造成封闭自证

解决：Paper 1 的真实 hidden test 在 Paper 2 训练前冻结；增加公开外部数据集和小规模 rolling live test；报告跨后端、跨时间结果。

## 建议项目顺序

```text
Step 1  冻结 Paper 1 的任务定义与 schema
Step 2  做 30–50 个真实 pilot cases，验证标注可行性
Step 3  跑 5–8 类 baseline，确认存在稳定能力缺口
Step 4  扩到 Paper 1 benchmark 并投稿
Step 5  同时用已稳定 schema 做 10–50 world 合成训练 spike
Step 6  Paper 1 test 冻结后开始 Paper 2 大规模训练
Step 7  先做 SFT/action ranking，再决定是否投入 RL
```

最终一句话：

> 第一篇回答“我们究竟要测什么，以及现有 Agent 为什么不会”；第二篇回答“怎样用可控、互联网事实锚定的训练信号，让 Agent 真正学会”。

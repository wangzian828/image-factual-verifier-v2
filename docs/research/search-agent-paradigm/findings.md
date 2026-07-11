# Research Findings
## Research Question

当前 Search Agent，尤其是开放域多模态、证据驱动的单张图片事实核查 Agent，最重要的系统性问题是什么，已有前沿工作如何处理，下一步应该优先解决什么？

## Current Understanding

Search Agent 已经历五次明显转向：浏览器增强问答与引用、ReAct 式检索—推理交错、自适应/纠错 RAG、图规划或多 Agent Deep Research、端到端 RL 与过程奖励。2026 年前沿开始把重点移向 evidence tree/graph、动态 scaffold、信息增益信用、停止判断、claim-level auditability 和对抗安全。

对单图事实核查，最可靠的内部对象不是一条自然语言推理轨迹，而是四本不可混淆的账：Claim、Evidence、Source、Failure。规划、检索与重规划只能产生候选；经过页面访问、精确 span/region、时效、相关性、立场、来源质量和独立性验证的 evidence edge 才能改变 verdict。

Coverage 应定义为关键 claim 的证据充分性，而不是“每个规划问题都有一条搜索结果”。它至少需要必要 evidence slots、质量下界、独立来源族、冲突状态、工具完整性、边际信息增益和漏证风险。`unverifiable` 是这种约束系统的正式输出，不是失败兜底。

## Key Results

- 83 个引用 URL 已做连通性审计，均返回 HTTP 200；其中大多数为 arXiv 原始论文页，另有 Google 与 Anthropic 官方技术说明。
- AVerImaTeC 与 2026 shared task 最贴近目标任务：真实 image–text claims、网页 QA evidence，并将 verdict 得分条件化在 evidence score 达标。
- 最新研究显示链接有效/主题相关与事实支持存在显著鸿沟；更多工具调用可能降低引用事实支持度。
- 完整性与停止远未解决：SeekerGym 最佳方法仍只找回预定义完整材料的一小部分；DeepSearchQA 观察到早停与低置信广撒网两种相反失效。
- 反向搜图是噪声且受时间影响的发现工具；无结果不能证明图片新或假，高排名不能证明来源可信。
- 像素级生成/篡改检测器跨生成器、平台传播和重数字化的泛化不足，并可能降低整体多模态 claim verification 表现。
- LLM-as-judge 对长轨迹和证据错误的检测不够可靠，必须用人工 gold 标定方向性 FP/FN，并配合确定性检查。

## Patterns and Insights

1. **能力瓶颈从检索转向闭环控制**：会搜索已不是主要差异，能否表示缺口、验证证据、判断停止和拒答才是。
2. **结构化中间状态正在替代自由 CoT**：evidence graph/tree、dynamic outline、claim provenance 是共同趋势。
3. **训练正在从终局奖励转向过程信用**：信息增益、hop anchor、tool-aware credit 用于长时程工具行为。
4. **可复现 search world 与 live web 双轨化**：固定语料支持科学比较，live web 检验真实鲁棒性。
5. **多模态检索转向主动感知**：视觉不再是一次性输入，而是根据证据缺口做 crop、OCR、局部图搜与增强。
6. **可信根不能是同一个 LLM**：planner、actor、auditor、judge 若共享模型与上下文，会共享锚定和盲点；需要不可变工具日志、规则 gate 与抽样人工审计。

## Lessons and Constraints

- 不将厂商系统说明等同于可重复的学术证据。
- 不把答案正确率等同于证据正确、覆盖完整或过程可靠。
- 对图片分别核验像素/生成或篡改、来源与时间地点、配文声明以及传播语境。
- 搜索 snippet 与模型摘要只用于发现，不能成为终局证据。
- URL 数量不能代理独立证据数量；需按来源族去重和折扣。
- 关键工具失败必须保真传播到最终输出；禁止语言模型构造替代的“伪成功”。
- 未标定的 LLM judge 不应用作唯一 RL reward 或确定性 judgment gate。
- 2026 预印本提供前沿机制与研究假设，但需要后续复现。

## Open Questions

- 如何将开放世界漏证风险校准为可以控制的 risk–coverage 曲线？
- 如何从网页引用、近重复、发布时间与共同图像推断 source-family independence？
- 如何用决策价值而非通用信息增益选择下一步多模态工具？
- 如何为 `unverifiable` 建立可解释且可评测的原因 taxonomy？
- 如何在不暴露隐私或被网页 prompt injection 劫持的前提下进行深度搜索？

## Recommended Direction

**DEEPEN**：以“quality- and independence-aware coverage constraints”为主贡献，先实现四本账、deterministic judgment gate、故障注入和 AVerImaTeC 基线，再扩展到主动视觉搜索。第一篇实验应比较固定流水线、普通 ReAct、question-level coverage 与 provenance-aware evidence ledger。

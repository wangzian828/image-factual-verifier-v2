# Search Agent 与单图事实核查：文献地图
更新时间：2026-07-10。范围以 2023–2026 为主，纳入少量奠基工作。以下结论优先来自论文原文摘要、论文页面或机构技术报告；`†` 表示截至本报告日期仍属预印本/最新技术报告，结论应视为前沿信号而非已经充分复现的定论。

## 1. 范式演进

### 1.1 从“浏览器 + 引用”到“推理—行动闭环”

- **WebGPT** 将浏览器操作、证据收集和带引用的长答案纳入人类反馈训练，是现代搜索型 Agent 的直接先驱 [Nakano et al., 2021](https://arxiv.org/abs/2112.09332)。
- **ReAct** 把显式 reasoning trace 与环境 action 交错，奠定 `Thought → Tool → Observation` 循环；其优点是计划可更新、工具结果可进入后续推理，但 trace 本身并不是正确性证明 [Yao et al., 2022](https://arxiv.org/abs/2210.03629)。
- **IRCoT** 证明多跳问题中检索与 CoT 交错优于一次性检索，开启“推理决定下一次检索”的主线 [Trivedi et al., 2022](https://arxiv.org/abs/2212.10509)。

### 1.2 从静态 RAG 到自适应、纠错和反思式检索

- **FLARE** 用未来句预测和低置信 token 触发主动检索，回答“何时搜、搜什么” [Jiang et al., 2023](https://arxiv.org/abs/2305.06983)。
- **Self-RAG** 学习按需检索并用 reflection token 判断相关性、支持度和生成质量 [Asai et al., 2023](https://arxiv.org/abs/2310.11511)。
- **CRAG** 为检索结果增加质量评估，低质量时触发网页搜索，并对文档分解—重组以滤除噪声 [Yan et al., 2024](https://arxiv.org/abs/2401.15884)。
- **Adaptive-RAG** 按问题复杂度选择不检索、单步或迭代检索 [Jeong et al., 2024](https://arxiv.org/abs/2403.14403)。

这些方法解决的是局部路由与纠错，但多数仍把“模型自评”当作控制信号，没有给证据充分性、来源独立性和停止条件以可验证语义。

### 1.3 从单轨 ReAct 到图规划、并行探索与深度研究

- **STORM** 用多视角提问先收集素材、再组织提纲，显示覆盖与结构化必须在写作前显式建模；作者也指出来源偏差传递和无关事实过度关联 [Shao et al., 2024](https://arxiv.org/abs/2402.14207)。
- **MindSearch** 用 WebPlanner 动态构造子问题图，多个 WebSearcher 分层检索，为“规划图 + 并行工作者”范式提供代表实现 [Chen et al., 2024](https://arxiv.org/abs/2407.20183)。
- Google Gemini Deep Research 的公开说明采用“可修改的多步计划 → 反复浏览与再搜索 → 带原始来源链接的报告” [Google, 2024](https://blog.google/products/gemini/google-gemini-deep-research/)。这是产品说明，不是独立学术验证。
- Anthropic Research 使用 orchestrator–worker 多 Agent：lead agent 规划并生成并行 subagent，汇总后决定是否继续，再由 CitationAgent 绑定引用；其内部评测宣称相对单 Agent 提升 90.2%，但也报告多 Agent 约消耗聊天 15 倍 token，且结果属于厂商内部评测 [Anthropic, 2025](https://www.anthropic.com/engineering/multi-agent-research-system)。

### 1.4 从提示工程到端到端搜索策略训练

2025 年起，核心方向变成用 SFT/RL 让模型内生学习何时搜索、如何改写查询、如何吸收结果及何时停止：

- **Search-o1** 在长推理链中交错搜索，并用 Reason-in-Documents 处理检索噪声 [Li et al., 2025](https://arxiv.org/abs/2501.05366)。
- **R1-Searcher** 用两阶段 outcome-based RL 激励自主调用搜索，无需过程奖励冷启动 [Song et al., 2025](https://arxiv.org/abs/2503.05592)。
- **Search-R1** 把 search rollout 纳入 RL，代表“检索作为推理动作”的训练范式 [Jin et al., 2025](https://arxiv.org/abs/2503.09516)。
- **ReSearch** 在无人工 reasoning step 监督下，用 RL 学出搜索、反思和自纠正 [Chen et al., 2025](https://arxiv.org/abs/2503.19470)。
- **DeepResearcher** 直接在真实、噪声和动态网页环境训练，报告相对提示基线最高 +28.9 点、相对固定 RAG-RL 最高 +7.2 点，并观察到规划、交叉验证与找不到时保持诚实等涌现行为 [Zheng et al., 2025](https://arxiv.org/abs/2504.03160)。
- **WebThinker** 交错 think–search–navigate–draft，并用在线 DPO 优化工具使用 [Li et al., 2025](https://arxiv.org/abs/2504.21776)。
- **WebDancer** 给出浏览数据构造、轨迹采样、SFT 冷启动和 RL 四阶段训练路线 [Wu et al., 2025](https://arxiv.org/abs/2505.22648)。
- **WebSailor** 进一步面向高不确定、长时程网页导航训练 [Li et al., 2025](https://arxiv.org/abs/2507.02592)。

### 1.5 2026 前沿：过程信用、显式证据结构和受控停止

- **Argus†** 将任务视为拼图：Searcher 搜局部证据，Navigator 维护共享 evidence graph、识别缺口并派发搜索；并行扩展获得增益且汇总上下文受控 [Li et al., 2026](https://arxiv.org/abs/2605.16217)。
- **DeepRubric†** 先构造 evidence tree 和原子可核验叶节点，再反向合成 query/rubric，避免“给定问题后由 LLM 猜 rubric”产生漏项；报告以约 13 倍更少 RL GPU 小时达到可比表现 [Zhu et al., 2026](https://arxiv.org/abs/2606.17029)。
- **ScaffoldAgent†** 把动态提纲更新定义为 Expansion/Contraction/Revision，并根据检索增益、结构一致性、试写质量决定更新和终止 [2026](https://arxiv.org/abs/2606.20122)。
- **VeriTrace†** 用显式认知图与解释更新、偏差反馈、schema 修订三个闭环，试图防止中间表示被混合质量信息污染 [2026](https://arxiv.org/abs/2605.26081)。
- **R²-Searcher†** 显式校准 retrieval–reasoning boundary，抽取 query token 对应的主体、动作、时间和程度等事实后再反思检索 [2026](https://arxiv.org/abs/2606.28566)。
- **IGPO / IGRPO†** 用每轮信息增益给长轨迹稠密信用，或将 rollout 预算投向高信息量树节点，回应终局奖励稀疏和计算浪费 [2025](https://arxiv.org/abs/2510.14967), [2026](https://arxiv.org/abs/2607.06223)。
- **SIGHT†** 用 self-evidence 压缩检索结果、信息增益触发去重/反思/分支，针对早期噪声造成的 tunnel vision [2026](https://arxiv.org/abs/2602.11551)。
- **Don't Stop Early†** 通过 coverage-driven objectives、依赖控制的信息流和 evidence-aware termination 减少早停；当前验证含内部企业任务，外部泛化仍需复现 [2026](https://arxiv.org/abs/2604.24978)。

## 2. 多模态搜索与单图事实核查

### 2.1 必须先区分四类“真假”

1. **像素/文件真实性**：是否 AI 生成、局部篡改、重编码；这不是“图片所配声明是否真实”。
2. **来源与原始语境**：首次/较早发布、拍摄对象、时间、地点、事件、作者/机构。
3. **图片—声明一致性**：真实图片是否被错配到错误人物、事件、地点或时间（cheapfake / out-of-context）。
4. **声明真实性与可核验性**：图片可能只起修辞作用；最终需文本、视觉及外部来源共同支持、反驳或不足。

[Akhtar et al., 2023](https://arxiv.org/abs/2305.13507) 的综述把多模态 AFC 拆成 claim detection、evidence retrieval、claim–evidence relation 和 verdict/explanation 等子任务；这支持“不要将图像检测器输出直接当 verdict”。

### 2.2 开放域外部证据取代封闭式相似度分类

- **NewsCLIPpings** 构造真实图与真实文但错配的 OOC 数据，显示像素真实不等于语境真实 [Luo et al., 2021](https://arxiv.org/abs/2104.05893)。
- **CCN / open-domain OOC fact-checking** 首次系统化利用在线文本/视觉证据，并对 caption–text evidence、image–visual evidence、image–caption 做 cycle consistency [Abdelnabi et al., 2021](https://arxiv.org/abs/2112.00061)。
- **VERITE** 通过真实数据和 modality balancing 暴露旧 benchmark 的 unimodal shortcut [Papadopoulos et al., 2023](https://arxiv.org/abs/2304.14133)。
- **RED-DOT** 在融合前先判定外部证据是否相关，说明“搜到”不等于“可用” [Papadopoulos et al., 2023](https://arxiv.org/abs/2311.09939)。
- **Similarity over Factuality** 表明简单相似度特征可击败复杂方法，反而暴露数据集/证据收集在奖励 surface shortcut，而非真正事实性 [Papadopoulos et al., 2024](https://arxiv.org/abs/2407.13488)。
- **LRQ-FACT** 让 LLM 先生成事实核查问题，再跨模态检索，直接对应“Perception → Planning Questions” [Beigi et al., 2024](https://arxiv.org/abs/2410.04616)。
- **DEFAME** 用六阶段、动态工具和搜索深度的零样本 MLLM 流水线，对图文 claim 和多模态证据生成结构化报告，并以知识截止后的 ClaimReview2024+ 检验时效泛化 [Braun et al., 2024](https://arxiv.org/abs/2412.10510)。
- **Aletheia** 将不足明确定位在搜索覆盖和 evidence quality，使用覆盖增强与无用证据过滤；论文报告相对既有检索策略最高 +30.8% verification accuracy [2025](https://arxiv.org/abs/2505.03135)。
- **RAMA** 结合精确查询构造、跨权威来源交叉验证和多 MLLM ensemble [Yang et al., 2025](https://arxiv.org/abs/2507.09174)。

### 2.3 AVerImaTeC 是与你们最接近的公开任务锚点

- **AVerImaTeC** 含 1,297 个真实 image–text claims，每项带来自网页的 QA evidence；用规范化、时间约束证据和两阶段充分性检查减轻上下文依赖、时间泄漏和证据不足 [Cao et al., 2025](https://arxiv.org/abs/2505.17978)。
- 2026 shared task 的分数把 verdict accuracy 条件化在 evidence score 超过阈值，避免“猜对标签但没有证据”得分；冠军分数为 0.5455，说明任务仍远未饱和 [Cao et al., 2026](https://arxiv.org/abs/2602.11221)。
- 一个双检索器系统将文本相似检索与 API 反向搜图解耦，仅一次多模态 LLM 生成，报告平均成本约 $0.013/例，适合作为简单、可复现实验基线 [Ullrich & Drchal, 2026](https://arxiv.org/abs/2602.15190)。
- **RW-Post†** 将原社媒 post、人工事实核查文章衍生的推理轨迹和显式证据链接对齐，并区分 closed-book、evidence-bounded、open-web 三种评测设置；当前 LVLM 在 faithful grounding 上仍有明显余量 [Xu et al., 2026](https://arxiv.org/abs/2605.10357)。

### 2.4 反向搜图是高价值但不可靠的传感器

- 2026 年对 Google RIS 的 15 天审计收集 34,486 个高排名结果，发现大量无关内容和重复误信息，辟谣内容不足 30%，结果质量随时间呈倒 U 型；这意味着“RIS 无命中”不能作为未发现来源的证据，“高排名”也不是可信性 [2026](https://arxiv.org/abs/2603.09130)。
- 因此 RIS 结果必须保留引擎、查询图/crop、时间、rank、URL、缩略图/页面 hash，并经过页面访问、时间线和来源族聚类后二次验证。

### 2.5 多模态原生 Search Agent 正从“看图后文本搜”转向主动视觉操作

- **MMSearch** 将多模态搜索拆为 requery、rerank、summarization 和端到端任务，300 个手工实例覆盖 14 子域 [Yu et al., 2024](https://arxiv.org/abs/2409.12959)。
- **OpenSearch-VL†** 统一文本搜索、图搜、OCR、crop、锐化、超分和透视校正，并用 fatal-aware GRPO 在工具失败后屏蔽级联错误 token，同时保留失败前有效推理 [2026](https://arxiv.org/abs/2605.05185)。
- **ProMMSearchAgent†** 在确定性静态 sandbox 中用 process-oriented reward 学习仅在视觉/事实不确定时发起搜索，再迁移到 live search [2026](https://arxiv.org/abs/2604.20486)。
- **DR-MMSearchAgent†** 将过早 interaction collapse 归因于终局奖励和冗余上下文，尝试用轨迹结构信号鼓励更深探索 [2026](https://arxiv.org/abs/2604.19264)。
- **Visual-Seeker†** 将视觉作为可反复关注、提取细节并在搜索过程中动态收集视觉证据的 active process，而不是一次性输入 [2026](https://arxiv.org/abs/2606.15231)。
- **SearchEyes†** 用 typed KG 同时生成多跳训练数据、可复现 search world 和 hop-level reward anchor，缓解数据—环境—奖励三者割裂 [2026](https://arxiv.org/abs/2607.05943)。
- **TAPO†** 指出 GRPO 把终局 advantage 均匀广播到所有 token 会惩罚失败轨迹中的正确工具动作，并用相似参数调用的反事实 witness 修正信用 [2026](https://arxiv.org/abs/2606.05784)。

### 2.6 像素检测器只能作为弱证据，不能裁决整条 claim

- Chameleon “sanity check” 显示 9 个现成 AI 图像检测器在真正具挑战的生成图上几乎都把假图判为真，问题远未解决 [2024](https://arxiv.org/abs/2406.19435)。
- 真实传播中的压缩、平台转发和重数字化继续显著破坏检测器泛化 [RRDataset, 2025](https://arxiv.org/abs/2509.09172)。
- 对 deepfake detector 在多模态 misinformation 中的系统研究报告：检测器单独 F1 较低，把其预测注入证据式核查反而下降 0.04–0.08 F1；语义和外部证据比非因果的“像素真伪先验”更关键 [2026](https://arxiv.org/abs/2602.01854)。
- C2PA/Content Credentials 等 provenance signal 在存在时是高精度来源线索，但缺失只代表没有可验证凭据，不能反推为假。可参照将 C2PA 与视觉 attribution 结合的早期工作 [EKILA, 2023](https://arxiv.org/abs/2304.04639)。

## 3. 评测与可靠性证据

### 3.1 终局答案正确不代表过程可靠

- **GAIA** 联合考察推理、多模态、浏览和工具使用；初始报告中人类 92%，带插件 GPT-4 仅 15%，显示工具拼装不自动带来鲁棒能力 [Mialon et al., 2023](https://arxiv.org/abs/2311.12983)。
- **AssistantBench** 的 214 个现实耗时任务中，无模型超过 26 分，最先进 web agents 接近 0；closed-book 模型会以低 precision 幻觉事实 [Yoran et al., 2024](https://arxiv.org/abs/2407.15711)。
- **WebWalkerQA** 要求纵向遍历站点子页，揭示普通 SERP 的浅层检索不足 [Wu et al., 2025](https://arxiv.org/abs/2501.07572)。
- **BrowseComp** 有 1,266 个短答案但需持久、创造性浏览的问题，适合检验“能不能找出来”，但有意回避长报告、歧义和真实用户分布 [Wei et al., 2025](https://arxiv.org/abs/2504.12516)。
- **BrowseComp-Plus** 固定语料并提供人工核验支持文档和困难负例，使 retriever、agent、context engineering 可分离评测 [Chen et al., 2025](https://arxiv.org/abs/2508.06600)。

### 3.2 动态网页让可复现性、泄漏和归因成为核心问题

- **CRAG** 覆盖从年到秒的动态事实、长尾实体和多类问题；直接 RAG 提升有限，工业 RAG 也只有 63% 问题无幻觉作答 [Yang et al., 2024](https://arxiv.org/abs/2406.04744)。
- **FreshQA** 包含快速变化知识和错误前提，所有规模模型均困难；证据数量、顺序和回答冗长度都会影响正确性 [Vu et al., 2023](https://arxiv.org/abs/2310.03214)。
- **Deep Research Bench** 用 89 个多步任务与冻结 RetroSearch 网页快照，分析 hallucination、tool use、forgetting，以减小 live web 漂移 [FutureSearch, 2025](https://arxiv.org/abs/2506.06287)。
- **DeepResearchGym** 用 ClueWeb22/FineWeb 和稳定检索构成免费、可复现 sandbox，但自动评测仍依赖 LLM judge [Li et al., 2025](https://arxiv.org/abs/2505.19253)。
- **Search-Time Contamination†** 指出 Agent 会在推理时搜到公开 benchmark 信息/答案，六个 benchmark 上最高可虚增 4%，主张隔离 sandbox 和透明轨迹 [2026](https://arxiv.org/abs/2606.05241)。
- **Mind-ParaWorld†** 用不可分 Atomic Facts 和模拟 SERP 构造模型知识之外的平行世界，发现瓶颈不仅是收集/覆盖，还有证据充分性判断与停止 [2026](https://arxiv.org/abs/2603.04751)。

### 3.3 引用“存在”与引用“支持”是两件事

- **ALCE** 将 citation recall、citation precision/entailment 等与答案正确性分开；最佳系统在 ELI5 上仍有约 50% 内容缺完整引用支持 [Gao et al., 2023](https://arxiv.org/abs/2305.14627)。
- **FActScore** 把长答案拆成原子事实并计算可靠来源支持比例，避免整段二元评分 [Min et al., 2023](https://arxiv.org/abs/2305.14251)。
- **SAFE** 用搜索 Agent 核查每个原子事实并将 factual precision 与响应长度式 recall 平衡，但它仍受搜索与判定 Agent 自身偏差影响 [Wei et al., 2024](https://arxiv.org/abs/2403.18802)。
- **RAGChecker** 分别诊断 retriever 与 generator，并报告比其他自动指标更好的人类相关性 [Ru et al., 2024](https://arxiv.org/abs/2408.08067)。
- **Cited but Not Verified†** 实际取回 citation URL 后评估 link works、relevance、fact support；强模型链接有效率 >94%、相关性 >80%，但事实支持仅 39–77%，且工具调用从 2 增至 150 时两模型平均 Fact Check 准确率下降约 42%，说明“搜更多”会稀释引用可靠性 [2026](https://arxiv.org/abs/2605.06635)。
- **Claim-level auditability / AAR†** 提出 provenance coverage、soundness、contradiction transparency、audit effort 四类指标与持久 provenance graph [2026](https://arxiv.org/abs/2602.13855)。

### 3.4 Coverage 与停止仍是最薄弱环节

- **LiveDRBench** 将 Deep Research 定义为高 concept fan-out，而非长报告；100 个任务上不同子类 F1 低至 0.02，最佳总体 F1 0.55 [Java et al., 2025](https://arxiv.org/abs/2508.04183)。
- **TaxoBench†** 上最佳 Agent 仅找回专家引用论文的 20.92%，模型 taxonomy 还有高 sibling overlap、MECE violation 与结构失衡，显示“找回 + 组织”双重瓶颈 [2026](https://arxiv.org/abs/2601.12369)。
- **DeepSearchQA†** 专测穷尽列表、去重/实体解析和开放空间停止；现代 Agent 在早停与为抬 recall 而低置信广撒网之间摇摆 [2026](https://arxiv.org/abs/2601.20975)。
- **SeekerGym†** 最佳方法仅找回 Wikipedia 42.5% passages、ML survey 29.2%，并首次把对“可能漏了多少”的不确定性校准纳入目标 [Kim et al., 2026](https://arxiv.org/abs/2604.17143)。

### 3.5 LLM-as-judge 不能作为唯一可信根

- **REFLECT†** 对真实轨迹做可控局部错误注入，最佳 judge 在 reasoning、tool-use、report failure 上总体准确率仍低于 55%，证据验证尤其差 [Wang et al., 2026](https://arxiv.org/abs/2605.19196)。
- **Citation verifier calibration†** 在 1,248 个经人工复核的 rubric decisions 上发现较便宜 judge 可具竞争力，但相似 F1 下 false-positive/false-negative drift 差异很大；若直接作 RL reward，这种方向性偏差会被策略放大 [Leung et al., 2026](https://arxiv.org/abs/2607.08700)。
- 因此 judge 应被人工标定、集成、做 abstention/不确定性，并与确定性检查（URL 可访问、时间、文本 span、hash、规则）组合。

### 3.6 安全：网页不仅有噪声，还可能主动操纵 Agent

- **AgentDojo** 系统化评测外部工具返回内容中的 prompt injection，表明现有攻击与防御都不充分 [Debenedetti et al., 2024](https://arxiv.org/abs/2406.13352)。
- **UGC poisoning†** 发现多个 Deep Research 系统对 Reddit/Wikipedia 等页面存在检索重叠，单个高频页面的短攻击文本即可影响多类查询和推荐 [2026](https://arxiv.org/abs/2605.24245)。
- **FORGE†** 展示攻击文档可从局部文本注入上升为 subtask planning 劫持；Root Query Anchoring 将小规模子集 PRISM 从 38.5% 降至 18.3%，但未消除 [2026](https://arxiv.org/abs/2607.04718)。
- **SearchGEO†** 在 13 个后端、每个 308 cases 上观察到 0–31.4% 的 endorsement attack success，且同一 scaffold 可随 backend 放大或减小风险 [2026](https://arxiv.org/abs/2606.16821)。
- **MosaicLeaks†** 指出公开查询序列可聚合泄漏私有上下文；仅优化任务 RL 会加剧泄漏，privacy-aware RL 可降低但不能证明消除 [2026](https://arxiv.org/abs/2605.30727)。

## 4. 适用于本项目的文献共识

1. 动态规划是必要的，但计划必须由显式证据状态驱动，而非由同一 LLM 的“感觉”驱动。
2. 搜索结果必须经过页面访问、证据片段定位、相关性/立场/时效/来源质量和独立性检查；snippet 不能作为终局证据。
3. 结论应绑定原子 claim，而不是绑定整份报告；每个 claim 都要有支持、反驳、冲突或缺失状态。
4. 覆盖不是“所有规划问题有一条结果”，而是关键 claim 的证据充分度、来源多样性、冲突处理和搜索空间剩余风险。
5. `unverifiable` 不是失败兜底，而是由可观察的检索边界、工具失败、冲突或证据不足导出的正式裁决。
6. 多 Agent 有利于宽搜索，但也放大成本、重复、上下文压缩损失和攻击面；只有可独立并行的 evidence slots 值得并行。
7. 像素伪造检测、元数据、C2PA、OCR、地理/时间线索、反向搜图均应作为带校准可靠度的传感器，而非某个传感器一票裁决。

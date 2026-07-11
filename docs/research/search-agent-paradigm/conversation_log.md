# 项目研究对话记录

> 这是面向项目协作的对话记录，不是聊天平台的原始数据导出。用户关键表述按原话保存；较长的助手回复以完整研究结论形式整理在 `discussion_2026-07-10.md`，避免同一内容维护两份并逐渐不一致。后续相关讨论可继续追加到本文件。

## 2026-07-10：启动 Search Agent 范式调研

### 用户

> 启动一个全面的调查，调查当前search agent的范式。我们的项目大概有这样的特点。这是一个面向单张图片事实核查的、多阶段证据驱动 Agent：采用 Perception → Planning → Verification/ReAct → Coverage Audit/Replanning → Judgment 架构，通过 Gemini Interactions API 的原生函数调用迭代使用反向搜图、网页搜索、来源访问和视觉一致性分析等工具；系统会围绕规划问题持续搜索、检查证据覆盖并动态重规划，最终输出 real/fake/unverifiable。核心范式是多阶段 Agent、开放域多模态检索、工具增强推理、证据溯源和确定性覆盖约束，强调结论必须绑定真实工具结果，失败显式暴露，不使用人脸识别、固定工具流水线或伪成功 fallback。我希望你调研最前沿的论文和技术报告，告诉我现在search agent有哪些问题，我们应该解决哪些问题，现在都是怎么进行解决的。落成一个报告，要标注好引用来源等，方便另一个agent去看和学习。

### 结果

- 完整报告：`to_human/search_agent_paradigm_report_zh.md`
- HTML 版：`to_human/search_agent_paradigm_report_zh.html`
- 文献地图：`literature/survey.md`
- Agent 学习指南：`to_human/agent_learning_guide.md`
- 机器可读元数据：`literature/references.json`、`literature/references.csv`

## 2026-07-10：澄清项目位置

### 用户

> 哦对，我们的项目位置其实是这里，你可以把之前的东西也放进去 `D:\image-factual-verifier-v2`。

### 处理

调研资料被复制到：

```text
D:\image-factual-verifier-v2\docs\research\search-agent-paradigm
```

原位置保留为备份：

```text
C:\Users\wangza\search-agent-paradigm-research
```

## 2026-07-10：一边看、一边搜的主动视觉调查

### 用户

> 唔，我想和你直接聊聊，首先是我们的agent链路，我大概想做出来的效果是那种一边看一边搜的效果，逐步加深对图像的认知，你感觉这个的创新程度怎么样。

### 达成的共识

“一边看、一边搜”不应只是循环调用工具，而应形成：

```text
Perception ↔ Search ↔ Hypothesis Update
```

搜索结果需要反向生成新的视觉观察任务，例如局部 crop、OCR、局部反向搜图和候选事件差异检查。认知更新需要显式保存为：

```text
旧假设状态 → 实际工具结果 → 新假设状态
```

单纯循环的创新度有限；如果进一步加入显式视觉假设状态、面向最终 verdict 的行动价值、来源独立性、覆盖约束以及校准停止，可以形成较强的完整研究贡献。

建议定位：

> Evidence-Conditioned Active Visual Investigation Agent

完整讨论见 `discussion_2026-07-10.md` 的第 2–3 节。

## 2026-07-10：用户确认研究方向

### 用户

> 对对对，我和你想的一样，我只是没表述清楚。

### 共识

真正的贡献不是“多一个 Agent 阶段”，而是把“逐步加深认知”定义成可观察、可约束、可评测的证据更新过程。

## 2026-07-10：RL 必要性、难度和数据构造

### 用户

> 接下来我想的是这样的链路，rl的必要性和rl的难度，以及数据构造上的问题。

### 达成的共识

RL 有价值，但不是整个系统成立的必要条件：

- RL 适合优化 `Belief State → Next Investigation Action`；
- Evidence/Source/Failure ledger、真实工具结果绑定、错误传播和最终 judgment gate 应由确定性系统保证；
- Prompting/SFT 先负责感知、问题分解、候选动作生成与证据抽取；
- 应依次比较启发式、Prompted ReAct、SFT、动作排序/contextual bandit 和长时程 RL，实证证明 RL 的必要性。

RL 的主要难点：

1. 稀疏终局奖励和长轨迹信用错配；
2. coverage、证据数、信息增益等 proxy 容易被 reward hacking；
3. live web 非平稳且不可复现；
4. 任务是部分可观测的 POMDP；
5. query、crop、工具和参数形成巨大动作空间。

建议采用高低层解耦：

```text
RL 高层：调查哪个 claim、选哪类工具、探索/验证/反证、继续/停止
SFT 低层：生成具体 query、crop box、页面抽取参数
```

数据需要记录 belief、候选动作、实际 observation、状态更新和剩余缺口，而不只是“工具调用序列”。核心数据类型包括成功轨迹、失败轨迹、成对反事实分支、工具故障、时间变化和来源依赖。

建议三层数据体系：

1. 可控合成 Search World；
2. 真实网页冻结快照；
3. Live Web 迁移评测。

完整讨论、奖励草案和训练阶段见 `discussion_2026-07-10.md` 的第 4–9 节。

## 当前最值得继续讨论的决策

1. 第一篇工作主打 active visual investigation，还是 coverage/stopping？
2. 输入是否始终带 textual claim？
3. 第一版 belief state 的最小 schema 是什么？
4. RL 的高层动作集合如何定义？
5. gold evidence graph 与 source-family 标注怎样低成本获得？

## 2026-07-10：审阅前期 Benchmark 构造方案

### 用户

> 我们接着讨论思路吧，接下来是数据构造相关的，这是我们前期一个初步的数据构造计划，我们现在有不限量的gemini的生图api。

### 输入方案

原文件：

```text
D:\wangza\xwechat_files\wxid_0xjg7sn6hmf622_a974\temp\RWTemp\2026-06\e2016d84e2397cde470c1f9b7aa7b729\benchmark构建方案(2).md
```

### 初步共识

- 保留 `context_description` 语义槽位、单变量扰动和 evidence requirement 思想；
- 单图必须包含可从像素恢复的显式/受控隐式 claim，否则“图片传达的信息”没有唯一真值；
- 全合成 real/fake 若没有配套可搜索 evidence world，会训练生成器指纹和标签先验，而不是调查策略；
- 不限量 Gemini 生图最适合生成同一世界的多视图、局部线索、反事实配对、困难负例和网页配图；
- 数据单元应从 `image` 升级为 `Investigation World`：world state + claim + image assets + evidence/source graph + search index + action branches + stop states；
- 教师不能看到标签后再伪造“最优轨迹”。应由 blind investigator 真实 rollout，privileged verifier 只做过程标注；
- 全合成部分适合作为训练/RL Gym，最终 benchmark 需加入严格隔离的真实冻结网页集和 live rolling test；
- 原 AF2/EF4 中的人脸识别/人脸反搜与项目硬约束冲突，应排除或改为非生物特征的公开上下文核验。

完整审计与改造方案见：`data_construction_review_2026-07-10.md`。

## 2026-07-10：如何具体构造“完整、可查询的世界”

### 用户

> “应该先构造一个完整、可查询的世界”这个你具体打算怎么弄，我感觉稍微有点困难。

### 澄清

这里的“完整世界”不是模拟整个互联网，而是构造 **case-scoped Evidence Capsule / Micro-Web**：只要求围绕一个目标 claim 的决定性事实、候选解释、支持/反驳来源、转载关系、视觉资产与停止条件闭合。

多个 Evidence Capsules 进入共享文本和图像索引，并加入大量跨世界困难负例后，Agent 使用任意自然语言 query 检索，对它而言仍是开放搜索；构造者则拥有可计算 coverage、动作价值和正确停止点的 gold evidence graph。

建议 MVP：20 个世界，每个 5–10 个原子事实、10–15 个文档、4–8 个图像资产和 2–4 个来源族；先验证完整管线，不直接追求数千个独立世界。

## 2026-07-10：闭合环境如何证明开放互联网能力

### 用户

> 但是这样就相当于是在一个闭合环境里训练测试了吧，这怎么能说明在开放互联网上的能力呢？相关论文有这么做的？

### 结论

- 用户的质疑成立：闭合环境内训练并在同一环境测试，不能证明开放互联网能力；
- 相关论文确实使用模拟站点、固定网页语料、缓存工具交互或 simulated search world，但严谨工作会另做 sim-to-real 或 live-web 对照；
- ProMMSearchAgent 在本地静态 sandbox 训练，测试时零样本切换 live Google Search/Lens，是最贴近的多模态案例；
- DeepResearchGym 用 ClueWeb22/FineWeb 训练，并统一在商业搜索后端测试迁移；Deep Research Bench 对比 RetroSearch 与 live web；WebShop 早期即做模拟环境到 Amazon/eBay 的零样本迁移；
- SearchEyes 和 Mind-ParaWorld 更适合证明结构学习、无泄漏评测和过程诊断，如果缺少 live-web 对照，不能单独支持开放 Web 声明；
- 本项目应采用 `synthetic capsule → frozen real web → hidden live rolling test → cross-backend/time replication` 的分层证据链，而不是把合成闭环当最终 benchmark。

详细论文证据、混合训练比例、live-web 协议和声明边界见 `closed_to_open_validation_2026-07-10.md`。

## 2026-07-11：Internet-grounded synthetic world

### 用户

> 合成世界应该也来自于互联网，相当于清洗了一下互联网，这样不就行了，清洗出来一部分内容，作为锚点，然后通过内容，去生成对应的图片，这样图片和对应的图片世界就匹配了，这样的思路呢？

### 当前共识

这个方向成立，并且比凭空生成事实世界更贴近本项目。可将其定义为 **Internet-grounded synthetic visual worlds**：互联网提供事件事实、时间线、实体关系、真实来源及转载结构；构造流程把它们清洗为带溯源的原子事实和 evidence graph；Gemini 只负责将其中可视化的事实实例化为图片、多视图、局部线索和受控反事实。

关键区分是：

```text
互联网事实世界 / 来源图（真实锚点）
        ↓ 清洗、原子化、保留 provenance
结构化 world model + claim/evidence graph（构造者 gold）
        ↓ 受约束生成与人工/模型一致性验收
事实一致图 / 反事实图 / 不可判定图（视觉 observation）
```

这能够解决纯合成世界“事实是任意编写的”、语义与真实 Web 脱节以及训练时缺少 gold evidence graph 的问题，但不会自动解决以下问题：

- 新生成图片本身仍没有互联网首发记录，不能用来训练或评测原图 provenance/RIS 命中能力；
- “符合真实事件事实的生成图”只是事实一致的合成描绘，不是该事件的真实现场照片；若 claim 是“这是一张现场实拍”，它必须判假，若 claim 是“图中描绘事件 X”，则可判支持；
- 如果训练时 Agent 直接访问清洗后的干净 facts，它学到的是查数据库，不是处理真实互联网噪声；clean world 应作为隐藏 gold，Agent 仍应访问原始/冻结网页、SERP、转载和失败结果；
- 如果所有支持样本和反驳样本使用不同生成流程，模型仍可能利用生成器、编辑、画风、分辨率、模板或 OCR 指纹猜标签；
- 图像必须只编码可以从像素恢复、且能与网页证据连接的 claim，否则 Agent 无法知道应核查哪一项事实；
- 互联网来源并不天然正确，必须对来源依赖、时间版本、冲突和事实核查结论做人工或高可靠审计。

真正有价值的数据连接是：

```text
image region ↔ visual atom ↔ claim atom ↔ webpage span ↔ source family
```

其中 clean graph 只用于 Coverage、过程奖励、候选动作价值和停止条件；运行时 observation 必须来自实际工具结果。

初步建议采用三种图像角色而不是统一称为 real/fake：

1. `documentary_original`：真实互联网中的原始/可追溯图片；
2. `synthetic_supported_depiction`：生成图与目标 claim 的事实内容一致，但不声称是现场原图；
3. `synthetic_counterfactual_depiction`：在同一生成管线中只改变一个决定性事实槽位；
4. `insufficient_depiction`：刻意移除决定性可视线索，使正确答案为 `unverifiable`。

最终产品标签应由 `claim semantics + evidence state` 映射，而不能由“是否 AI 生成”映射。最终开放互联网能力仍需在严格隔离的真实图片、冻结真实网页和 hidden live rolling cases 上单独验证。

## 2026-07-11：原始真实图是否必要，以及如何说明合成图价值

### 用户

> 我感觉还是有点乱，我担心的可能还是原始真实图的获取比较困难，还有如何讲清楚合成图的价值。

### 澄清

不应把“每个互联网事实 world 都需要一张配对原始实拍图”设为数据构造前提。建议把两个目标拆开：

1. **主训练任务：互联网事实锚定的合成视觉调查。** Query image 可以完全由生成模型产生，互联网网页只负责提供独立于生成器的事实与证据。这里训练的是从视觉线索归纳待核查命题、搜索真实 Web、再返回图片主动检查的策略；不训练原图首发溯源。
2. **小规模真实迁移测试：真实图片事实核查。** 只需在最终 dev/test 中准备一批严格审计的真实图片，用来验证由合成数据学到的调查策略是否迁移，并单独测试 RIS/provenance。

因此合成图不是廉价冒充真实图片，也不是“真实性证据”；它的角色是一个**由真实互联网事实图约束、可精确干预的视觉 observation / query generator**。它相对真实图最不可替代的价值是：

- 保持互联网 evidence world 不变，只改变一个视觉事实，得到因果配对的支持/反驳样本；
- 精确知道哪个 image region 表达哪个 claim atom，进而监督 search-conditioned reinspection；
- 系统生成真实采集很难覆盖的歧义、遮挡、局部线索、困难负例和正确 `unverifiable`；
- 对同一 belief state 实际执行多个候选动作，计算动作带来的 evidence/coverage delta，支撑 action ranking 与 RL；
- 避免大规模真实图版权、隐私、许可和人工配对成本。

它明确不能替代：真实图片的首发来源、传播路径、裁剪转载历史、平台压缩以及真实 RIS 分布。因此这些能力只能由小规模真实图测试集证明。

论文中最清楚的核心表述可以是：

> 我们不是使用生成模型创造事实，而是把独立互联网来源所定义的事实和反事实，渲染成可控制的视觉观测。结构化事实图使我们能够知道哪个像素区域表达哪个待核查命题，并生成只改变一个决定性事实的配对样本，从而监督“搜索后重新观察图片”的调查策略。所有关于真实图片泛化的结论，只由未修改的真实图片测试集给出。

为避免标签语义混乱，研究数据内部建议使用：

```text
supported_depiction / refuted_depiction / insufficient_evidence
```

而不是直接把生成图称为 `real`。只有在产品层明确将问题定义为“图片表达的事实是否得到支持”时，才映射为 `real/fake/unverifiable`。

最重要的验证实验不是证明合成图“看起来像真实照片”，而是固定模型、真实训练数据量和工具预算，比较：

```text
真实数据 only
真实数据 + 普通合成图
真实数据 + internet-grounded counterfactual 合成图
```

然后全部只在隐藏真实图片 + live/frozen Web 上评测。如果第三组显著提高决定性证据发现、搜索后正确重看区域、停止校准和最终有据裁决，才能实证说明合成图的价值。

## 2026-07-11：训练生成管线与真实评测数据难度

### 用户

> 我大概明白了，你的意思是训练数据可以不需要大规模的真实来源的图片，但是评测时候收集一部分去评测是这样吗？接下来我需要你评估一下构建训练数据的生成管线的难度和评测那部分的真实数据来源的难度。

### 结论

是，但增加一个限定：训练主体可以不依赖大规模真实 query images，最好保留少量真实图片做开发期格式校准；用于能力声明的真实图片必须严格隔离。两部分的难点不同：

- 训练生成管线是**中高难度**。生图本身容易，最难的是验证支持图和反事实图是否只在指定 claim atom 上不同，以及 clean evidence graph 中的网页片段是否真正支持事实、来源是否独立；若进一步构造真实候选动作分支和 RL 过程奖励，难度上升到高。
- 真实评测集是**高难度、低规模高单价**。图片发现不是最大问题，最大问题是恢复像素可表达的 claim、原始/早期来源、准确证据、来源族、时间截点、冻结 SERP/RIS observation、许可和最小充分证据集。
- 如果输入严格只有裸图片，真实案例还必须在像素中带可恢复的显式或隐式 claim；普通新闻照片没有唯一待核查主张。若允许图片加原始配文，数据来源会显著扩大、标注难度会下降。
- 真实评测可进一步拆为较大的 Semantic/OOC investigation 轨道与较小的 Provenance/RIS 轨道。前者不要求每例恢复绝对首发图，只要求真实 query image、明确 claim 和充分外部证据；后者才要求原始或“最早可确认”图片及传播链。这样可避免原始图获取成为整个评测集的硬阻塞。

建议先做：

```text
Gate 0：10 worlds × 4 合成变体，验证机制
Gate 1：50 worlds / 200 合成图 + 100 个公开真实案例适配 + 30–50 个自建真实案例
Gate 2：300–500 worlds / 1,500–3,000 合成图 + 200–300 个冻结真实测试 + 30–60 个 hidden rolling cases
```

粗略人力估算：50-world 训练 MVP 约 50–110 人日；若不做反事实动作分支/RL 可降为 30–60 人日。论文级静态真实测试集 200–300 例约 100–300 人日；适配公开数据 100–200 例约 10–25 人日；hidden rolling test 每轮 30–60 例约 20–60 人日。估算假设已有 LLM、生图、搜索、浏览、OCR 接口；实际成本受 claim 定义、许可、网页冻结深度和双人标注要求影响很大。

完整分步骤难度、自动化率、风险、来源路线、许可策略和 Go/No-Go 标准见：`data_pipeline_feasibility_2026-07-11.md`。

## 2026-07-11：RIS 不只是找原图

### 用户

> 我对这里的图片反搜的理解，并不是一定要找到原图，图片反搜不能找到相关图片然后再进行判断吗，找到原图能直接判定真，但是找到相关图片也是有用的。

### 修正后的共识

用户的理解更接近本项目目标。此前把 RIS 与 provenance 过度绑定；RIS 更一般的角色是**视觉相关证据发现与候选假设生成**。它可能返回：同图转载、裁剪/编辑版本、同一事件不同视角、同一地点或实体的其他时间、以及视觉相似但无关的困难负例。

推荐关系标注：

```text
R0 exact_duplicate
R1 derived_version
R2 same_event_different_view
R3 same_entity_place_other_time
R4 visual_analogue_unrelated
R5 unresolved
```

R2/R3 对“搜索驱动重新感知”尤其重要：Agent 可以从相关图及其页面得到候选事件或区别线索，再返回 query image 检查地标、结构、路牌、天气、物体关系等区域。训练和评测因此都不需要把“找到绝对原图”作为 RIS 成功的必要条件。

但“命中完全相同图片即可直接判真”需要增加条件。完全相同或更早出现的图片是强 provenance/context 线索，仍需访问承载页面，验证来源、时间和页面陈述，并检查 query 是否被裁剪或编辑。只有该页面语境确实支持当前 claim 时，它才成为强支持；若页面给出不同时间/地点，则反而成为强反证。视觉相似和 RIS 排名本身不是 verdict evidence。

真实评测因此拆为：较大的 Semantic/RIS investigation 主轨道，不要求找到原图，标注 top-k 视觉关系及其页面的证据价值；较小的 Provenance 子轨道，才要求原始或“最早可确认”来源。建议指标包括 useful-visual-neighbor recall@k、false-bridge rate、RIS 后正确重观察区域比例、evidence-bearing page recall、相对 no-RIS 的 decision gain，以及无结果时的校准。

这一修正也改变训练数据验收：合成图虽不需要互联网原图，但要训练 RIS 策略，就必须存在可检索的视觉邻域。生成后应真实调用目标 RIS，保存 top-k，并按 `RIS-actionable / text-search-only / unsearchable` 分类。只有 top-k 中存在能建立或区分候选的同地点、同对象、同事件不同视角等结果，并且承载页面含可核查事实时，才能把它视为 RIS-actionable 训练样本。纯粹相信“生成内容来自真实事实，因此反搜自然能找到相关图”是不够的。

## 2026-07-11：两篇论文按任务/Benchmark与训练/Agent拆分

### 用户

> 等会，这拆分对吗，我大致的想法是第一篇可能以任务命题的提出和数据bench的发布为主要贡献，第二篇以训练策略，agent为主要贡献。

### 共识

这一拆分是正确且更符合依赖关系的：

```text
Paper 1：定义能力，并证明现有系统缺少这种能力
Paper 2：使用 Paper 1 的任务与指标，训练出这种能力
```

数据也应相应拆开：第一篇主要构建少而精、严格审计的真实 evaluation data；第二篇主要构建大规模 Internet-grounded synthetic learning data。第一篇不承担大规模合成训练世界、完整轨迹树和 RL，只用强而透明的 prompted/scaffold baselines 诊断现有 Agent 在 `search result → new visual question → target region → belief update` 上的能力缺口。

Paper 1 的贡献为：Evidence-Conditioned Active Visual Investigation 任务定义、真实图片/冻结 Web/RIS benchmark、region–claim–evidence 与 R0–R5 标注、新的过程和终局指标，以及系统性 failure taxonomy。Paper 2 的贡献为：互联网事实锚定的合成视觉世界、显式 belief state、搜索—重观察策略、证据/覆盖约束，以及 SFT/action ranking/RL 的真实迁移实验。

Paper 1 的 hidden real test 必须在 Paper 2 大规模 world generation 和训练前冻结并隔离。Paper 2 的核心结果必须来自 Paper 1 未见真实图片测试，而不能只报告合成环境分数。如果 RL 不能优于相同预算的 SFT/action ranking，则不宣称 RL 必要。

完整贡献边界、两篇的最小实验、数据接口和风险控制见：`two_paper_split_2026-07-11.md`。

## 2026-07-11：增强第一篇的自动构造与参考 Agent 贡献

### 用户

> 我感觉第一篇贡献稍微单薄了一些，可能至少也要包装一下自动化数据构造，然后给出一个简易的解决的模型或者agent把。

### 修正后的 Paper 1

同意。单独的任务定义、约 200 个真实案例和现成 Agent 基线容易被认为是增量 benchmark。第一篇应升级为一条统一的五段论证链：

```text
新任务命题
→ human-in-the-loop 半自动真实数据构造引擎
→ 真实主轨道 + 受控合成诊断轨道 benchmark
→ 无需 RL 的 ReInspect / Evidence-to-Vision 参考 Agent
→ 系统诊断与剩余能力缺口
```

半自动 builder 从事实核查文章、公开 image–text claims 或许可事件页面抽取 query image、claim atoms、evidence spans 和 source families，实际执行并冻结 RIS top-k，提出 R0–R5 关系、search-conditioned visual question 与 target region，最后通过多检查器和人工接受/修改/拒绝。Paper 1 应评估 builder 自身的标注准确率、case acceptance yield、人工分钟/case 和相对纯人工节省，而不能把 LLM 自动输出直接称为 gold。

参考 Agent 的关键机制不是长 prompt，而是显式的 `web/RIS observation → visual question → region action → visual observation → hypothesis update` 状态转移。外部搜索文本不能直接修改视觉 belief；必须先生成目标假设、区别性视觉属性和目标区域/动作，再由真实 crop/OCR/compare observation 更新。Agent 维护 hypothesis/evidence/visual-question/region-observation/coverage-failure ledgers，并受 deterministic judgment gate 约束。

第一篇可以加入 50–100 个 internet-grounded worlds 的小型受控合成诊断子集，用于测试区域、单变量反事实和 stopping，但不用于大规模训练。第二篇仍独占大规模 learning worlds、候选动作分支、SFT/action ranking/RL 和真实迁移优化。因此两篇边界变成：Paper 1 研究如何定义、构造和以结构化基线解决任务；Paper 2 研究如何规模化学习最优调查策略。

## 2026-07-11：纯图片、图片加 Claim 与开放图像调查

### 用户

> 我其实是在想图片+claim，和现在有的工作重合度太高了，我因此才想着只去用图片。但是其实感觉这样自我放弃了一个模态也有点难做。只有图片，和图片内容理解的工作是不是也有点重合，我不是很清楚。这里我确实还在思考。

### 当前判断

这不是简单的输入格式选择，而是三种不同任务。图片加外部 claim 最容易定义和评分，但与 AVerImaTeC、OOC fact-checking 和现有多模态核查高度接近；任意纯图片直接输出 `real/fake` 则既容易退化为生成图/篡改检测，又存在语义不成立的问题，因为无配文图片通常没有单一真假值。

推荐第三条路径：**Image-Centric Open-World Investigation**。系统只接收图片，但不是直接猜真假，而是区分 direct observations、图内 explicit claims 和 Agent 自己提出的 attribution hypotheses；主动选择少量有决策价值且可搜索的 investigation targets，经 RIS/Web/OCR/crop 和搜索驱动的局部重观察，最终输出带来源的 `Verified Image Account`，分别报告来源、事件、地点、时间、内容来源、冲突和未知项。

这并没有放弃文本模态：运行时仍是 `visual input → textual search questions → multimodal evidence → visual reinspection`。与普通图片理解的关键区别是，输出不是 caption，而是有外部证据、有来源、有冲突和拒答状态的事实档案；与 knowledge VQA 的区别是问题不是给定的，Agent 必须决定什么值得核查；与 image+claim fact-checking 的区别是系统还承担 target/claim induction。

为了使开放 target induction 可评分，每个 case 应构建 `Investigation Rubric Graph`，包含决定性 slots、visual markers、evidence spans 和 source families。Agent 的自由语言 target 映射到 rubric leaves，评估 target utility/coverage、region grounding、evidence soundness、belief update 和 stopping，而不是逐字匹配问题文本。

不建议现在凭直觉二选一。先各做 30 例 pilot：A=`image+external claim`，B=`single claim-bearing image`，C=`claim-free public-world image → verified account`。比较标注者对目标的一致率、单例 gold 成本、no-tool VLM 表现、外部搜索必要性、搜索后产生新视觉问题的比例和输出可评分性。若 C 的 target agreement 太低则先采用 B；若 B 可被 OCR-only 轻易解决，则采用 B+C 分层并分别报告。

完整任务定义、重合矩阵、结构化输出和 pilot 标准见：`image_only_task_definition_2026-07-11.md`。

## 2026-07-11：第一篇必须以互联网调查为核心

### 用户

> 等下等下，这个描述还是有点毛病，第一篇不能不提互联网的。

### 修正

同意。此前用“纯图片任务”概括第一篇容易错误地把它描述成封闭式图片理解。准确设定是：**查询侧只有一张图片，但 Agent 的环境、证据和推理过程始终是开放互联网、多模态的。**

第一篇应定义为 `Single-Image-Initiated Open-Web Investigation` 或 `Internet-Grounded Image Investigation`：输入为单图且没有外部提供的 claim；Agent 必须从图像诱导可核查目标，使用 RIS、网页搜索和来源访问发现候选语境，再让互联网证据触发局部重观察，最终输出 Internet-grounded verified image account。

互联网在第一篇中承担四个不可替代的角色：任务环境、事实与来源 gold、数据自动构造来源、以及检索/证据/停止指标的评测对象。因此创新不是“少输入一个文字模态”，而是“从图片自主决定该向互联网追问什么，并让互联网结果改变后续视觉观察”。

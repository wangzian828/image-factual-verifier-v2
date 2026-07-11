# 主动视觉调查 Agent：项目讨论记录

日期：2026-07-10

项目：`D:\image-factual-verifier-v2`

> 本文整理本次对话中已经达成的研究共识、尚未确定的问题和建议实验。它不是运行时实现规范；当前实现以项目的 `AGENTS.md` 和 `docs/architecture.md` 为准。

## 1. 项目目标与当前链路

项目当前面向单张图片事实核查，采用：

```text
Perception
→ Planning
→ Verification / ReAct
→ Coverage Audit / Replanning
→ Judgment
```

系统通过 Gemini Interactions API 的原生函数调用，迭代使用反向搜图、网页搜索、来源访问和视觉一致性分析等工具，最终输出：

```text
real / fake / unverifiable
```

已有硬原则包括：结论必须绑定真实工具结果；工具失败必须显式暴露；不使用人脸识别；不采用固定工具流水线；不提供伪成功 fallback。

## 2. 我们真正想实现的体验

用户对目标效果的表述是：

> Agent 一边看、一边搜，逐步加深对图像的认知。

进一步澄清后，这不是简单重复：

```text
看图 → 搜索 → 再看图 → 再搜索
```

而是让外部证据反过来改变 Agent 下一次如何观察图像：

```text
观察图像
  ↓
形成视觉事实、候选假设与未知项
  ↓
搜索外部证据
  ↓
证据更新或分裂候选假设
  ↓
针对新的关键区别重新观察原图
  - 裁剪局部
  - OCR
  - 局部反向搜图
  - 检查标志、建筑、路牌、天气或几何关系
  ↓
更新证据状态与剩余缺口
  ↺
  ↓
充分时裁决；不足或冲突时 unverifiable
```

核心闭环可概括为：

```text
Perception ↔ Search ↔ Hypothesis Update
```

### 示例

1. 初始观察认为图片可能来自洪灾事件 A 或 B。
2. 搜索发现 A、B 的关键区别在桥梁护栏和远处路牌。
3. Agent 回到原图裁剪护栏区域，并对路牌做 OCR。
4. 局部证据排除 A，提升 B 的可能性。
5. 新的地点线索触发更具体的网页搜索和局部反向搜图。
6. 页面来源、时间线与图片区域共同形成可审计证据，而不是仅凭新的语言描述改变结论。

## 3. 创新程度判断

### 3.1 仅有循环交互时

单纯让模型循环调用 crop、OCR、反向搜图和网页搜索，创新度中等偏低。现有 DEFAME、Visual-Seeker、OpenSearch-VL、ProMMSearchAgent 等工作已经接近 active visual reasoning 或 multimodal deep search。

### 3.2 可以形成较强创新的四个条件

#### A. 搜索驱动重新感知

搜索不是只给最终答案补充文本，而是产生下一次视觉观察任务：看哪里、裁剪哪里、读取什么、比较什么。

#### B. 显式认知更新

认知变化不能只藏在自由 CoT 中。系统应维护可审计的视觉假设状态，例如：

```text
H1：图片来自事件 A        confidence: 0.45
H2：图片拍摄于 2024 年    confidence: 0.30
H3：画面中的建筑是 B      confidence: 0.70
```

工具结果只能执行明确状态更新：

```text
support(H1)
refute(H1)
split(H1 → H1a, H1b)
create(H4)
mark_unknown(H2)
```

必须保存：

```text
旧状态 → 实际工具结果 → 新状态
```

#### C. 面向最终裁决的视觉信息价值

下一步动作不应按“能不能发现新东西”选择，而应按“是否可能改变最终 verdict”选择：

```text
action utility
= expected reduction in verdict uncertainty
× evidence reliability
× claim criticality
− tool cost
− failure risk
```

暂定概念名称：**decision-relevant visual information gain**。

#### D. 可验证的停止与拒答

循环必须判断：

- 哪些决定性假设仍未验证；
- 新结果是否来自新的独立来源族；
- 当前是否只是重复同一上游消息；
- 同等级冲突是否仍无法解决；
- 关键工具是否失败；
- 继续搜索改变 verdict 的可能性是否足够高。

无法消除关键不确定性时，正式输出 `unverifiable`。

### 3.3 当前创新定位

```text
循环看图与搜索                         中等偏低
搜索结果驱动重新感知                   中等
显式视觉假设状态 + 证据更新            中高
再加裁决价值、来源独立性、覆盖与校准停止  较强的完整研究贡献
```

建议定位名称：

> **Evidence-Conditioned Active Visual Investigation Agent**
> 通过外部证据持续重新观察原图、更新视觉假设，并在证据充分性约束下作出裁决的主动视觉调查 Agent。

## 4. RL 是否必要

### 4.1 总体结论

RL 有价值，但不是整个系统成立的必要条件。RL 最适合学习长时程的“下一步调查行动策略”，不应负责系统不变量。

建议链路：

```text
初始图像与 claim
    ↓
构造 Belief State
  - 当前视觉事实
  - 候选假设
  - 已验证/未验证 claim
  - 证据缺口与冲突
    ↓
选择下一步调查动作
  - 看哪个区域
  - OCR / crop / RIS / Web Search
  - 访问哪个来源
  - 验证哪个候选
  - 搜索反证
  - 停止或拒答
    ↓
工具返回真实 Observation
    ↓
Evidence Verification
    ↓
更新 Belief State
    ↺
    ↓
Deterministic Judgment Gate
```

RL 主要优化：

```text
Belief State → Next Investigation Action
```

### 4.2 不应交给 RL 的部分

以下应作为显式程序、schema 或硬约束：

- Claim 原子化后的唯一标识与状态；
- Evidence、Source、Failure ledger；
- URL、页面片段、图像区域和 `tool_call_id` 的绑定；
- 来源去重和来源族聚类；
- 工具错误类型及失败传播；
- 最终裁决的最低证据门槛；
- 不满足条件时强制拒绝确定性 verdict；
- 禁止引用不存在或失败的工具结果。

这些是系统不变量，不是需要通过 reward 猜出来的行为。

### 4.3 Prompting / SFT 足以解决的部分

- 初始视觉观察；
- 将 claim 拆成调查问题或 evidence slots；
- 依据缺口生成候选查询；
- 从页面抽取候选证据；
- 判断 support / refute / unclear；
- 生成合理的候选动作。

### 4.4 RL 真正适合解决的部分

1. 下一步应观察哪个图像区域；
2. 下一步应搜索什么，如何区分候选假设；
3. 何时从文本搜索返回图像重新观察；
4. 何时探索新候选，何时验证当前候选，何时主动找反证；
5. 如何在有限预算中分配昂贵工具调用；
6. 何时停止或输出 `unverifiable`。

这些决策的好坏通常数步后才能体现，存在明确的长时程 credit assignment 问题。

### 4.5 在证明 RL 必要性之前的对照路线

```text
固定规则/启发式
    ↓
Prompted ReAct
    ↓
SFT Investigation Agent
    ↓
候选动作排序 / contextual bandit
    ↓
Long-horizon RL
```

论文中应比较以上层级。如果动作 ranker 已经解决大部分问题，就不能只因“RL 更新”而宣称 RL 必要；如果 RL 在相同预算下显著提高决定性证据发现率，才能证明它的独立价值。

## 5. RL 的主要难点

### 5.1 稀疏终局奖励与信用错配

一次轨迹可能是：

```text
crop → OCR → search → visit → RIS → compare → judgment
```

最终正确不代表每一步都好；最终错误也不代表前面的正确 crop 或有效查询应受罚。将终局 advantage 广播到整条轨迹会错误惩罚失败轨迹中的高价值动作。

可考虑的过程信号：

```text
Δ 决定性 claim 的证据覆盖
Δ 候选假设的可区分度
新独立来源族
发现有效反证
排除错误候选
减少未解决冲突
正确识别工具失败
```

这些奖励应尽量由外部可观测的 evidence state 计算，而不是让另一个 LLM 仅凭语言评价“这一步看起来不错”。

### 5.2 Reward hacking

- 奖励证据数 → 搜大量重复网页；
- 奖励问题覆盖数 → 生成容易满足的问题；
- 奖励自报信息增益 → 通过修改语言置信度作弊；
- 只奖励最终答案 → 依赖模型记忆或数据泄漏；
- 奖励域名多样性 → 找许多互不相关的低质量来源。

更合理的是分层约束：

```text
硬门槛：
  工具结果真实
  无伪引用
  证据可访问
  关键失败未被隐藏

主任务：
  verdict 正确
  decisive claims 有充分证据

质量：
  来源独立
  冲突透明
  coverage 完整
  校准良好

效率：
  工具次数、token、延迟、费用
```

效率收益不能补偿一次伪证据或关键失败隐藏。

### 5.3 Live Web 非平稳且不可复现

同一查询会随时间、地域、搜索引擎、索引和 API 状态变化。直接在 live web 做大规模 RL 会带来高方差、难复现、高成本和搜索引擎 shortcut。

建议双层环境：

```text
训练：冻结 Search World / 缓存网页 / 可控故障
评测：冻结环境 + Live Web 双轨
```

### 5.4 问题本质上是 POMDP

Agent 无法直接观察：

- 是否仍有关键证据未发现；
- 多个页面是否来自同一上游；
- RIS 无结果是事实还是索引缺口；
- 候选事件是否已经穷尽；
- 页面是否在主动操纵它。

因此需要显式 belief state，而不是把完整聊天历史当状态：

```text
候选假设及置信区间
决定性 claim 状态
独立证据与冲突
未满足 evidence slots
工具健康状态
剩余预算
最近动作的边际收益
```

### 5.5 动作空间过大

自然语言 query、连续 crop 坐标、工具和参数构成巨大动作空间。建议分层：

```text
RL 高层策略：
  选择调查目标
  选择动作类型
  选择探索 / 验证 / 反证 / 停止

SFT 低层执行器：
  生成具体 query
  生成 crop box
  生成页面抽取参数
```

## 6. 数据构造问题

### 6.1 轨迹不能只有工具序列

普通数据：

```text
问题 → 若干搜索 → 最终答案
```

所需数据：

```text
当前 belief
→ 候选动作及选择理由
→ 实际 observation
→ 哪个 claim / hypothesis 被更新
→ 更新前后状态
→ 下一步还缺什么
```

建议单步记录：

```json
{
  "state_id": "S4",
  "open_hypotheses": ["H1", "H3"],
  "target_claim": "C2",
  "missing_slot": "original_event",
  "candidate_actions": [
    "crop_ris(sign_region)",
    "search_web(ocr_text)",
    "search_counterevidence(event_A)",
    "stop"
  ],
  "chosen_action": "search_web(ocr_text)",
  "tool_result_id": "T17",
  "new_evidence_ids": ["E8"],
  "belief_update": {
    "H1": "0.55 -> 0.15",
    "H3": "0.30 -> 0.72"
  },
  "coverage_delta": 0.18,
  "decision_relevance": "high"
}
```

`candidate_actions` 很重要：只有 chosen action 通常只能训练行为克隆，难以学习相对行动价值。

### 6.2 必需数据类型

#### 成功轨迹

搜索产生新的视觉关注，重新观察发现决定性证据，并形成正确裁决。

#### 失败轨迹

至少覆盖：

- 过早停止；
- 在错误候选上不断深挖；
- 重复同一来源族；
- 将 snippet 当正文；
- RIS 无结果后错误下结论；
- 正确工具动作后错误整合；
- 工具失败后伪成功；
- 忽略反证；
- 为提高 recall 低置信广撒网。

#### 成对反事实轨迹

```text
相同 state
├─ A：继续宽泛搜索 → 重复网页，不改变裁决
└─ B：裁剪路牌并 OCR → 找到地点，排除主要候选
```

还应构造：

- 搜支持证据 vs 搜反证；
- 整图 RIS vs 关键区域 RIS；
- 第十个转载网页 vs 首发来源；
- 继续搜索 vs 正确停止；
- confident verdict vs 正确 `unverifiable`。

#### 工具故障轨迹

对同一 case 注入：

- RIS timeout；
- 空搜索结果；
- 页面 403；
- OCR 关键字符错误；
- snippet 与正文不一致；
- 只有转载、原始来源缺失；
- 网页 prompt injection；
- benchmark answer contamination。

观察 Agent 是否切换等价策略、降低 belief、保留缺口、拒答，并且不把失败写成成功。

#### 时间与来源依赖数据

- 同一图片在多个时间被重新配文；
- 多网页复制同一社交媒体 post；
- 辟谣出现前后的 RIS 结果不同；
- 原始页面删除只剩转载；
- 新旧事件视觉相似；
- 裁剪版本隐藏关键上下文。

### 6.3 三层数据体系

#### 第一层：可控合成 Search World

从已知 evidence graph 构造：

```text
事件
├─ 正确首发来源
├─ 两个独立支持来源
├─ 一个合理反证
├─ 三个转载来源
├─ 一个恶意网页
└─ 若干无关相似图片
```

优点是可知道完整覆盖、来源依赖、动作价值和正确停止点，适合 SFT、process reward 与 RL。

#### 第二层：真实网页冻结快照

从 AVerImaTeC、人工事实核查文章和真实新闻保存：

- SERP；
- 网页正文与图片；
- RIS 结果；
- 时间戳和内容 hash；
- source family；
- gold claim–evidence graph。

#### 第三层：Live Web

只用于环境迁移、最新事件、工具故障、搜索引擎差异以及真实成本/延迟评测，不把单次 live 结果当稳定 reward。

## 7. 建议训练路线

### Phase 1：无 RL 的完整调查 Agent

用 prompting/SFT 实现 hypothesis state、四本账、主动视觉工具、证据更新与 deterministic gate，先获得失败分布。

### Phase 2：候选动作生成与排序

给定状态生成 3–5 个动作，以 pairwise preference、DPO 或 contextual bandit 学习：

```text
V(state, action)
≈ 动作对最终“正确且有证据的 verdict”的贡献
```

### Phase 3：冻结 Search World 的分层 RL

RL 只控制：

```text
调查哪个 claim
选哪类工具
探索 / 验证 / 反证
继续 / 停止
```

query 与 crop 参数仍由 SFT executor 生成。

### Phase 4：有限在线优化与真实评测

以缓存回放、少量在线 rollout 和拒绝采样减少 live web 方差，最后在未见 rolling cases 上测试。

## 8. 初步奖励设计

终局奖励建议采用 gate 或乘法关系：

```text
R_terminal =
  verdict_correct
  × decisive_evidence_sufficient
  × citation_sound
  × tool_integrity
```

猜对标签但证据不充分，不能获得满分。

过程奖励候选：

```text
R_process =
  Δ decisive-slot coverage
+ Δ hypothesis discrimination
+ 新独立高质量证据
+ 发现有效反证
+ 正确识别工具失败
+ 正确停止或拒答
− 重复来源
− 无效工具调用
− 未验证内容进入 belief
− 隐藏冲突
```

不能直接奖励模型自报的置信度变化；应依据 gold evidence graph 或经验证 ledger 的客观变化。

## 9. 最重要的研究命题

> RL 不是用来教 Agent“事实是什么”，而是教它在不完全信息下，如何用有限预算选择最有裁决价值的下一次观察。

可能形成三层贡献：

1. **系统层**：可审计的 claim / hypothesis / evidence / source / failure state；
2. **学习层**：decision-relevant active investigation policy；
3. **数据层**：带反事实分支、来源依赖、时间变化和工具故障的多模态调查轨迹。

## 10. 下一步待讨论问题

1. 输入是否总包含外部 textual claim，还是有时只有图片？这决定 claim induction 的任务定义。
2. `real` 的产品语义是否应改为 `supported`，避免与像素原生性混淆？
3. 第一版 belief state 是规则结构、模型生成 JSON，还是独立状态更新器？
4. 哪些视觉操作进入 RL 高层 action space，哪些固定由 executor 完成？
5. 能否为首批 200–500 个 case 建立可用的 gold evidence graph 与 source family？
6. 第一篇论文主打 coverage/stopping、active visual investigation，还是数据集与轨迹训练？

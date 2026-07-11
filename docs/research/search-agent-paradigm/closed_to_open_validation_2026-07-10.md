# 从闭合训练环境到开放互联网能力：论证边界与评测协议

日期：2026-07-10

## 核心结论

闭合/冻结环境不能单独证明开放互联网能力。它的作用是提供：

- 可复现训练；
- 可计算过程奖励；
- gold coverage 与正确停止状态；
- 故障、冲突和反事实动作的受控干预；
- 检索器、策略与生成器的可分离诊断。

开放互联网能力必须通过严格隔离的真实网页和 live-web 测试另行证明。推荐证据链：

```text
Tier 1  Synthetic Evidence Capsule
  学习策略、过程监督、因果诊断
        ↓
Tier 2  Frozen Real-Web Snapshot
  测真实网页噪声与可复现泛化
        ↓
Tier 3  Live-Web Hidden / Rolling Test
  测开放互联网零样本迁移、时效与工具鲁棒性
        ↓
Tier 4  Cross-backend / Temporal Replication
  测是否过拟合搜索引擎、日期或地域
```

论文只能根据通过的层级做相应强度的声明。

## 相关工作的实际做法

| 工作 | 受控/冻结部分 | 开放或真实验证 | 能支持的结论 | 不能单独支持的结论 |
|---|---|---|---|---|
| [WebShop, 2022](https://arxiv.org/abs/2207.01206) | 模拟电商站，含 118 万真实商品；IL/RL 训练 | 100 个指令零样本迁移到 Amazon/eBay，人工评分 | 模拟中学到的部分搜索/导航行为可迁移 | 不能证明任意网站或任意开放任务泛化；真实成功率仍远低于人类 |
| [DeepResearchGym, 2025](https://arxiv.org/abs/2505.19253) | ClueWeb22/FineWeb 固定大规模网页索引 | 统一在 Serper live web 上评测闭合训练和商业搜索训练的 Agent；GAIA/HLE 等出现可比增益 | 固定真实网页语料可用于低成本训练，部分策略跨引擎迁移 | 不是小型合成世界；不能保证所有任务或视觉 RIS 迁移 |
| [Deep Research Bench, 2025](https://arxiv.org/abs/2506.06287) | RetroSearch 冻结抓取网页 | 将 offline RetroSearch Agent 与 live-web Agent 比较并报告相近表现 | 对这 89 个任务，冻结回放可作为稳定近似 | 不代表训练出的策略必然泛化到未来事件或其他搜索分布 |
| [BrowseComp-Plus, 2025](https://arxiv.org/abs/2508.06600) | 固定 corpus、人工支持文档、困难负例 | 主要价值是与原 live BrowseComp 形成可控补充 | 可分离 retriever、Agent 和 context engineering | 固定 corpus 高分不能直接等同开放互联网能力 |
| [ProMMSearchAgent, 2026](https://arxiv.org/abs/2604.20486) | 本地静态 corpus + 预计算多模态 cache 做 RL | 测试时零样本切换 live Google Web/Lens；做 search-backbone swap；本地训练保留在线训练 96% 以上表现（论文报告） | 最接近严格 sim-to-real：同一策略跨本地/Google 工具后端迁移 | 仍只覆盖选定知识型 VQA benchmark；需要独立复现，且是预印本 |
| [SearchEyes, 2026](https://arxiv.org/abs/2607.05943) | typed KG 构造 simulated search world 与 hop reward anchors | 在六个 held-out 多模态 benchmark 测试 | 学到的多跳视觉—知识结构可跨任务集泛化 | 若没有 live search 对照，不能仅凭 held-out benchmark 宣称开放 Web 鲁棒性 |
| [Mind-ParaWorld, 2026](https://arxiv.org/abs/2603.04751) | 平行世界 law model 与模拟 SERP | 目的主要是无泄漏、可归因评测 | 可诊断 coverage、sufficiency、stopping | 明确不是开放互联网能力证明 |
| [DeepResearcher, 2025](https://arxiv.org/abs/2504.03160) | 对照 local RAG 训练 | 直接在真实网页环境做 RL 与 OOD 测试 | 提供“真实环境训练可能超过闭合 RAG 训练”的反方证据 | live RL 高成本、高方差且复现困难；不能提供精确 gold 过程奖励 |

## 对本项目的设计修正

不应选择“全闭合”或“全 live”二选一，而应使用混合训练：

```text
SFT / 动作预训练：
  合成 Evidence Capsules + 真实冻结轨迹

RL 初期：
  Evidence Capsules，获得精确过程 reward

RL 后期：
  大规模真实网页快照 / cached real interactions
  少量 live-web on-policy rollouts

最终测试：
  绝不使用合成 Capsule 作为唯一结论
  必须包含 frozen real-web + hidden live rolling cases
```

## 推荐的数据占比（起点，不是固定配方）

```text
训练：
  40% 可控合成 Capsules / counterfactual branches
  40% 冻结真实网页与真实图片轨迹
  20% live/cached-live 轨迹或策略适配

开发：
  合成诊断集 + 冻结真实网页集

最终测试：
  0% 训练模板复用的合成 IID 样本作为主结论
  50% 冻结真实事件（可复现）
  30% 隐藏 rolling live claims（开放 Web）
  20% 跨搜索后端/跨日期重复测试
```

如果真实数据不足，可先改变比例，但必须分别报告，不得用合成综合分代替 live-web 分数。

## 开放互联网评测协议

### 1. 任务选择

- 来自评测开始前 7–30 天的新事件/新谣言；
- 在模型训练数据和合成模板之外；
- 由事实核查员先建立 claim atoms 和证据包；
- 包含跨语言、低资源来源、RIS data void 和页面删除案例；
- 不公开问题或答案，防止 search-time contamination。

### 2. 运行日志

每次运行保存：

- 时间、地区、搜索引擎/API 版本；
- 每次 query、SERP、rank、URL；
- 页面与图片快照/hash；
- 工具失败；
- claim–evidence edges；
- token、成本和延迟。

### 3. 多后端复测

至少选两个文本搜索后端；视觉搜索若无法获得两个供应商，则对同一图使用整图、crop 和时间重复检索，并明确限制。

### 4. 时间复测

同一批 claim 在 T0、T+7、T+30 重跑，测：

- verdict stability；
- evidence turnover；
- citation survival；
- RIS result drift；
- data void 随时间的变化。

### 5. Sim-to-real 关键指标

不能只看最终 accuracy，还应测策略迁移：

- 决定性 claim 的 evidence recall；
- 搜索后返回正确图像区域的比例；
- query specificity / reformulation gain；
- 新独立来源族发现率；
- premature stop / over-search；
- 工具失败后的恢复与正确拒答；
- synthetic→frozen-real→live 的性能下降幅度。

## 必须做的训练环境消融

以相同模型、数据规模和预算训练：

```text
A  仅合成 Capsule
B  仅冻结真实网页
C  合成 + 冻结真实
D  合成 + 冻结真实 + 少量 live adaptation
```

全部在同一 hidden live-web test 上评估。

只有当 A 或 C 在 live test 上超过无训练/SFT 基线，才能证明闭合训练策略迁移；只有当 C/D 明显超过 A，才能说明真实网页噪声暴露不可替代。

## 论文声明边界

| 已完成实验 | 可以声称 | 不应声称 |
|---|---|---|
| 只在 Evidence Capsule 测试 | 学会了受控环境中的主动调查、coverage 或 stopping | 具备开放互联网事实核查能力 |
| Capsule 训练 → frozen real-web 测试 | 对真实网页快照存在零样本迁移 | 能处理实时开放 Web |
| Capsule/real frozen 训练 → hidden live test | 对选定真实任务和工具后端具有开放 Web 迁移 | 对所有互联网、语言、日期和搜索引擎普遍可靠 |
| 再加跨后端/跨日期/跨语言 | 对环境变化有更强鲁棒性证据 | 绝对完备、不会漏证或不会受攻击 |

## 最适合本项目的研究故事

不是：

> 我们在闭合世界训练和测试一个 Agent，因此它会搜索互联网。

而是：

> 我们用可验证 Evidence Capsules 学习无法从 live web 获得的过程级调查策略，再通过冻结真实网页、隐藏 live-web rolling claims、跨搜索后端和时间复测，测量这种策略的 sim-to-real transfer。我们同时量化迁移落差，明确哪些能力能迁移、哪些仍需真实环境暴露。

# Search Agent 过程评分与 Credit Assignment 核查

本页核对 Search、Web、Tool、Multimodal Agent 怎样评价调查过程。结论以三个层级
分别记录：论文声称、公开源码实现、发布训练配置是否真实启用。只满足其中一层，
不能写成“已有可运行训练方案”。

## 先纠正上一版结论

上一版只核对了 Search-R1、ReCall、DeepResearcher、rLLM、WebAgent-R1 和
R-Search，因而把第一代 Search-RL 的 outcome-only 特征错误外推成整个领域的现状。
扩展核验后的结论是：

- 第一代公开 Search-RL 的确以终局答案 reward 为主；
- 近期过程 credit assignment 已成为一条重要主线，并有多种真实训练实现；
- “过程评分”不是单一做法。检索覆盖、状态可回答性、反事实归因、图状态传播、
  教师/PRM 逐步判断和 outcome advantage 重分配解决的是不同问题；
- R-Search 的 selected-Evidence verifier 仍有价值，但它只评价终局证据包，不是最接近
  逐 turn 过程监督的唯一实现。

## 六种概念不能混用

| 名称 | 实际含义 | 是否直接给单步 policy 梯度 |
|---|---|---|
| terminal outcome | 只检查最终答案、任务状态或 verdict | 否，通常整轨迹共享 |
| trajectory process quality | 对完整轨迹的证据性、rubric 或风格给一个分数 | 否，仍是整轨迹 scalar |
| turn reward | 每个 action-observation transition 有独立 reward | 取决于 advantage 实现 |
| segment/token advantage | 把不同 credit 写到对应 action segment/token | 是 |
| inference PRM | 推理时给候选下一步排序、剪枝 | 否，除非另行蒸馏/训练 policy |
| process benchmark | 测 PRM 能否区分正确/错误工具步骤 | 否 |

## 已核对的训练实现

### 1. Outcome-only 基线仍然存在

| 项目 | 核对提交 | 实际 reward | 落点 |
|---|---:|---|---|
| Search-R1 | `598e61b` | 最终答案 EM；公开 format recipe 的 retrieval 分为 0 | 最后 response token |
| ReCall/ReSearch | `aaf16b3` | 最终答案 token-F1；格式正确但答案错误给 0.1 | 终局 response |
| DeepResearcher | `82c6dc2` | 最终答案 F1；格式错误为 -1 | 终局 response |
| rLLM | `cd9ea0e` | 内置 SearchReward 为最终答案 EM/F1 | 默认向 action steps 广播轨迹 reward |
| WebAgent-R1 | `354f5c1` | WebArena evaluator 的任务完成分 | rollout 后组内归一化 |
| R-Search | `d7038ca` | 答案 F1 + 格式 + selected Evidence 可回答性 | 终局 response |

这组项目只能证明第一代 recipe 大量使用稀疏 outcome reward，不能证明过程评分不重要。
Agent Lightning（`f0a77cf`）也说明了“框架能力”和“算法实现”的区别：它能记录并
关联中间 reward span，但官方 veRL adapter 当前仍把最终 reward 传播给前面的
triplet，并明确提示细粒度 credit assignment 需要定制算法。

### 2. 检索覆盖与结构锚点

**StepSearch**（论文 `2505.15107`，源码 `43215ba`）在每次搜索后计算：

- 返回文档相对 gold supporting documents 的 TF-IDF 覆盖增量；
- information gain 和重复检索惩罚；
- query 相对 gold search keys 的 F1；
- 终局答案 F1。

过程 reward 写在相应搜索段末端 token，发布的 `train.sh` 真实启用这些项，过程项
scale 为 0.6。这是可运行的逐搜索监督，但依赖 gold supporting documents 和 search
keys，无法原样搬到开放网页 IFV。

**SearchEyes/HaPO**（`d00a870`）利用合成数据保留的 gold entity chain：到达相同
gold entity 的轨迹组成 hop-anchor group，再根据后续 outcome 的组内差异给 anchor
之后的 step 分配 advantage。`scripts/training/run_rl.sh` 启用 `adv_estimator=hapo`，
`searcheyes/hapo.py` 实现 token-level advantage 和 fatal-aware masking。它证明结构化
训练数据可以提供强过程锚点，但 IFV 开放网页没有天然 gold entity chain。

### 3. 状态可回答性与 likelihood potential

**TIPS**（论文 `2603.22293`，源码 `c8687e3`）在每个 tool response 后估计 gold
answer likelihood potential，以 `Phi(s_t)-Phi(s_{t-1})` 作为过程 reward；终局再减去
最终 potential，保持 potential-based reward shaping。发布脚本真实启用
`info_reward_llm`。

**OASES**（论文 `2604.03675`）让当前 policy 根据每个中间搜索状态生成答案，用与
终局相同的可验证指标评分，再以相邻状态答案分的增量作为过程 reward；搜索 policy
和 state evaluator 共享参数并共同训练。论文给出 PPO 接线、默认过程权重 0.5 和
消融，但本次未取得可稳定定位的公开仓库，因此只记为“论文级强证据”，不写成已核验
源码 recipe。

**TRACE**（论文 `2607.13988`）使用冻结 reference model 计算每个工具边界上的 gold
answer log-probability，经 log-ratio state value 和 TD 差得到 turn credit，再与 outcome
advantage 联合。论文记录了真实 launch 参数和开启远程 reference scorer 的方式，但
未找到公开代码仓库，当前同样只记论文实现。

这条路线的共同优点是 reward 与最终答案可回答性紧密相关；共同限制是需要 gold
answer，并可能把模型对 gold 的先验熟悉度误当作证据进展。IFV 的二元 verdict 过于
粗糙，不能只比较 `real/fake` 两个 token 的 likelihood。

### 4. 反事实贡献归因

**LOTAPO**（论文 `2607.13501`，源码 `ea2f01b`）对完整轨迹逐 turn 做 backward
leave-one-turn：删除当前 action 和 observation，比较删除前后 gold-answer
log-likelihood。归因经组内稳健缩放、tanh 和 sign gating 后，只加到对应搜索 turn；
全体 token 仍保留 outcome advantage。论文报告额外开销约 3.7%，发布源码实现了该
训练路径。

**TACO**（论文 `2606.30251`）面向视觉 code-tool agent，在工具调用前后各探测一次
答案，用 outcome 差判断该调用是 useful、redundant、misleading 还是
necessary-but-failed，再用 Outcome-Gated Advantage Routing 把终局 advantage 送到
负责的 token segment。它很适合解释 IFV 中“重新看图是否真正改变了判断”，但本次
未找到公开训练仓库，不能把论文公式当作现成依赖。

反事实路线比“看起来像好搜索”更接近真实贡献，但删除一个 turn 会改变后续上下文，
不是严格因果；开放网页轨迹还要处理 observation 缺失后后续动作失去语义的问题。

### 5. 图状态和同状态相对 credit

**RewardFlow**（论文 `2603.18859`，源码 `cd06546`）把多轨迹的工具边界组成 state
graph，合并相同或相似 state，从终局 reward 向前折扣传播 state value，再以
`V(s_next)-V(s_prev)` 作为 turn reward。真实脚本启用
`algorithm.adv_estimator=rewardflow`。它不需要逐步教师或 gold supporting docs；
难点是开放文本、多模态调查状态怎样可靠判为语义等价。

**GDCR + SAPO**（论文 `2605.29697`）根据训练期 Entity-Relation graph 中新检索/
引用实体到 answer node 的距离计算 step progress，并用
`A_t = A_outcome + lambda * |A_outcome| * A_step` 注入对应 step。没有找到公开仓库，
更适合启发合成训练集，而不是直接用于真实网页。

### 6. 教师或 PRM 的逐步语义评分

**CW-GRPO**（论文 `2604.14267`，源码 `c9d5ab3`）是对 IFV 最直接的训练先例：

- LLM judge 读取每轮 partial trace；
- `retrieval_reward` 判断检索是否相关且带来新信息；
- `thinking_reward` 判断推理是否被已有证据支持、下一动作是否有用；
- 只对 outcome 正确轨迹调用 round judge；
- `retrieval * thinking` 经 softmax 后重分配正确轨迹的正 advantage；
- 错误轨迹仍保留均匀负 outcome advantage。

训练脚本真实启用 `algorithm.adv_estimator=cw-grpo` 和 `reward_manager=cw-grpo`。
这说明 Gemini 可以做逐 turn 教师，但更稳妥的用法是分配 outcome credit，而不是让
主观过程分脱离任务结果成为唯一优化目标。

**FaithMed**（论文 `2607.01440`，源码 `095f4e6`）把 Gemini 接成逐步 rubric
judge，分别记录 process reward 和 outcome reward，并在 GiGPO 中形成 step return。
论文推荐命令显式开启 `process_reward_enable=1` 和 `step_scoring=True`；默认脚本中
该功能可关闭。当前实现的 step judge 主要读取此前检索材料和当前 action，评价当前
推理/搜索计划是否合理；当前 action 刚返回的 observation 留给下一步评分，因此它不
等同于“本次网页实际带来了多少新证据”。

**PiCA**（论文 `2605.09287`，源码 `35b15c7`）由强教师标注 pivot step 的 gold
sub-query/sub-answer，训练 Qwen2.5-3B PRM 预测历史条件下的成功概率，再以相邻
turn 的 log success-probability ratio 形成 reward。训练器、remote RM 和 scale=0.3
的接线真实存在；但 README 说明关键 PiCA 数据尚未发布，示例仍有占位路径，因此
属于“算法接线可核验，release 尚不能即下即跑”。

### 7. PRM、偏好学习和全轨迹评分

这些项目重视过程，但不能误写成“逐 turn reward 已直接进入 policy RL”：

- **PRA**（`cf9c93d`）：强教师用 gold answer 和文档标注 reasoning correctness，训练
  Qwen3-4B PRM；部署时 PRM 不看 gold，用于 beam pruning 和 search dependency。
- **PRInTS**（`be455be`）：用 Monte-Carlo 后续 rollout 的成功率变化标注当前步骤的
  information gain，训练 Qwen3-4B generative PRM；PRM 同时递归压缩历史，并在
  推理期对候选下一步做 Best-of-N 排序。公开了标注、SFT/GRPO 和评测代码。
- **SWEET-RL**（`38daf2a`）：用成功/失败轨迹和带最终答案 privileged information 的
  step RM 生成逐步 preference，再以 DPO 训练 policy。
- **ARBOR**（论文 `2606.03239`）：从成功/失败轨迹诱导动态 rubric，LLM 对完整轨迹
  给 process-quality scalar；它能细排 outcome 相同的轨迹，但不是 turn attribution。
- **ToolPRMBench**（`b43164f`）：提供工具 history、正确 action、迷惑错误 action 和
  ToolPRM 权重，主要是过程模型 benchmark，不是完整 RL 训练实现。

## 对 IFV 的直接结论

### 当前代码还没有过程 credit

`ifv_training/rewards.py` 当前把 outcome、blind/aware semantic judge 和 strict audit
组成一个 episode scalar；`export_framework_reward()` 只把它写到 terminal step/token。
也就是说，现有 scorer 能评价终局结果和证据包，但尚不能回答：

- 哪一次搜索发现了决定性事实；
- 哪一次 observation 只是重复；
- 哪一步根据新证据重新理解了图像；
- 哪一步从已有正确方向回归到错误结论；
- 错误轨迹的前半段是否其实有价值。

### Gemini 应承担的角色

Gemini 应是冻结的、rollout 后异步运行的**过程教师和校准器**，不是 Qwen 轨迹的
唯一生成器，也不是最终真值源。评分单位应是可审计的
`pre-state + action + observation + post-state delta`，而不是不可验证的隐藏 thinking。

每个 turn 至少分开记录四类短标签：

1. `evidence_gain`：是否新增可用且与命题有关的材料；
2. `direction_quality`：当前调查动作是否针对仍未解决的关键不确定性；
3. `belief_update`：新材料是否被忠实吸收，含重新理解图像或修正假设；
4. `regression`：是否重复、忽略已有证据、过度解释无结果或把调查带偏。

教师必须返回引用的 Evidence/turn ID 和低中高置信度；无法引用输入材料的分数不能
直接进入 optimizer。Prompt 保持一个任务说明、一组字段和一个 JSON schema，不加入
案例专用的 if/then 规则。

### 推荐的双通道过程 credit

```text
终局锚点
  verdict correctness
  selected-basis recoverability
  fatal engineering mask

过程通道 A：Gemini 短 rubric
  评价调查方向、证据增量、belief update、regression

过程通道 B：反事实/状态差
  删除或遮蔽该 turn 接受的证据，重算 basis recoverability
  或比较该 turn 前后的命题可裁决性
```

初始优化采用 outcome-gated redistribution：

- 正确轨迹的正 advantage 按经过校准的 turn contribution 分配；
- 错误轨迹保持总体负方向，但把主要负 credit 集中到 regression/harmful turns，
  有价值的早期调查只减轻负值，不立即翻成正 reward；
- 只有 rubric 与反事实通道一致时，才提高 contribution 置信度；
- 工程错误后的 suffix 全部 mask；格式只做门禁；搜索次数、长度和成本先只记录。

这样既保留 outcome 的可验证方向，也能学习 Queen 失败轨迹中真正缺少的能力：从
“查是否坐过 bus”扩展到“该事件中实际交通工具是什么”，并在 Gold State Coach
证据出现后更新对图像争议点的理解。

## 实施前必须做的排序校准

1. 冻结 Qwen checkpoint、runtime、采样参数和工具环境，每题采样 K=4–8 条轨迹。
2. 人工逐 turn 标出 useful、neutral、redundant、misleading、regression。
3. 同时跑 Gemini rubric、反事实 contribution 和终局 selected-basis recoverability。
4. 分别测 turn-level macro-F1、同题 pairwise ranking、一致率、置信度校准和教师 token。
5. 先离线回放 advantage 分布，确认没有奖励搜索长度、措辞或“看起来像反思”的行为。
6. 通过后再接 optimizer；冻结 20 例仍只用于开发评测，不进入训练 group。

## 主要来源

- StepSearch: https://github.com/Zillwang/StepSearch
- TIPS: https://github.com/ucsd-wang-lab-lm/tips
- LOTAPO: https://github.com/zhuq-111/LOTAPO-Leave-One-Turn-Attribution
- RewardFlow: https://github.com/tmlr-group/RewardFlow
- CW-GRPO: https://github.com/zsxmwjz/CW-GRPO
- FaithMed: https://github.com/cxcscmu/FaithMed
- PiCA: https://github.com/novdream/PiCA
- PRA: https://github.com/eth-medical-ai-lab/pra
- PRInTS: https://github.com/G-JWLee/PRInTS
- SWEET-RL: https://github.com/facebookresearch/sweet_rl
- SearchEyes: https://github.com/Frostlinx/SearchEyes
- ToolPRMBench: https://github.com/David-Li0406/ToolPRMBench
- Agent Lightning: https://github.com/microsoft/agent-lightning

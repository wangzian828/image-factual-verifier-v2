# Visual Fact Discrepancy Agent v4 实施计划

日期：2026-07-17
状态：执行中
设计合同：`2026-07-17-discrepancy-first-v4.md`
交互顺序：`docs/gemini-interaction-sequence.md`

> 本文保存用户确认的独立执行计划。详细服务器命令与验收边界以本次接手消息为准；实施时不得跳过确定性门禁直接调用真实 Gemini。

## 当前执行进度（2026-07-20）

- Phase 0～7：本地实现完成。默认 Agent workflow policy 已切换为
  `discrepancy-first-v4`；数据管线 release manifest 继续声明其独立协议 `reinspect-v2`，
  不作为 Agent selector。公开 `Orchestrator.run()` 不再执行 v3。冻结 v3 仅由测试专用回放
  harness 调用旧 schema/reducer。
- Phase 1 确定性门禁：原子 Planning/Decision reducer、非法输出零修改、稳定 ID、
  序列化往返、Evidence/Claim/Hypothesis 所有权、单次视觉复查和 post-verdict 禁止均有测试。
- Phase 2～5：Image Account Planning 受控原图 standalone request、claim/hypothesis ReAct 短链、稀疏
  Discrepancy Decision、v4 Coverage/basis/Judgment 已通过完整 mock Interactions 轨迹和
  默认 workflow canonical trace strict audit。
- Phase 6：v4 strict audit、`ifv-policy-v2`、Evidence-chain/discrepancy/stop-quality
  scoring 和训练排除门禁已实现。
- Phase 8：完成。已从 gpu-13 只读定位并白名单脱敏四条冻结 v3 成功轨迹；回放 fixture
  不含凭据、private gold 或 provider interaction ID。当前 v4 reducer 对 Andreea、Queen、
  Pillars 和 Monarch 双跑状态完全一致，预期 verdict、Evidence 所有权、discrepancy 对齐、
  及时停止、post-verdict 禁止和 strict audit 均通过。
- Phase 9：执行中。四条历史回放门禁已放行；2026-07-20 Queen 真实 canary 已得到
  evidence-determined `fake`，Evidence/Discrepancy/停止/训练门禁均为 1.0，0 次协议拒绝。
  14 次请求累计 prompt token 为 490,691，最大单请求约 41.4k，低于 128k；累计值不能误作
  单请求上下文。canary 外层 strict audit 失败已定位为递归误扫 `traces/runtime` 中的档案 JSON，
  现已限制为直接 `traces/*.json` 并补回归测试，待服务器复验。
- Phase 10：核心文档已更新；v4 主实现已提交。
- Phase 11～14：上下文测量、不可变档案、显式工作区、按需回读、图像分辨率控制和调查后
  图像重检已实现。
- Phase 15：进展记账保留；2026-07-20 按最新决策删除无进展 soft checkpoint、grace actions
  和 `information_saturated` 提前结算。当前本地门禁为 `422 passed`、`compileall` 通过、
  `git diff --check` 通过。下一步在 gpu-13 复验 Queen strict audit，再运行 Monarch、Pillars、
  Andreea 三条异构真实样例。
- Phase 16：Queen 已通过；Pillars、Monarch、Andreea 首轮均在 Discrepancy Decision 的合法
  原子更新门禁失败。当前修复保持通用：neutral Evidence 显式不可定向，Task ownership 与
  Evidence 的实际语义覆盖分离，网页 Evidence 不再确定性排除 mixed task 中的 `text_claim`，
  Decision 上下文显式列出每个 Claim 的合法视觉锚点与已审阅定向链。未加入样例专用规则；
  本地更新后门禁为 `424 passed`，下一步重新运行三条真实 canary。
- Phase 16 复验：Pillars `real` 与 Monarch `fake` 已真实通过 strict audit。Andreea 得到
  `fake`，但 21-action bounded Judgment 没有 Evidence，且 strict audit 发现一次 URL 与 Task
  的跨归属组合。后续通用修复把每轮 ReAct 限为一个 task-scoped route family；网页访问由
  模型从该 Task 拥有的 Claim ID 中选择单一 stance 目标，runtime 注入 Claim 原文，同时保留
  action 自己的 passage retrieval goal。修复后只重跑 Andreea，再决定是否进入正式 20 例。
- Phase 16 第二轮：单 Claim 网页取证修复后的 Andreea 真实 canary 得到 evidence-determined
  `fake`，strict audit 通过，0 次工程错误和协议拒绝；工具调用从 21 降至 16，LLM 调用从 42
  降至 30。首个 Decision 已取得 Esca 本人“从未服用或推荐减肥产品”的直接反证，却把已
  refuted 的高显著 Claim 降为 supporting discrepancy 并继续寻找另一 Claim 的原图。当前
  通用修复要求高显著 Claim 的合格 refutation 必须形成 decisive discrepancy 和 `fake`，其他
  未解决 Claim 不降低该反证；失败输出仍由原子 reducer 拒绝，不新增样例、人物或产品规则。
- Phase 16 第三轮：真实运行仍得到 `fake` 且 strict audit 通过，但属于 15-action bounded
  Judgment，Evidence 链未进入 verdict basis。根因是 Planning 把 Claim 写成“广告声称 Esca
  代言”，使 Esca 本人否认代言的原文在字面上支持“广告存在这种虚假声称”，而不是反驳
  待核查的现实命题。当前通用修复只明确 ImageClaim 必须直接表达图像让读者相信的现实命题，
  不得退化为“图片、文字、帖子或广告作出了该声称”的元命题；不修改抽取器 stance，不使用
  字符串门禁。
- Phase 16 第四轮：现实命题 Planning 已真实生效。Andreea 在 5 个调查动作后得到
  evidence-determined `fake`，Evidence chain、discrepancy alignment、Decision/basis 一致性和
  stop quality 均为 1.0，strict audit 通过，0 工程错误、0 post-determination action；整案为
  7 次工具、13 次 LLM、181.73 秒。照片真实性 Claim 仍未解决，但代言现实命题已有决定性
  refutation，因此没有继续做完整来源复原。网页抽取首次把 Esca 本人的否认误标为 support，
  造成一次可恢复的 protocol correction；20 例需继续统计该 stance 噪声和训练排除比例。
- 2026-07-22 Qwen3.5 更新：正式 Queen 证明 Planning 失败来自无界 Thinking，而非输入
  上下文。已按官方采样建议选择 Planning 1,024-token reasoning wall，并用简短字段签名让
  模型看见输出结构；相同输入从 5 分钟超时降至 17.4 秒，同时产生现实命题和开放交通工具
  调查方向。reasoning 将独立归档且不进入后续请求。完成本地门禁后重跑 Queen，零工程错误
  后再运行冻结 20 例。

### 2026-07-20 决策更新

Phase 0～10 保留为重构历史，但下面四项新决定覆盖旧计划中的冲突条款：

1. 最终事实结论固定为 `real | fake` 二元出口；`supported | refuted |
   conflicted | insufficient | unclear` 仅作为 Claim/Evidence 的内部评估状态，
   不再产生 `unverifiable` 最终 verdict。
2. 不再把全案 Gemini `previous_interaction_id` 隐藏历史当作调查记忆或事实来源。
   原始材料进入不可变调查档案，每轮由显式工作区和按需回读材料构造输入；
   `previous_interaction_id` 只保留在一次原生工具调用所需的短链内。
3. 保留确定性的“有意义路线耗尽”停止作为兜底；取消所有由连续无实质进展触发的
   提前结算和 `information_saturated`。进展记账只用于诊断、评估和训练评分，不影响
   v4 控制流。调查仅因证据结论成立、路线账本确实耗尽或 24-action 安全上限而结束。
4. 先完成上下文管理和停止机制。该项原定“先跑 Gemini 20 例再部署 Qwen”的顺序，
   已被下面 2026-07-21 的 Qwen 部署决策替代。

### 2026-07-21 Qwen 部署决策

Qwen 部署、SFT 与 Agent RL 的唯一执行主计划为
`2026-07-21-qwen3-vl-8b-deployment-and-training-infrastructure.md`。已确认：

1. 正式学生统一为服务器已有的 `Qwen3-VL-8B-Thinking`，不再执行 Qwen3.5 下载、
   4B 快速启动或 9B 迁移路线。
2. gpu-13 只使用物理 GPU `4,5,6,7` 中当时空闲的卡，最多四张，不占用 0～3。
3. 不先运行 Gemini 完整 20 例。完成 Qwen3-VL serving、v4 adapter 和四条 Qwen
   base canary 后，由 Qwen base 运行完整 20 例；相同冻结评测在 SFT、RL 后重跑。
4. 20 例是开发评测，不进入训练上下文；Gemini 已通过的四条 canary 保留为教师和基线。
5. SFT 以 ms-swift/DeepSpeed 的 Qwen3-VL-8B 全参数多模态门禁为首选；Agent RL 保留
   v4 runtime 环境所有权，以 rLLM/veRL gateway 为首选，不复制第二套状态机。框架必须
   经过 gpu-13 实测；不兼容时动态切换成熟替代框架，但不得降低契约和验收标准。

## 目标

将 v3 的单一核心事实核查运行时迁移为 v4 discrepancy-first 调查链：

```text
原图
→ Image Account
→ 1～3 条高显著性 ImageClaim
→ SearchHypothesis
→ 搜索与视觉 Evidence
→ Gemini 多模态 Discrepancy Decision
→ real | fake
```

核心问题：图像传达的主要事实中，是否存在由证据支持、与图像可见内容绑定的实质性错误或篡改？

调查可以保留不确定性和冲突，但最终 Judgment 必须根据完整调查账本给出更受支持的
二元结论、置信度和仍未解决的关键缺口。不能在零证据时随机默认某一类别，也不能把
“没有搜到图片中的声称值”直接当作反驳；应先检查是否还能通过同一事实槽位、相邻关系、
事件背景或重新观察图像取得能改变结论的材料。

## 强制边界

- v3 由 `runtime-v3-final-20260717` 冻结，不再修改。
- v4 分支：`codex/image-factual-verifier-v4`。
- 旧字段只作为迁移残留，不建立 v2/v3 runtime 兼容层。
- Prompt 保持简短，负责开放调查、语义判断和结构化输出；不得堆叠样例化的
  “如果……那么……”推理规则。确定性代码只负责协议、归档、ID、引用、预算、
  原子更新、进展记账和终止安全，不以关键词或字符串匹配代替语义调查。
- 搜索标题和 snippet 只是 Discovery；可追溯正文、同图比较等才是 Evidence。
- provider/协议故障是工程失败，不得伪装成事实结论。
- private gold 不得进入 runtime 或 Gemini 上下文。
- 不得通过关键词、固定 URL、固定查询或样例专用 Prompt 条款修 case。

## 实施阶段

### Phase 0：冻结迁移基线

- 盘点所有 `core_verdict_fact_id` 活跃依赖。
- 盘点 Target Planning、Evidence Decision、Reflection、Query Replan、Coverage、Judgment 入口。
- 保存 v3 历史 trace 为只读回放样本。
- 保留当前用户改动。

### Phase 1：原子状态 reducer

实现：

- `apply_image_account_planning(...)`
- `apply_discrepancy_decision(...)`

Planning reducer 必须验证 Claim key、视觉锚点、高显著性 Claim、Hypothesis 唯一性、可执行首跳工具和预算，创建稳定 ID 与 account-owned task，并在完整验证后一次性写入。初始 Hypothesis schema 不包含 Claim key 或逐 Claim 核查问题；reducer 写入的 account 归属只用于 Evidence、预算和停止账本，不构成语义判断。

Decision reducer 必须验证 Claim、Evidence、Hypothesis、视觉锚点及调查归属；验证新增 Hypothesis 和视觉复查预算；discrepancy 必须引用 Evidence；verdict 必须满足前置条件；完整验证后一次性提交。

退出条件：原子性、非法输出零修改、序列化往返稳定。在此之前禁止接真实 Gemini。

### Phase 2：Image Account Planning

- 用 `ImageAccountPlanningOutput` 替换 Target Planning。
- 原图以不可变图像资产保存；Planning 显式上传受控分辨率版本。
- 输入 Perception、OCR、VisualFact、retrieval anchors。
- 生成 1～3 条 ImageClaim；外部识别仅生成 SearchHypothesis。
- 后续阶段按需显式附加当前图像视图、crop 或参考图，不依赖全案隐藏历史继承。

### Phase 3：Claim/Hypothesis 调查循环

- ResearchTask 必须归属当前 image account 和 Hypothesis；该内部归属不得反馈成 Planning 的搜索语义约束。
- ReAct 每次只选择一个有限动作。
- 搜索发现可更新 Hypothesis，不自动扩大 ImageClaim。
- 禁止相同语义、工具、目标的重复路线。

### Phase 4：稀疏多模态 Discrepancy Decision

触发：直接支持/反驳 Evidence、同图或同事件比较、累计实质性 Evidence 的 checkpoint、unresolved 终止前。不得每条 Evidence 都调用。

Gemini 可更新 ClaimAssessment、建立 MaterialDiscrepancy、退休/新增有限 Hypothesis、请求一次聚焦视觉复查、提议 verdict。

### Phase 5：停止、Coverage 和 verdict basis

- `fake`：decisive discrepancy + 有效 Evidence + 高显著性 Claim + 视觉锚点，满足即停。
- `real`：所有 high-salience Claim supported；无 established/unresolved decisive discrepancy；Gemini 明确提议 real。
- `insufficient/conflicted/unclear`：保留为内部状态；有意义路线饱和后进入二元 Judgment，
  由模型结合现有正反证据、图像理解、未解决缺口和调查失败给出更受支持的结论。
- `meaningful_routes_exhausted`：继续保留为独立的确定性停止原因；所有已登记的有效路线
  已解决、失败、阻塞或有证据地耗尽，且无待回读关键材料或待执行图像重检时进入 Judgment。
- 保留 24 action、重复路线、Hypothesis、视觉复查预算和 post-verdict 禁止；24 action
  仅是安全上限，不应成为正常结束的主要方式。

### Phase 6：审计、评分和训练导出

- strict audit 验证 Claim—Discrepancy—Evidence 对齐。
- 导出 ImageClaim、SearchHypothesis、Decision checkpoint。
- 指标聚焦 evidence-chain recovery、discrepancy alignment、stop quality。
- 排除无效引用、无锚点、post-verdict、协议拒绝、状态不完整或证据结论不一致的轨迹。

### Phase 7：删除 v3 执行结构

从正式执行链删除：`core_verdict_fact_id`、单-core Coverage、Target Planning、Evidence Decision refinement、core replacement、旧 Reflection/replan、旧 compiled verdict 依赖。历史 schema 可只读保留，但不得进入 v4 主路径。

### Phase 8：历史 trace 确定性回放

- Andreea `case_8d9e67f836117760` → fake
- Queen `case_5cb541d78661ad17` → fake
- Pillars `case_e38eb5bcd2f433e4` → real
- Monarch `case_21ea1e62758d3d7c` → fake

回放不重新搜索，验证 reducer、Evidence 对齐、discrepancy、及时停止、post-verdict 禁止和 strict audit。

2026-07-18 执行结果：四条冻结语义/Evidence fixture 已迁移到当前 v4 reducer；每条 fixture
均执行两次并要求完整状态逐字段相等。四条新生成的 canonical v4 trace 均通过
`audit_real_trace.py --strict-scheduler`。旧 v3 JSON 不被伪装成 v4 trace；它们只作为只读
语义/Evidence 来源，因为早期产物尚未持久化当前 strict audit 所需的完整 Interactions 父链。

### Phase 9：真实 Gemini canary

先跑一条并人工审查完整 trace + strict audit；通过后再选 3～4 条异构 case，不直接跑全量 20 条。失败先分类为模型语义、查询规划、检索、抽取、视觉工具、reducer、停止、verdict、审计或基础设施。

2026-07-20 更新：既有真实运行暴露出全案隐藏上下文膨胀和饱和停止弱化；后续真实
验收由 Phase 11～16 接管，修复完成前不把旧 canary 结果视为最终验收。

### Phase 10：文档、清理和提交

更新 `docs/architecture.md`、`docs/agent-prompt-and-runtime-guide.md`、`docs/gemini-interaction-sequence.md`、`AGENTS.md`、README。最终运行：

```bash
python -m pytest -q
python -m compileall -q src scripts
git diff --check
```

单元测试通过不等于 v4 完成；还需真实 canary 和 strict audit。

## 下一阶段：上下文管理与信息饱和停止

### Phase 11：冻结测量基线与契约

先保留当前真实运行作为只读基线，不先改 Prompt 来掩盖问题：

- 记录每次 Gemini 请求的显式输入 token、provider 报告输入 token、图片数量与分辨率；
- 记录每案 action 数、连续无实质进展长度、跑满 24 action 的比例和结束原因；
- 记录关键支持、反驳、冲突 Evidence 是否在 Judgment 前仍可访问；
- 固化一条已观察到的异常：单轮显式 Judgment payload 约 3.7 万字符，但 provider
  报告输入约 46.9 万 token，证明全案隐藏继承不可继续作为默认上下文机制；
- 为工作区、档案、回读、进展事件和停止状态建立版本化 schema，先写序列化、引用完整性、
  非法更新零修改和工程错误隔离测试。

上下文管理必须有可解释的上下文账本，不能只记录一轮请求的总 token。每个请求生成
`ContextManifest`，至少记录：

```json
{
  "request_id": "req-042",
  "case_id": "case-001",
  "stage": "planning | react | decision | recall | judgment",
  "prompt_version": "v4-judgment-003",
  "model": "provider-model-id",
  "parent_interaction_id": null,
  "workspace_version": "ws-018",
  "archive_version": "archive-042",
  "explicit_input_chars": 37200,
  "explicit_input_tokens": 9100,
  "provider_input_tokens": 469168,
  "output_tokens": 1200,
  "image_count": 2,
  "image_tokens": 3400
}
```

其中 `explicit_input_*` 是本地序列化测量，`provider_input_tokens` 是服务端报告值；两者
必须同时保留，不能用估算值覆盖服务端值。若 provider 没有返回分项 token，则记录
`null` 和测量缺失原因。`parent_interaction_id`、请求阶段和 workspace/archive 版本必须
可用于判断是否发生了意外的历史继承。

每个请求再拆成有顺序的 `ContextItem`，记录实际进入模型的组成：

```json
{
  "context_item_id": "ctx-188",
  "request_id": "req-042",
  "kind": "system_prompt | task_prompt | workspace_state | recent_action |
           tool_result | evidence_excerpt | evidence_summary | failure_summary |
           open_question | hypothesis | image_asset | image_reinspection |
           recall_result | judgment_basis",
  "source_ids": ["evidence-17"],
  "text_hash": "sha256:...",
  "char_count": 1840,
  "token_count_estimate": 470,
  "priority": "decisive | conflict | open_gap | recent | background",
  "inclusion_reason": "supports_high_salience_claim",
  "retrieval_mode": "workspace | archive_recall | exact_read | inherited_protocol",
  "compressed": false
}
```

`ContextItem` 必须能反查到原始档案、工作区版本或图像资产；同一条目在多个请求中出现
时保留每次请求的实例，而不是只记录一个全局计数。这样可以拆出 Prompt、Evidence、工具
结果、失败摘要和图片各自的 token 成本。

退出条件：旧基线可复现；新契约不改变 Evidence provenance、工具结果和原始 trace。

### Phase 12：不可变调查档案

每次工具调用完成后，先持久化原始结果，再进行摘要或语义更新。每条档案至少包含：

```text
memory_id / evidence_id
case_id / action_id
query / tool / URL
原始网页、快照或工具 JSON
exact span / offsets / artifact hash / retrieval time
摘要及其 provenance
原图、crop、OCR、参考图的资产 ID 与路径
success | error
关联 claim / hypothesis / task
```

原则：

- 原始材料只追加、不改写；摘要错误不能污染原始档案；
- Discovery、Evidence 和 Failure 继续分账，向量或关键词召回结果本身不能升级为 Evidence；
- 确定性代码只验证保存成功、ID 唯一、引用存在、hash/offset 可复核、分页与预算，
  不通过字符匹配判断材料支持还是反驳；
- 工程错误进入 Failure/运行状态，不能进入事实 Judgment；
- canonical JSON 是事实记录，HTML 只用于诊断展示。

档案还必须保存上下文相关事件，形成可回放的 lineage：

`WorkspaceSnapshot` 保存每次 action 前后的主工作区版本，包括当前 ImageClaim、假设、
开放问题、高价值支持/反驳/冲突 Evidence 引用、待回读档案、待图像重检、路线状态、
剩余预算、连续无实质进展计数和本次 gain。每个 action 都关联
`workspace_before → action → archive_write → gain_assessment → workspace_after`。

`CompactionEvent` 保存压缩前后差异：

```json
{
  "compaction_id": "compact-007",
  "before_workspace": "ws-017",
  "after_workspace": "ws-018",
  "removed_item_ids": ["ctx-101", "ctx-104"],
  "retained_item_ids": ["ctx-102", "ctx-108"],
  "created_summary_ids": ["sum-031"],
  "protected_source_ids": ["evidence-17", "conflict-04"],
  "reason": "token_budget",
  "validation": {
    "all_protected_items_reachable": true,
    "provenance_complete": true
  }
}
```

每次压缩都必须记录被移出、保留、摘要化和受保护的条目；验证失败时回退到上一个有效
工作区，并将失败写入 Failure，不得继续使用未验证的压缩结果。

建议新增独立模块 `investigation_archive`，由 `pipeline`、`stage_runner`、Evidence 账本和
trace persistence 共同调用；不得把档案实现塞进 Prompt 或 Gemini 隐藏会话。

### Phase 13：显式主工作区与有损压缩

每轮模型请求只显式构造当前所需信息：

```text
简短 system/task prompt
当前图像命题与最新图像理解
当前假设、开放问题和高优先级任务
高价值支持/反驳/冲突 Evidence 引用
最近少量行动与本轮工具结果
主动回读的原文或图像
剩余预算、无进展状态和可用工具
```

这里采用短 Interaction 的原因不是“上下文越少越好”，而是当前隐藏长链的信息组织质量差、
重复累积严重且无法审计。短链改造的首要目标是提高有效信息密度和阶段交接质量；降低 token
只是约束之一。禁止先切断 `previous_interaction_id`，再用一个过度压缩的摘要代替原有信息。

每次跨阶段调用必须先生成版本化 `StageHandoffPacket`：

```text
handoff_id / source_workspace_version / target_stage
当前任务目标与可用工具
最新 Image Account 与图像理解变化
ImageClaim / ClaimAssessment / MaterialDiscrepancy
开放问题、活跃 Hypothesis、路线状态与失败原因
决定性支持/反驳/冲突 Evidence 引用
本轮新增原始结果和最近少量相关行动
待回读材料、待图像重检和剩余预算
```

它是从 canonical state、不可变档案和工作区确定性组装的数据包，不是让模型自由概括的一段
交接文字。公共 Prompt 仍保持简短，只说明当前阶段任务、可用工具和输出 schema。

不同阶段的最低信息要求：

- Planning：受控分辨率原图、Perception、OCR、retrieval anchors、输入契约和完整工具清单；
- ReAct：当前图像理解、活跃 Claim/Hypothesis、开放问题、路线账本、近期行动、相关 Evidence、
  Failure、待访问页面/参考图和当前可用工具；默认附带低成本全图视图，区域性任务再附相关 crop；
- Discrepancy Decision：本轮新增 Evidence 的精确内容、此前仍有效的决定性正反证据、冲突、
  Claim 状态、Evidence 所有权，以及与判断有关的全图/crop/参考图；
- Reflection/结算检查：全局开放问题、所有活跃或有依据关闭的路线、高价值 Evidence、失败簇、
  recall/reinspection 待办和进展历史，不只给最近两次空结果；
- Judgment：完整 verdict basis、所有决定性支持/反驳/冲突 Evidence、未解决高显著性缺口、
  stop reason、来源独立性和必要的图像视图。不得仅靠压缩摘要作最终裁决。

“足够的信息传递”由保护集合而不是固定条数保证。下列内容标记为 `protected_context`：

- 所有决定性 Evidence 及其精确可回读引用；
- 所有仍未解决的实质冲突；
- 当前高显著性 Claim 与其最新 assessment；
- 尚可改变结论的开放问题、待执行路线、待回读材料和待图像重检；
- 最近一次产生 `decision_gain` 或 `visual_understanding_gain` 的状态变化；
- 当前阶段完成任务所需的工具能力和输出约束。

protected 内容必须显式进入本轮请求，或以短预览和稳定 ID 进入且能在本阶段主动精确回读；
不得因为“最近 N 条”截断而消失。背景材料才允许摘要化或移出主上下文。模型发现交接包不足时，
必须能使用 recall/read/inspect 补材料，不能在缺失上下文的情况下猜测。

主工作区保留的是可继续调查的状态，不是完整聊天记录。已解决的低价值过程、重复查询、
重复来源、导航噪声和旧模型措辞可移出主上下文，但完整内容仍在 Phase 12 档案中。

压缩由独立的 schema-constrained 模型调用完成，Prompt 只要求“保留决定性材料、冲突、
开放问题和引用，输出指定结构”。确定性验证器检查：

- 所有引用 ID 真实存在且仍可回读；
- 决定性证据、实质冲突、未解决高显著性问题不能无故消失；
- 压缩前后的 provenance 链仍可到达原始工具调用；
- 压缩不改写原始账本，不直接产生事实结论；
- 超限或验证失败时回退到上一个有效工作区，而不是继续污染状态。

摘要必须带来源链，不能成为无出处的新事实：

```json
{
  "summary_id": "sum-031",
  "source_ids": ["evidence-17", "evidence-21"],
  "source_spans": [
    {"evidence_id": "evidence-17", "start": 3820, "end": 4510},
    {"evidence_id": "evidence-21", "start": 901, "end": 1302}
  ],
  "summary_text": "...",
  "summary_model": "provider-model-id",
  "summary_prompt_version": "v4-context-summary-001",
  "validated": true
}
```

确定性校验器验证 source ID、span、hash 和可回读性；不评价摘要的事实立场。摘要被后续
Decision/Judgment 使用时，必须记录 `used_by_request_id` 和最终影响的 Claim/Discrepancy。

Gemini 交互改为三种明确生命周期：

- `standalone_request`：Planning、Decision、Reflection、结算检查和 Judgment 独立调用，
  通过 `StageHandoffPacket` 获得完整显式上下文；
- `tool_roundtrip`：一次 ReAct action 内的 `function_call → function_result → output` 短链；
- `protocol_correction`：同一个逻辑请求内的一次结构/引用纠正短链。

后两者可在自己的短链中使用 `previous_interaction_id` 完成协议闭环；闭环结束后不能把该 ID
传给下一阶段。运行时记录 `logical_request_id`、`lifecycle_kind`、父 Interaction 和阶段，检测到
跨阶段 parent 时立即作为协议错误拒绝。

短链切换必须通过“显式信息等价门禁”后才能成为默认路径：

1. `StageHandoffPacket` schema、protected-context 校验和 ContextManifest 全部通过；
2. 能从 manifest 重建实际请求，并证明所有决定性材料、冲突和开放问题仍可达；
3. Planning、ReAct、Decision、Reflection、Judgment 各选真实轨迹人工对照长链基线；
4. Queen 等关键失败样例中，调查方向、图像理解更新和决定性证据利用不能因切链退化；
5. 任一阶段信息覆盖不足时先修 handoff/context builder，不通过恢复全案隐藏历史掩盖问题。

### Phase 14：主动回读与图像再理解

向 Agent 暴露三个有界接口，Planning/ReAct/结算检查均能看到它们：

```text
recall_evidence(query, filters, top_k)
  → 候选 ID、来源、关联任务和短预览

read_evidence(evidence_id, offset/span, include_raw)
  → 精确原文、artifact/provenance 和分页信息

inspect_image(image_id, crop, resolution)
  → 受控分辨率的原图/crop/参考图视图
```

回读分两步：语义检索只找候选，Agent 再按 ID 精确读取；最终裁决只能引用精确读取后的
原文或图像资产。过滤条件至少支持 claim、hypothesis、task、tool、domain、stance、时间和
资产类型。召回排序不能覆盖 Evidence 资格门禁。

每次召回和精确读取记录 `RecallEvent`，至少包含 query、filters、候选 ID、实际选中 ID、
读取 span、下一次使用它的 request ID，以及是否改变了路线、Claim 状态或 Judgment basis：

```json
{
  "recall_id": "recall-014",
  "query": "人物在该事件中使用的交通工具",
  "filters": {"claim_ids": ["claim-03"], "stance": ["support", "refute"]},
  "candidate_ids": ["evidence-17", "evidence-22"],
  "selected_ids": ["evidence-17"],
  "read_spans": ["evidence-17:3820-4510"],
  "used_in_request_id": "req-045",
  "decision_impact": "changed_hypothesis"
}
```

这能区分“召回但未读取”“读取但未使用”和“回读后改变调查方向”。

原图永久保存；默认上下文只放受控分辨率版本。调查获得新人物、事件、地点、物体或编辑
假设后，Agent 可以重新读取原图或指定 crop，以新的问题再次观察并更新图像理解。图像
重检是调查动作，有独立预算、资产 ID 和新旧理解差异记录，不能依赖首次 Planning 的印象。

所有图像上下文项记录原图和发送版本的可复核信息：

```json
{
  "image_id": "img-001",
  "sha256": "sha256:...",
  "original_size": [2048, 1365],
  "sent_size": [1024, 683],
  "crop": null,
  "encoded_bytes": 238441,
  "estimated_tokens": 2800,
  "purpose": "planning"
}
```

图像重新观察另记 `ImageViewEvent`，保存 crop、分辨率、观察问题、前后理解版本和决策影响，
以便验证“调查后是否真正重新认识图像”，而不是只统计调用次数。

### Phase 15：实质进展记账与确定性停止

每个已接受 action 归档后记录一种主要进展状态：

```text
lead_gain
evidence_gain
decision_gain
visual_understanding_gain
no_gain
```

含义：

- `lead_gain`：只有新 URL、RIS 邻居或尚未验证的线索；不重置无进展计数；
- `evidence_gain`：新增合格 Evidence、独立来源族或可复核的反向材料；
- `decision_gain`：Claim/Discrepancy 状态实质变化、关键冲突形成或被解决；
- `visual_understanding_gain`：基于新调查材料重新观察图像后，出现与结论相关的新理解；
- `no_gain`：空结果、错误、重复来源、同义摘要、重复路线或未增加判断能力的材料。

`evidence_gain`、`decision_gain`、`visual_understanding_gain` 才重置连续无实质进展计数。
进展分类由 reducer 创建的 ID 和状态变化生成；确定性代码验证引用、状态变化和计数，
不通过关键词猜测语义。连续计数只进入轨迹、诊断和训练评分，不触发 Decision、路线切换
或停止。

原有路线耗尽兜底继续存在，并与连续无进展机制分开：

- 每条由模型提出并经 reducer 接受的调查路线都有稳定 route ID，关联开放问题、Claim、
  Hypothesis、目标材料、工具以及实际 attempt/failure/evidence ID；
- 路线状态为 `pending | active | resolved | blocked | exhausted | retired`，不能物理删除；
- `blocked/exhausted/retired` 必须有真实尝试、Failure、重复路线引用或已被更有效路线替代的
  provenance，不能仅凭模型一句“搜不到”关闭；
- 当所有未解决高显著性问题都不存在 `pending/active` 有效路线，且没有待精确回读的关键
  档案、待处理的新 Evidence 或待执行的图像重检时，Coverage 可确定性产生
  `meaningful_routes_exhausted`；
- 确定性代码只审计路线账本是否真的没有可执行项，不用关键词判断世界上是否还可能存在
  其他信息。进入该状态前执行一次简短的路线审计，让模型只能提出有依据、非重复、工具
  可执行的新路线；若没有合法新增路线，路线耗尽成立。

停止状态机：

```text
执行新动作
→ 归档原始结果
→ 更新工作区与 gain
→ Coverage 检查路线账本
   └─ 有意义路线已确定性耗尽：meaningful_routes_exhausted（兜底）
→ 若已形成符合 Coverage 前置条件的证据结论：verdict_determined
→ 若达到 24-action 安全上限：hard_budget_exhausted
→ 否则继续调查
→ 进入二元 Judgment
```

v4 最终停止原因区分
`verdict_determined | meaningful_routes_exhausted | hard_budget_exhausted |
engineering_error`。尚有档案待精确回读、新 Evidence 待图像重检或明确的独立来源路线时，
路线账本不得判定为耗尽。全工具不可用或核心协议失败以 `engineering_error` 结束。

### Phase 16：真实隔离实验与 20 例门禁

按相同数据、工具和总体预算依次比较：

1. 当前全案 hidden-history 基线；
2. 显式工作区，无主动 recall；
3. 工作区 + archive + recall + 图像重检 + 确定性路线耗尽/24-action 兜底。

每组保存完整 canonical trace、请求 token、工具成本和停止状态。先跑少量代表性 case
确认协议无误，再跑用户此前使用的完整 20 例。至少审计：

- 最终二元准确率和逐例错误类型；
- 决定性支持/反驳 Evidence 保留率、冲突保留率、Evidence ID 可追溯率；
- recall 命中决定性历史材料的比例和回读后结论变化；
- 调查后重新观察图像并产生有效理解更新的比例；
- 各阶段 `protected_context` 覆盖率、handoff 后关键状态保持率和因信息缺失产生的无效动作；
- 每轮/每案最大输入 token、超过 128k 的请求次数；
- 平均 action 数、跑满 24 action 比例、误判路线耗尽和无效过搜；
- `meaningful_routes_exhausted` 触发次数、触发时剩余可执行路线数和误判路线耗尽率；
- 成本、延迟、工程错误率和 Prompt/协议重试率。

验收条件：

- 单次模型输入以不超过 128k 为硬门禁目标，任何例外必须有逐例原因和后续削减方案；
- token 和 action 显著下降不能以事实准确率、关键反证、冲突或 provenance 丢失为代价；
- 决定性 Evidence 在压缩、回读和 Judgment 中保持可达；
- 所有阶段 protected-context 覆盖率必须为 100%；短链的调查方向、证据利用和图像理解更新
  不得显著差于长链基线，发现退化先修 StageHandoffPacket；
- 无进展计数不触发提前结算；明确新路线或待回读关键材料存在时不得误判路线耗尽；
- 路线账本确实耗尽时能确定性结束，不依赖模型继续生成无意义路线直到 24 action；
- 工程错误不产生 `real` 或 `fake`；
- 不为单个样例添加查询、关键词、URL、交通工具或人物专用规则。

如第 3 组精度下降，按组件回退到第 2 组定位 recall、图像重检或路线账本问题；任何回退只
切换新组件，不恢复全案隐藏历史作为长期方案。

### Phase 17：训练基建前置门禁

本阶段以 `2026-07-21-qwen3-vl-8b-deployment-and-training-infrastructure.md` 为准。
先清理训练仓库中的失效实验脚本、重复 schema、旧生成物和 Qwen3.5 活动入口，但保留
冻结 trace、迁移记录和审计工具。随后部署 Qwen3-VL-8B-Thinking、接入 v4 并由 Qwen 运行 20 例，
再依次完成 SFT 和 Agent RL：

- SFT 与 RL 共享同一 archive/workspace/action/gain/verdict 轨迹契约；
- Gemini 在开发阶段负责教师示范、语义评分和过程评审，可生成候选轨迹，但不是唯一真值；
- Qwen 学生更新后重新采样训练轨迹，Gemini 教师/评分模型冻结用于可比的过程评价；
- provenance、工程错误、预算、档案和终止门禁仍由确定性运行时执行，不交给学生学习绕过；
- Qwen base 四条 canary 通过前不得运行完整 20 例；有意义的 SFT 数据门禁通过前不得正式
  SFT；SFT checkpoint 通过真实 canary 前不得启动真实工具 RL。

## gpu-13 实施与验收摘要

- 本地 Windows worktree 是唯一代码编辑位置：`D:\image-factual-verifier-v2-worktrees\visual-fact-discrepancy-agent-v4`。
- gpu-13 checkout：`/gs/home/wza/projects/image-factual-verifier-v2-worktrees/visual-fact-discrepancy-agent-v4`。
- 数据根：`/gsdata/home/wza/image-factual-verifier-v2-data`；运行产物不得写入 Git checkout。
- 服务器只拉取已提交代码、测试、调用 provider、保存轨迹和审计；禁止直接改服务器源码，工作树不净时停止。
- 所有项目命令经过 `scripts/server/run_gpu13.sh`，设置 `OMP_NUM_THREADS=1` 和 `IFV_DATA_ROOT`。
- 凭据从 `/gs/home/wza/.config/image-factual-verifier/runtime.env` 加载，权限 `600`，不得打印或写入 Git/trace。
- 门禁顺序：确定性测试 → 四条历史回放 → 单条真实 canary → 3～4 条异构验收。
- v4 Development Preview：`${IFV_DATA_ROOT}/releases/automatic-diverse-20-development-preview-v4-20260715`。
- Jupyter 当前端口为 8333；本地 `jupyter_remote.py` 调用必须 bypass proxy；远程 Bash 命令整体用单引号避免 PowerShell 提前展开。
- 服务器 git pull 使用项目代理 `http://100.10.1.210:47899`；旧 47894 废弃。

## 完成定义

正式执行链不依赖单一 core fact 或全案 Gemini 隐藏历史；Planning、调查、回读、图像重检、
压缩、进展和停止均可追溯且有界；原始材料不可变保存，工作区可压缩，决定性材料可精确
回读；`real | fake` 二元 Judgment 与内部不确定状态分离；历史回放和四组真实隔离实验通过；
Qwen base、SFT 和 RL 使用相同冻结 20 例完成人工与 strict audit 对照；单次输入满足
128k 门禁目标；无样例专用规则；停止机制不因无进展计数误杀有效调查，且不降低关键证据
与结论质量；Qwen SFT/RL 设施按 2026-07-21 统一计划完成保存、恢复、部署、on-policy
rollout、reward 和重新采样门禁。

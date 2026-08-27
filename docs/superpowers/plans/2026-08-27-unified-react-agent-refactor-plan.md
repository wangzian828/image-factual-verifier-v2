# IFV 统一大 ReAct Agent 重构计划

日期：2026-08-27
状态：本地实现与回归已完成；真实 Gemini smoke 与生产切换待执行。
目标版本：`unified-react-v1`（新 trace / 审计 / 训练版本；历史
`discrepancy-first-v4` trace 保持只读兼容）

## 1. 目标与结论

当前运行时把以下内容固定为多个独立阶段：

```text
固定 perceive_scene + OCR
→ Image Account Planning
→ ReAct 工具行动
→ Query Concept Extraction / Query Replan / Route-local Replan
→ Discrepancy Decision / Reflection
→ Judgment
```

这使真实的“为什么先看图、为什么 OCR、为什么换查询、为什么先访问候选页”被拆散到多次
Gemini 请求和多个结构化 JSON 中。它不利于训练 Qwen 学习完整的
`think → action → observation → think` 行为。

目标结构改为一个以工具为中心的大 ReAct：

```text
图片 + case 公共字段 + 空 workspace
  → ReAct：think → 一个工具或少数控制动作 → observation / state delta
  → ReAct：think → 下一个工具或控制动作 → ...
  → 低频 Global Reflection（仅整体卡住时）
  → 稀疏 Discrepancy Decision（证据语义检查点）
  → 最终 Judgment（一次）
```

`perceive_scene` 和 `ocr_with_position` 仍是每案必要的前置观察，但不再由
`Orchestrator.run()` 直接调用或并发预执行；它们必须成为大 ReAct 中可审计、可训练的真实
工具行动。runtime 可以按依赖关系限制工具可用性，但不能绕过模型直接产生这些步骤。

## 2. 设计原则

1. 模型只输出思考和下一步行动；不输出通用 `update_state` JSON。
2. 工具执行后的 Observation、Discovery、Evidence、Failure、预算、重复路线和状态变化由
   reducer 确定性写入。
3. 模型的调查意图由实际行动参数表达；首次外部调查行动携带目标事实和路线意图，runtime
   验证后自动建立 canonical state，不增加独立 Planning 请求。
4. “换 query”是下一次不同的 `text_search`；“换视觉检查”是下一次视觉工具调用。二者不再
   触发独立 Replan LLM 请求。
5. `stop_route` 是少数允许的控制动作：它不调用外部服务，但明确关闭一条已无价值的路线。
6. Reflection 只处理全局策略，不做每步局部规划；Decision 只处理 Evidence 的语义与 verdict
   前置条件；二者不替代 ReAct。
7. 所有 Gemini 可见 thought 文本与其对应 action 绑定保存。SFT 对每个 ReAct turn 学习
   `<think>...</think><tool_call>...</tool_call>`，不伪造缺失 thought，也不导出不可见隐藏
   CoT。
8. private gold、判定标签、构造信息和 SFT judge 数据不得进入 ReAct、thought 或工具参数。
9. 已成熟的外部工具、视觉工具及其 provider 集成不因本次架构重构而改语义、请求参数、返回
   schema、重试、缓存、图片处理或错误分类。重构只改变“何时向模型暴露工具、如何校验模型
   行动、如何归并其既有结果”。

## 3. 文献对应关系

本方案借鉴的是通用模式，不复制任何单一论文的完整架构。

| 文献 | 对本项目的约束含义 |
| --- | --- |
| ReAct，arXiv:2210.03629 | 局部 replan 内嵌于每轮 reasoning + action；Observation 回到下一轮上下文，不需要单独的 Query Replan 阶段。 |
| WebArena，arXiv:2307.13854 | 环境状态由 action 执行后的状态转移自动更新；agent 不需要调用通用状态同步工具。 |
| SWE-agent，arXiv:2405.15793 | action 改变外部环境，harness 维护交互历史与后续上下文；不把运行时状态写入责任交给模型。 |
| Reflexion，arXiv:2303.11366 | Reflection 是低频、基于反馈的策略修正，而不是每个工具回合的固定步骤。 |
| MemGPT，arXiv:2310.08560 | 显式 memory write 适用于跨上下文窗口的持久记忆；它是例外，不应成为单案 workspace 的默认更新方式。 |
| WebAgent-R1，arXiv:2505.16421 | agent SFT 可以建模为完整历史条件下的下一步 action；完整 episode 是原始来源，训练视图可以是其紧凑序列化。 |

## 4. 目标运行时结构

### 4.1 初始状态与工具门控

初始 `ImageOnlyInvestigationState` 只含：

- case 公共字段、图片路径、图片 hash；
- 空的观察、目标、路线、Discovery、Evidence、Failure 和审计账本；
- action / request / visual / route 等预算；
- 运行时工具健康信息。

主策略默认不直接接收原图。原图只交给视觉工具，避免模型绕开可审计观察直接把视觉判断写进
thought。`direct_multimodal` 如需保留，只能作为后续明确标记的实验模式，不能成为正式
teacher / SFT 主链。

工具可用性按“已满足的依赖”计算，而不是按固定函数顺序执行：

| 当前已完成观察 | 当前允许工具 |
| --- | --- |
| 无 | `perceive_scene`、`ocr_with_position` |
| 仅完成其中之一 | 缺失的另一个视觉 bootstrap 工具；已完成者不可重复 |
| 两者都完成 | 外部搜索、反向搜图、网页访问、参考图比较、裁剪/OCR/视觉检查，以及必要的 `stop_route` |

因此，通常轨迹会先做 scene 再做 OCR，或相反；这是模型产生的两次 action，而不是 pipeline
在模型调用前写入的隐藏结果。外部调查工具在两项基础观察完成前不暴露，保证最低图像依据。

模型侧不必也不得填写真实图片文件路径。动态工具 adapter 在 action 通过后，把当前 case 的
不可变图片引用绑定到既有工具所需的 `image_input` 等参数；底层
`perceive_scene`、`ocr_with_position`、`focused_visual_inspection`、裁剪、比较和搜索工具的
原始参数合同不改变。

### 4.2 统一 ReAct 回合

一个正常回合固定为：

```text
runtime 渲染最小当前 handoff + 可用工具 schema
→ Gemini 返回可见 thought + 一个 native function call
→ runtime 校验 action / 参数 / 预算 / 依赖
→ 执行该工具
→ reducer 归并结果，写入 Observation 与 state delta
→ 下一回合重新渲染当前 handoff
```

每个回合最多一个 function call。并行工具调用、重复路线、不可用工具、未知 ID 和违反来源策略
的 query 都作为协议拒绝保留在 trace；它们不是主 SFT 的正样本。

`StageRunner` 的 native Interactions 生命周期仍可保留，但语义上不再存在“Perception 预执行
阶段”“Planning 请求阶段”“Query Replan 请求阶段”。新 trace 的主 action stage 统一命名为
`unified_react`，原始工具名和 reducer delta 区分实际行为。

### 4.3 调查意图如何进入状态

模型不调用泛化的 `update_state` / `initialize_investigation` 工具。

首次**非 bootstrap 的调查工具调用**必须在其参数中携带 `investigation_intent`；它是本次真实
工具行动的语义意图，不是单独的状态补丁。其最小内容为：

```json
{
  "target_fact": {
    "statement": "图片希望观众接受的正向现实世界命题",
    "kind": "attribute | relation | internal_consistency | text_claim",
    "predicate": "简短关系名",
    "anchor_fact_ids": ["来自已完成视觉/OCR观察的 ID"]
  },
  "route": {
    "route_focus": "调查角度",
    "expected_information": "该真实工具行动要获得的底层事实",
    "priority": 1
  }
}
```

工具 provider 实际只接收其原有参数；runtime 在执行前拆出并校验 `investigation_intent`，
随后调用 `apply_initial_action_intent(...)`：

1. 验证 target 的视觉/OCR 锚点、正向现实关系和非 verdict 边界；
2. 验证 route 与当前工具及 query / visual focus 一致；
3. 创建稳定的 `target_fact`、首条 route 和 task；
4. 原子提交 intent 与工具结果造成的状态变化；
5. 将接受或拒绝原因返回在 `role=tool` observation / state delta 中。

这让模型用“实际要做什么”表达计划，runtime 决定如何安全持久化计划。后续的工具调用引用
runtime 已生成的 task / route ID；不再重新创建初始 account。

目标事实只能在 Evidence 支持的受限 refinement 中由 reducer 更新，不能通过普通 ReAct action
任意替换。多 target 的需求保留为受限的后续新增 route / target 机制，必须有已记录 Evidence
或视觉观察作依据，不能回到独立 Planning 的自由 JSON。

### 4.4 局部 replan 的归并

删除以下独立 LLM 请求：

- `image_only_query_concept_extraction`
- `image_only_query_replan`
- `image_only_route_local_replan`

替代规则：

| 旧行为 | 新行为 |
| --- | --- |
| 从 Evidence 提取概念 | 写在本轮 `<think>` 中；下一次行动直接使用新线索。 |
| replacement query | 直接发起新 `text_search`；reducer 对同 task 历史 query 做语义去重。 |
| add visual route | 直接调用可用视觉工具，参数写明具体可观察属性。 |
| continue | 不需要特殊 action；下一轮继续选择工具。 |
| stop route | 调用受限 `stop_route(task_id, rationale)` 控制工具，runtime 校验无 pending 候选、无必做观察且预算允许。 |

`stop_route` 只影响指定 route / task，不得创建 Evidence、变更 target、提议 verdict 或结束整个
episode。

### 4.5 保留的低频语义检查点

保留三个非普通工具回合：

1. **Global Reflection**
   - 触发：连续多个有效 action 没有信息增益、可执行路线整体耗尽、接近 action 上限；
   - 输入：紧凑全局路线账本、未解决 gap、最近 Observation 和失败；
   - 输出：仅全局优先级、是否继续 / 停止调查的建议；
   - 不输出 query、不创建 route、不直接改 Claim / Evidence。

2. **Discrepancy Decision**
   - 触发：出现有方向的合格 Evidence、参考图/视觉比较完成、关键路线关闭、最终前；
   - 输入：已审阅 Evidence、关联视觉锚点、当前 target 和 route ledger；
   - 输出：ClaimAssessment、MaterialDiscrepancy、必要的受限 refinement、`continue/fake/real`；
   - 不承担下一步工具规划。

3. **Judgment**
   - 每案最多一次；
   - runtime 先编译允许的 basis；Gemini 只能复现已确定 verdict，或在确实未由 Evidence
     决定时执行受限二元判断；
   - 不允许继续调用工具或改变 workspace。

旧的 `Evidence Decision` 与 `Discrepancy Decision` 功能重叠。正式 `unified-react-v1` 只保留
后者；旧 Evidence Decision 保留在历史回放兼容层，不进入新生产路径。

## 5. 状态、trace 与 SFT 合同

### 5.1 reducer 是唯一状态写入者

每个 accepted ReAct action 生成一条不可变 state delta，至少含：

```text
action_id / interaction_id / function_call_id
工具与已验证参数
接受的 investigation_intent（仅首次或受限 refinement）
新建 Observation / Discovery / Evidence / Failure ID
route 与预算变化
拒绝或降级原因（若有）
下一回合可用工具集合
```

完整 workspace、原始工具输出和 handoff shadow 留在 canonical archive；模型上下文只拿到当前
所需的紧凑 observation、状态增量、活跃 route、未解决 gap 和授权工具。

### 5.2 Gemini thought 保留与训练

每个 Gemini interaction 都保存：

- provider 返回的可见 `thought` / thinking summary 文本；
- thought token 数；
- interaction ID、function call ID、工具 action；
- 原始 provider 响应的审计引用。

不返回可见 thought 的回合只记录空值和 token 统计；不得自行生成、补全或用另一个模型改写。

新的 policy SFT episode 以真实时间顺序导出：

```text
user: 图片 case 公共字段 + 空 workspace
assistant: <think>...</think><tool_call>perceive_scene</tool_call>
tool: 真实工具 observation + reducer state delta
assistant: <think>...</think><tool_call>ocr_with_position</tool_call>
tool: 真实工具 observation + reducer state delta
assistant: <think>...</think><tool_call>text_search(... investigation_intent ...)</tool_call>
tool: 真实工具 observation + reducer state delta
...
assistant: Decision / Reflection / Judgment 的结构化输出
```

Qwen 继续使用原生 `<think>` 与 `<tool_call><function=...><parameter=...>` 模板。assistant 的
thought、工具调用和保留的语义检查点输出计算 loss；`user`、`tool`、state delta 与原始观察
只作上下文。主 reasoning SFT 桶只接收有可见 ReAct thought 的动作链；thought 缺失但行动正确
的轨迹可另存为 action-only / RL 候选，不能伪装为 reasoning 监督。

现有“policy 不带图、perception 单独训练”的划分必须更新：新 policy episode 包含感知工具的
选择与其观察结果。独立 perception 训练可继续作为视觉工具能力的辅助数据，但不能再替代
policy 对“何时感知、何时 OCR、何时搜索”的监督。

### 5.3 版本与兼容性

- 新 runtime、strict audit、SFT exporter、RL adapter 使用 `unified-react-v1`。
- 旧 `discrepancy-first-v4` trace 不修改；继续允许只读回放、旧审计与旧导出。
- 新旧 trace、accepted release、SFT 包和 reward ledger 不混合；清单必须记录
  `decision_policy_version`。
- 不对既有 940 / 后续 teacher trace 静默重写；新架构通过单独 rollout 目录重新生产。

## 6. 代码改造范围

### 6.0 成熟工具保持不变

本次禁止修改下列成熟工具的业务语义：

- `src/tools/perceive_scene.py`
- `src/tools/focused_visual_inspection.py`
- `src/tools/compare_reference.py`
- `src/tools/crop_and_inspect.py`
- `src/tools/crop_and_search.py`
- `src/tools/reverse_image_search.py`
- `src/tools/check_consistency.py`
- `src/tools/count_objects.py`
- `src/tools/visual_anomaly.py`
- `src/integrations/browse/`、`src/integrations/gemini/` 中既有 provider 传输与提取逻辑。

同样不得借这次重构修改工具的图像压缩、上传、超时、重试、代理、缓存、OCR 坐标、网页抽取或
Evidence 语义。若后续发现工具自身缺陷，必须作为独立 issue / 提交 / 回归集处理，不能混入
统一 ReAct 重构。

允许变动的只有 orchestrator 层：

```text
动态可用工具集合
→ 模型侧的窄化 schema
→ runtime 参数注入与检查
→ 调用既有工具
→ 复用既有结果解析
→ reducer 写入 state delta
```

`investigation_intent`、task / route ID、当前图片引用等均属于 runtime adapter 字段。它们必须
在调用底层工具前剥离或绑定，绝不能扩散到工具 provider 的公开请求体。`stop_route` 是新的
orchestrator 控制工具，不伪装为或修改任何外部工具。

### 6.1 需要替换的主入口

当前 `src/orchestrator/pipeline.py::Orchestrator.run()` 固定执行：

```text
_run_perception()
→ build_bootstrap_investigation()
→ state_from_bootstrap()
→ _run_image_account_planning()
→ _run_discrepancy_investigation()
→ _run_discrepancy_judgment()
```

目标替换为：

```text
build_empty_investigation(runtime_case)
→ _run_unified_react_loop(...)
→ _run_global_reflection_if_needed(...)
→ _run_discrepancy_decision_if_needed(...)
→ _run_discrepancy_judgment(...)
```

`_run_perception()` 不再在正式入口调用。其工具执行、结果解析和坐标校验逻辑迁移到
`record_tool_observation()` / 新的 `reduce_visual_bootstrap_action()`；不得复制两份视觉解析逻辑。

### 6.2 模型与 reducer

新增或重构：

- 最小空 investigation state；
- `InvestigationIntent`、`RouteIntent` 与运行时动态工具参数 schema；
- `apply_initial_action_intent(...)`；
- `reduce_unified_react_action(...)`；
- `available_unified_react_tools(...)`；
- `stop_route` 控制工具与 reducer；
- 新版 stage handoff / compact context renderer；
- `unified-react-v1` strict audit、trace schema 和 replay adapter。

迁移 / 删除正式调用路径：

- `build_bootstrap_investigation(...)`：保留给历史回放；新路径不生成预置 task；
- `apply_image_account_planning(...)`：迁移其 target / route 验证逻辑到 action-intent reducer；
- `apply_query_replan(...)`、`apply_route_local_replan(...)`：保留只读 replay，移除新路径调用；
- `_run_image_only_target_planning(...)`、`_run_image_only_investigation(...)`：确认仅旧兼容后冻结；
- 独立 `image_only_evidence_decision`：从新路径移除。

### 6.3 Prompt 与输出上限

Prompt 整理在 action contract 固定后进行。不得直接删短旧 prompt；先逐条盘点其约束，
将每条约束标记为“保留、迁移、由 runtime 接管、删除”，并记录理由、目标位置和回归测试。

盘点清单如下：

| 现有 prompt / 常量 | 重构后的处理 | 必须人工复核的旧约束 |
| --- | --- | --- |
| `IMAGE_ACCOUNT_PLANNING_SYSTEM_PROMPT` | 不再作为独立请求；其有效 target / route 约束迁移到首次 action 的 `investigation_intent` schema、validator 与统一 ReAct prompt。 | 正向现实命题、视觉/OCR 锚点、中心 target、路线中性、禁止预设 verdict、可执行首跳。 |
| `DISCREPANCY_REACT_SYSTEM_PROMPT` | 重写为正式 `unified-react-v1` 主 prompt。 | pending candidate 优先、路线去重、底层事实检索、来源策略、Evidence 边界、单工具调用。 |
| `QUERY_CONCEPT_EXTRACTION_SYSTEM_PROMPT` | 删除独立请求；其“必须来自新 Evidence、不得复述旧 query”的约束迁移到统一 ReAct prompt 与 query validator。 | Evidence phrase 的新颖性、不可凭空引入概念。 |
| `QUERY_REPLAN_SYSTEM_PROMPT` | 删除独立请求；其约束迁移到新的 `text_search` action validator。 | 与尝试过的 query 语义不同、仍服务同一 target、不得成为 verdict。 |
| `ROUTE_LOCAL_REPLAN_SYSTEM_PROMPT` | 删除独立请求；具体路线切换改为下一次真实 action，关闭路线改为 `stop_route`。 | 不拼接互相竞争的猜测、视觉检查必须有可观察区分点、只停止当前路线。 |
| `REFLECTION_SYSTEM_PROMPT` | 保留，但缩成纯全局策略 prompt。 | 低频触发、全局 gap、不得创建 Evidence / 写 verdict / 改不可变事实。 |
| `EVIDENCE_DECISION_SYSTEM_PROMPT` | 新生产路径删除；逐条比对后将仍需要的 Evidence 语义约束迁移到 Discrepancy Decision reducer / prompt。 | exact span、source metadata 不是正文、source-to-image binding 的适用条件、受限 refinement。 |
| `DISCREPANCY_DECISION_SYSTEM_PROMPT` | 保留并精简，作为唯一证据语义检查点。 | admissible stance、视觉锚点、visual reinspection、discrepancy 原子性、real / fake 前置条件。 |
| `DISCREPANCY_JUDGMENT_SYSTEM_PROMPT` | 保留并精简。 | compiled basis 一致性、Evidence 已决定 verdict 时不得改写、受限终局视觉理由。 |
| 工具内部 prompt（场景感知、视觉复核、参考图比较、Jina 提取等） | 本次不改代码内容；只在文档中登记其存在、输入/输出与主 Agent 的边界。 | 不得误把工具 prompt 规则丢进主 Agent prompt，也不得把主 Agent policy 混进工具 prompt。 |

每个保留或新建的主 Agent prompt 都必须：

- 使用清楚的总起和编号结构；
- 与实际动态 schema、允许工具和 reducer 合同一一对应；
- 不重复 runtime 已硬性保证的冗长细节；
- 不隐含样例、固定实体、固定 URL 或训练标签；
- 具有独立 prompt version；
- 在 `docs/active-agent-system-prompts-zh.md` 中有逐段中文备份，在
  `docs/agent-prompt-and-runtime-guide.md` 与中文 guide 中说明其运行时边界。

每次 prompt 修改均应更新一张变更表：

```text
旧常量 / 旧条款
→ 保留、迁移、runtime 接管或删除
→ 新常量 / schema 字段 / validator
→ 删除或迁移理由
→ 覆盖该条款的测试名
```

完成盘点并由人工确认前，不得删除旧常量、历史 prompt 版本或其中文备份。

实际改动顺序如下：

1. 重写统一 ReAct prompt 与动态工具说明；
2. 重写 Reflection 为纯全局策略；
3. 精简 Discrepancy Decision；
4. 精简 Judgment；
5. 同步 schema field descriptions、中英文 prompt 备份、架构文档。

当前临时修改的 `DISCREPANCY_REACT_SYSTEM_PROMPT` 只可视为方向性草稿；在统一 action schema、
工具门控和 trace stage 完成前，不得把它当作最终 prompt，也不得用其启动正式全量 teacher
rollout。

统一 ReAct 的输出预算应覆盖“可见 thought + 一个工具调用”，但不因旧 Planning / Replan 的预算
继承而膨胀。具体 token 上限在 mock trace 长度和 Gemini 实测后确定；单条训练 episode 的真实
Qwen processor 审计仍以 128K 为硬准入线。

## 7. 实施阶段与验收

### Phase 0：冻结基线与契约测试

- 记录当前 `discrepancy-first-v4` trace / exporter / audit 版本；
- 为当前 `_run_perception()` 固定直调、独立 Planning、Query Replan、Route-local Replan 编写
  明确的负向回归断言，防止新路径悄悄复用旧顺序；
- 冻结成熟工具的输入/输出、错误分类、缓存和 provider 请求合同；为每个工具建立最小
  contract regression，确保后续只改 orchestrator adapter；
- 完成上节 prompt 条款盘点表，逐条确认迁移位置和删除理由；
- 不修改历史 trace、accepted release 或训练包。

验收：旧回放和现有全量测试保持通过；成熟工具 contract 无差异；所有旧主 Agent prompt 条款
都有保留、迁移、runtime 接管或删除记录。

### Phase 1：空状态与视觉 ReAct 工具

- 实现 `build_empty_investigation(...)`；
- 将 scene/OCR 纳入统一工具注册、动态 schema、原生 function call 和 `record_tool_observation()`；
- 新 mock provider trace 必须显示 Gemini 先后选择两个视觉工具，并有对应 thought、function
  call、tool result 和 reducer delta；
- 删除正式入口对 `_run_perception()` 的直接调用。

验收：

- 无模型 action 时不产生 perception/OCR 观察；
- 新路径不能在视觉 bootstrap 完成前调用外部检索；
- 任一视觉工具失败形成 Failure / 工程错误，不伪装成事实结论；
- 感知和 OCR 顺序可由模型选择，但二者都必须完成才能开放外部调查。
- 视觉工具的既有 provider 输入、输出 schema、图片处理和错误分类与 Phase 0 基线一致。

### Phase 2：行动意图与首次调查

- 为所有首次非 bootstrap 调查工具添加动态 `investigation_intent` 参数；
- 实现原子 `apply_initial_action_intent(...)`；
- 将当前 Image Account Planning 的安全检查迁移为 reducer 验证，不再发独立 LLM request；
- 使 `text_search`、反向搜图和视觉检查均可作为合法首个调查行动。

验收：

- 首次调查 action 同时创建 target/route 与真实工具结果；
- 非法 target、无锚点、verdict/provenance 导向、无可执行路线、违禁 query 均零状态修改；
- 通过 intent 后的下一回合只看到 runtime 生成 ID，不重传完整 Planning JSON。

### Phase 3：Replan 收敛

- 把 query novelty、pending candidate 优先、路由预算和 source policy 检查移动到统一 action
  validator；
- 删除新路径中的 Query Concept Extraction、Query Replan、Route-local Replan 调用；
- 增加受限 `stop_route`。

验收：

- 新 query 直接表现为 `text_search` action；
- 新视觉方向直接表现为视觉工具 action；
- 新 trace 中没有 `image_only_query_concept_extraction`、
  `image_only_query_replan`、`image_only_route_local_replan` 请求；
- route 关闭不会影响其他 route 或产生 verdict。

### Phase 4：保留的语义检查点

- 只保留低频 Global Reflection、稀疏 Discrepancy Decision 和一次 Judgment；
- 删除正式路径的旧 Evidence Decision；
- 重新定义触发条件，使其完全依赖 reducer 记录的证据、路线和预算。

验收：

- 普通连续工具行动之间没有 Reflection / Decision 请求；
- 关键 Evidence、路线整体耗尽、全局持续无收益才触发检查点；
- Decision / Judgment 的 basis、Evidence 和 verdict 一致性仍通过 strict audit。

### Phase 5：trace、SFT 与 RL

- 增加 `unified-react-v1` exporter；
- 每个 ReAct assistant turn 用关联的 Gemini 可见 thought + Qwen 原生 tool-call 模板导出；
- tool observation 与 reducer delta 放入同一 `role=tool`；
- 更新 processor 审计、SFT eligibility、semantic reward 与 RL adapter 的版本分流；
- 为 thought 缺失轨迹建立明确的 action-only / RL 分桶，不进入 reasoning SFT 主桶。

验收：

- 新 SFT 行从开头就包含 `perceive_scene`、OCR、调查和后续工具 action；
- 没有独立 Planning / Query Replan JSON 作为主训练行为；
- 消息交替合法、工具调用可被 Qwen parser 解析、无 evaluator-private 字段；
- 全量 episode 经真实 Qwen processor 编码后不超过 128K，超限只进入 holdout。

### Phase 6：回放、真实 smoke 与生产切换

顺序：

1. 本地 reducer / mock Interactions / exporter / strict audit 全量测试；
2. 旧 v4 只读回放，确认兼容层未回归；
3. 服务器单条 Gemini smoke：审查原始 thought、工具 action、state delta；
4. 10 条并发 10 smoke：检查工程错误、协议拒绝、thought 捕获率、每案工具顺序和 token；
5. 通过后再启动独立的 `unified-react-v1` teacher rollout，不复用旧 rollout 目录。

生产验收最低要求：

- scene 与 OCR 都由模型产生的 tool action 完成；
- 新路径零固定 bootstrap 直调；
- 新路径零独立 Query/Route Replan 请求；
- 成功 ReAct action 的 thought 可见文本捕获率单独报告；
- 工程错误自动排队重跑；重跑不覆盖已成功 episode；
- strict audit、SFT eligibility、Qwen processor 审计和训练包清单均按新版本通过。

## 8. 非目标与风险控制

本次不做：

- 不重写或删除旧 teacher trace；
- 不将 private gold、SFT judge 结果写入运行时；
- 不为了减少请求而取消 Evidence / basis 审计；
- 不把 thought 当作 Evidence；
- 不在未完成新 mock / smoke 前启动全量 8,490 条新 rollout；
- 不把“初始视觉工具门控”伪装成模型自由选择外部搜索的能力。

主要风险与处理：

| 风险 | 控制措施 |
| --- | --- |
| 模型跳过 OCR 或视觉感知 | 动态工具门控；未完成的 bootstrap 工具持续暴露，外部调查工具不开放。 |
| 首次 action 的 intent 过长或不合格 | 动态 schema、原子 reducer、有限 revision；失败不写状态。 |
| 将换 query 误判为重复路线 | 用现有语义路线等价检查迁移到 action validator，并回放历史 query 边界。 |
| thought 缺失 | 原样记录；reasoning SFT 与 action-only / RL 分桶分离。 |
| 新完整轨迹再次超 128K | 保持 canonical archive 与训练视图分离；仅串接真实 action、工具观察和 reducer delta，不重复累计 workspace。 |
| 旧 trace / 审计损坏 | 新 policy version 和新目录隔离；旧版只读兼容测试先行。 |

## 9. 完成定义

只有满足以下全部条件，`unified-react-v1` 才可替代当前生产 teacher 路径：

1. 每个新 episode 从空 workspace 启动；
2. scene perception 与 OCR 是 agent 产生、可训练的工具调用；
3. 目标与初始路线由首次真实调查 action 的 intent 经 reducer 建立；
4. query / 视觉路线调整直接表现为下一次 ReAct action；
5. 新路径没有独立 Planning、Query Replan、Route-local Replan 请求；
6. Reflection、Decision、Judgment 只在定义的低频边界运行；
7. 每个可见 Gemini thought 与具体 assistant action 可一一关联；
8. strict audit、SFT eligibility、Qwen tool parser、真实 processor 128K 审计和工程重试链全部通过；
9. 新旧数据版本、产物目录和训练清单完全隔离；
10. 成熟工具 contract 与冻结基线无差异；每个阶段 prompt 都有版本、中文备份和条款迁移记录。

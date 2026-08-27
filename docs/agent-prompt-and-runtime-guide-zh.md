# Agent Prompt 与运行时说明（中文备份）

## 当前生产候选：unified-react-v1

新路径从空 workspace 开始，主策略模型先选择 `perceive_scene` 与
`ocr_with_position`。完成二者后，首次真实调查工具调用携带
`investigation_intent`；reducer 原子创建 target、route、task，并把 adapter 字段从底层工具
请求中剥离。此后每轮一个工具动作，路线调整直接表现为下一个 ReAct action。

只保留低频的 Global Reflection、Discrepancy Decision 与一次 Judgment。Decision 不得创建新
route/query；它只处理已有 Evidence 的语义状态。旧 v4 的 Planning、Query Replan、Route-local
Replan、Evidence Decision 仍在代码中供历史回放，但不进入 `unified-react-v1` trace。

训练导出保持完整 episode：每个 ReAct action 是 Qwen 的
`<think>…</think>` 加原生 tool-call 模板，随后接真实工具观察与 reducer delta。可读 provider
thought 不完整的轨迹只进入 `action_only.jsonl` 与 `rl_candidates.jsonl`，绝不混入 reasoning
SFT。所有 release、SFT dataset、eligibility 和 reward artifact 都记录
`decision_policy_version`；`unified-react-v1` 不可与旧 v4 manifest 混合。

对应英文原文：[agent-prompt-and-runtime-guide.md](agent-prompt-and-runtime-guide.md)。
代码中的 prompt 常量和 schema 才是运行时唯一准则；本文用于说明各阶段看到了什么、可以
做什么，以及程序如何约束模型输出。

这是一份 guide 的中文翻译，**不是实际 system prompt 原文**。当前主 Agent 的实际 prompt
中文备份见 [active-agent-system-prompts-zh.md](active-agent-system-prompts-zh.md)。

## 阶段职责总表

| 阶段 | 语义判断由谁负责 | 程序强制保证什么 |
| --- | --- | --- |
| Case 校验 | 无 | 行只能有三个公开字段；路径不能越界；SHA-256 必须一致 |
| 感知 / OCR | Gemini 或工具 provider | 必需工具成功；输出必须满足字面 schema |
| Bootstrap | 无 | 生成稳定的视觉事实和检索锚点 |
| Image Account Planning | Gemini | 恰好一个核心高显著性 Claim；最多两个中显著性 Claim；必须关联像素/OCR 锚点；假设数量有上限；整体原子提交 |
| ReAct | Gemini | 每次只能对一个 Claim/Hypothesis 拥有的任务执行一次原生工具行动；路线和 provider 受限 |
| 观察结果归并 | 无 | 严格区分 Discovery、Evidence、Failure；来源不可变、可追溯 |
| Discrepancy Decision | Gemini | 只能审阅已有 Evidence；校验归属和锚点；受预算限制；整体原子提交 |
| Coverage / basis | 无 | 只允许 Evidence 已决定的结论，或在受限条件下作二元收束；选择最小允许 basis |
| Judgment | Gemini | 输出的 basis/ID 必须和程序编译的 basis 完全一致；只有 Evidence 尚未直接决定时才能做二元选择 |

## Image Account Planning：先确定图片到底在声称什么

Planning 是独立请求，并获得受控的原图视图。它还会看到一个紧凑的观察包，其中包括：

- 去重后的场景感知；
- 带位置的 OCR；
- 所有非机械性的像素/OCR `VisualFact` 锚点；
- 检索线索。

确定性的 bootstrap task 和生命周期交接元数据保留在 canonical archive 中，但不会作为
Planning 的输出示例提供给模型。Planning 输出的是正向 `ImageClaim` 和暂定的
`SearchHypothesis`；它不会创建核心 verdict fact，也不会直接给出 verdict。Hypothesis 的
schema 不暴露 Claim key，也不要求为某个 Claim 写专属验证问题。

但 Planning 至少必须输出一条可执行路线。这是结构要求，不规定它必须调查哪一个具体事实。
Hypothesis 的 `queries` 非空时，就已经声明存在文本搜索路线；reducer 会据此派生
`text_search`，但不会改写 query 原文。`suggested_tools` 只表达可追加的能力。

在 reducer 提交任何 Claim、Hypothesis 或 Task 之前，完整的 Planning 对象会先通过当前
`SourceAccessPolicy`。只要其中一个 query 被拦截，就会触发有上限的
`planning_revision`：程序既不会改写被拦截 query，也不会从该被拒对象中挑出看起来安全的
兄弟项提前提交。相同的 active policy 还会在 provider 返回后、标题/摘要/聚合/URL 进入
Discovery 或 provider-visible context 之前过滤这些行。对 active evaluation policy，
已知事实核查站点在这一检索边界被排除；过滤器不改写 query，也不决定 Evidence 的语义。

每条 `ImageClaim` 都应写成图片向观众传达的、关于现实世界的正向命题。它不能把这个命题
偷换成更容易的元命题，例如“可见文字、帖子或广告声称了某事”。

- Claim 必须写成图片希望观众接受的事实；
- 不能写成怀疑、矛盾、真实性判断或 verdict；
- 搜索 query 应寻找事实本身和来源，而不是寻找现成的 fact-check 答案；
- 恰好一个 Claim 是高显著性，并保留完整的中心关系；
- 可选的独立 Claim 是中显著性，不能拆成中心关系的碎片。

图片定义了需要核查的“图片账号”，并提供最初线索；但它不能限制调查会使用的事实、来源、
关系或 query 词汇。Planning 应独立建立现实中的基础事实，而不是只去寻找图片给出的值。
已有知识可以用于提出暂定的 `SearchHypothesis`，但只有工具产出的 `Evidence` 才能正式
建立事实。

Image Account Planning 默认使用 Gemini `thinking_level=high`，并请求官方提供的
`thinking_summaries=auto`。trace 可以记录 Gemini 返回的思考摘要和 thought token；
这不是完整隐藏推理。调查、提取、视觉工具、Decision 和 Judgment 保持简洁，且思考摘要
不能当作 Evidence。

## ReAct 与原生工具协议

每个行动都从一条新的原生工具短链开始，并且只能选择一个活跃的
Claim/Hypothesis task。`previous_interaction_id` 只能在该行动内部的
`function_call` 和 `function_result` 之间使用；不得传给下一次行动或其他语义阶段。

运行时会动态暴露尚未尝试、且有上限的路线。一次检索批次最多只会开放一个具体的 `visit`
或 `compare_with_reference` 行动；在重复检索前必须先检查 pending candidates。
Claim/Hypothesis 的 ownership 用来保存谱系，不是语义牢笼。Gemini 可以选择任何有用的
query 角度，包括独立调查底层现实事实的问题；这种 query 本身不会改写 ImageClaim，也不会
自动创建 Evidence。

Archive recall 是在可执行 task 内部受到严格限制的辅助能力：

- task 尚未产生 archive 中的调查材料时，不暴露 recall；
- 每个 task 最多两轮 recall/read；
- 只能读取处于 pending 状态、且由 recall 返回的确切 ID；
- 被 recall 回来的候选仍然只是记忆，不能因为再次取回就自动变成 Evidence。

一次行动只开放一个 task 范围内的路线族。网页检查时，Gemini 需要选择该 task 已拥有的一条
Claim ID，并写出希望阅读的段落；运行时会绑定准确 Claim 文本，供 stance extraction 使用。
这可以避免把无关路线中的 URL、Task 和 Claim 重新拼接在一起。

Planning 之后，reducer 会把每条初始路线广义地登记到当前 image account，使 Evidence、
预算和停止条件仍可审计。这个内部登记不会显示为 Planning 的选择，也不能在没有合格的、
有方向的 Finding/Evidence 链时建立 `ClaimAssessment` 或 `MaterialDiscrepancy`。

工具执行后，运行时会归并 canonical result，保存原生 `function_result`，并结束这一段行动。
下一个语义检查点会先提交该 pending result，再提交一条携带新编译 state 的 `user_input`。
任何 provider、模型或 wire-protocol 切换都不能把错误隐藏起来。
ReAct 不负责输出 segment-summary JSON；工具调用成功后由 runtime 关闭本行动段并编译下一次
handoff。Reflection、Replan、Decision 和 Judgment 是独立检查点，不要求在每次工具行动前
运行。

如果 v4 ReAct 持续选择已经被拒绝的重复路线，直到修正预算耗尽：

- Runtime 不会再额外请求模型，也不会虚构搜索；
- 它会写入一个与被拒请求 ID 绑定的确定性边界；
- 当前 Task 会被标为 `blocked`，但不增加 action count；
- 随后直接进入独立的 Discrepancy Decision。

这条 episode 仍保留给审计，但不能导出为 policy 训练数据。传输、schema、生命周期错误，
以及非 v4 的修正耗尽，仍会 fail closed。v4 Discrepancy Decision 在其最终更新也被拒后，
可以把同一确定性边界作为空的 `continue` 检查点；它绝不会借此虚构 Claim、Evidence、
discrepancy 或二元 verdict。

## Discrepancy Decision：根据已有证据更新调查状态

这个独立检查点只会稀疏地运行，典型触发点包括：

- 获得合格的直接 Evidence；
- 同一拍摄/参考图比较完成；
- 到达一个实质性的 Evidence 边界；
- 受限路线选择耗尽；
- 即将以 unresolved 状态终止前。

它会通过显式 workspace handoff 看到 ImageClaims、hypotheses、准确的 Evidence、可见锚点、
已尝试路线、Failure 和剩余预算。

它可以：

- 评估 Claim；
- 建立或标记冲突的 `MaterialDiscrepancy`；
- 有上限地增加/退休 hypotheses；
- 请求一次由 Evidence 驱动的视觉复核；
- 提议 verdict。

它不能：

- 把搜索摘要当作证据引用；
- 发明 ID；
- 扩大 image account；
- 把 provider 失败变成事实 verdict。

`supported` 表示准确的 ImageClaim 为真；`refuted` 表示它为假。进入修正回合前，运行时会
一次报告该输出中能够确认的、彼此独立的 ID、ownership、方向和 Finding-chain 合约错误；
同时也报告不兼容的 verdict/route 前提，不会把后续错误藏在第一个错误之后。

最后一个可解析 JSON 即使因 schema 或语义校验被拒，也会以 `output_rejected` 的 trace step
保留其确切字段和原因；不能被错误标为“空格式错误”。

耦合的原子结果必须一起输出：反驳一个高显著性 Claim，要求同一 JSON 同时包含其已建立的
decisive discrepancy 和 `fake` proposal。已经写入的 discrepancy 是 canonical state，
不是下一次更新可以套用的模板。若重复完全相同的受影响 Claim、Evidence、materiality 和
status，运行时会将其作为结构化重复拒绝，而不会要求再生成一个 ClaimAssessment。

只要有一条高显著性 Claim 得到合格反驳，就已经足以决定 `fake`。其他 Claim 尚未解决，
不会把这个矛盾降级为 supporting，也不会要求先重建图片的其他所有部分。

## 观察结果的语义

搜索和反向搜图输出只会创建 `Discovery`。抓到的准确网页段落，或成功的视觉观察，才可能
创建 `Evidence`。

提取器会分别标注：

1. 该段落覆盖的是同一个完整 Claim 关系、部分关系、不同实例，还是范围不清；
2. 它是支持、矛盾、背景信息，还是不确定。

只有“相同完整关系 + 支持/矛盾”才是有方向的材料。其他准确文本仍作为可审阅的中性上下文
保留。行动的 retrieval goal 只负责选择读哪段文本，不能决定这些标签，也不能改写
ImageClaim。只有被接受的 Discrepancy Decision 才能更新 Claim 的语义状态。

Evidence 和 Finding 会保留 task ownership。一个影响多个 Claim 的 discrepancy，必须对每个
受影响 Claim 都有该 Claim 所属的合格 Evidence，并有与每个 Claim 重叠的可见锚点。

## Judgment 与导出

确定性 compiler 选择准确的 Claim、discrepancy、可见锚点、Finding、Evidence 和 unresolved
gap。

- 当 Evidence 已经决定结论时，Judgment 只能复现 compiler 得到的二元 verdict；
- 当 case 在 unresolved 条件下终止时，才在同一份完整 basis 上执行受限二元 Judgment；
- 两种模式都不能新增事实。

严格审计器和 `ifv-policy-v2` exporter 会拒绝以下情况：

- 未知 ID；
- 证据/Claim/basis 不对齐；
- verdict 后继续行动；
- 协议被拒；
- private evaluator data 泄漏；
- 仍在使用 legacy core ownership。

被拒绝的 Planning revision 会保留在审计记录中，但不会成为 SFT target。

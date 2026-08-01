# Evidence-Triggered Visual Reinterpretation：首轮诊断结果

**状态：** 首轮最小样本集完成，暂不修改 Agent
**记录日期：** 2026-08-01
**对应计划：** `docs/research/evidence-triggered-reinterpretation-diagnostic-plan-2026-07-29.md`
**运行边界：** 冻结 reviewed-52 快照、真实 `ifv-qwen3.5-9b`、原样 v4 Decision 与 focused visual inspection

## 1. 结论

现有 v4 确实具备完整的 Evidence-to-Vision 闭环能力，但该能力尚不稳定。
Diana 是一个严格成功实例；其余四条 canonical trace 分别暴露了复看问题偏离来源、
不必要复看、未触发复看，以及复看结果未被后续 Decision 正确消费的问题。

当前证据不支持引入更大的 belief graph 或 `current_visual_interpretation`
架构。最早断裂主要发生在现有 Decision 的 trigger/question bridge 和复看后的
Evidence 消费环节。下一轮应先验证最小约束与消融，而不是扩展状态架构。

## 2. 运行与版本说明

- 远端执行通过 `docs/operations/gpu13.md` 规定的链路：Windows SSH control endpoint
  → jump host → Jupyter → gpu-13 `ifv-agent` kernel。
- 运行前确认 `hostname=gpu-13`、`id -un=wza`、`OMP_NUM_THREADS=1`。
- 只使用 `/tmp/ifv-github-dcf8452` clean clone；未修改服务器真实 checkout。
- clean clone 已更新到 detached `d7f4e69c053ce5bbb7325ed1f2e3ac746a97789f`，
  `git status --short` 为空。
- Diana、L'Hoest 和 surgery 的结果生成于更早的 `5d599b0` / `7e31a0a` harness；
  后续提交只涉及 replay harness、scoring release 路径解析和结果报告，未改变 Agent
  prompt、schema、reducer 或控制流。
- 所有样本均使用冻结快照和显式 source Evidence ID；没有使用 evaluator-private gold
  引导 Decision。

## 3. Canonical trace 诊断表

| Case | Source-side property | 复看问题与视觉结果 | 后续 Decision 影响 | 严格 composite | 最早断裂环节 |
|---|---|---|---|---|---|
| Diana `case_e141ff4d95d50483` | BusinessInsider：握手时未戴手套 | 定向检查手部；pixel Evidence 明确看到手套 | 新建 source-pixel composite Finding，最终 `fake`，`decision_mode=evidence_determined` | 成功 | 无；完整闭环 |
| L'Hoest `case_37dbca846edacd23` | 新物种的口鼻周围有橙色斑块 | 确实触发复看，但问题转成通用 AI artifact，而非橙色口鼻或物种鉴别特征 | 未创建 composite；最终 verdict 仍由旧 source Finding 决定 | 失败 | question generation / bridge |
| Surgery tray `case_fd6908a9bdddbaa3` | Alamy 只支持宽泛手术室场景，没有 tray/specimen 或手部配置对立事实 | 不必要地询问“一件器械还是双手协助”；视觉结果无法解决 tray/specimen 命题 | Decision 2 保持 `continue`；无 composite；非终局状态不能编译 verdict | 失败（受控负例） | trigger precision / source-question alignment |
| Bolt scoreboard `case_f5c90d970ac8f08e` | 2009 Berlin 100m 官方成绩为 `9.58`，图像命题写 `9.98` | 当前 `d7f4e69` canonical replay 未请求复看 | 旧 source Finding 直接导出 evidence-determined `fake`；basis 无 pixel Evidence | 失败 | Evidence-to-Vision trigger |
| Sarcophagus `case_61f8bff089386a99` | 来源描述长棕发、现代衣着的活人躺在木棺中并拿银色念珠；图像命题是裹尸木乃伊、石棺和纸草卷 | 准确询问“裹尸木乃伊还是长发活人”；pixel Evidence 明确观察到绷带木乃伊和卷轴 | Decision 2 仅产生 `insufficient` assessment，且只选择 source Evidence，漏掉新 pixel Evidence；无 composite、无终局 | 失败 | post-reinspection state consumption / provenance selection |

## 4. 逐样本证据

### 4.1 Diana：完整成功

- source Evidence：`evidence-4810c42e77675a547798`
- visual Evidence：`evidence-78be010a7382837b7769`
- composite Finding：`finding-2763f717e1b14f6dcf6f`
- 结果：`fake`，`decision_mode=evidence_determined`
- 远端结果：
  `/gsdata/home/wza/image-factual-verifier-v4/runs/replays/diana-snapshot43-composite-5d599b0/result.json`

该轨迹满足严格成功条件：Decision 2 新建 composite Finding，最终 basis 同时包含
composite Finding、source Evidence 和 visual Evidence。

### 4.2 L'Hoest：触发成功、问题错误

- snapshot：`snapshot-00000052.json`
- source Evidence：`evidence-f5fa953da280c6f23ec4`
- source Finding：`finding-f2e0f99ca86b929cd17c`
- visual Evidence：`evidence-b557b40bdd491ceaaac7`
- 远端结果：
  `/gsdata/home/wza/image-factual-verifier-v4/runs/replays/lhoest-snapshot52-composite-7e31a0a/result.json`

来源已经给出橙色口鼻这一可见鉴别属性，但复看问题没有绑定该属性。即使工具运行并
返回 neutral pixel observation，也没有形成可供 Decision 消费的目标性观察。

### 4.3 Surgery tray：受控负例

- snapshot：`snapshot-00000053.json`
- source Evidence：`evidence-3ef626241134a1e870e7`
- visual Evidence：`evidence-066003c4fda9e3117c28`
- Decision 1 问题：`Does the surgical professional hold a single instrument versus using two hands to assist another?`
- Decision 2：`continue`，`created_composite_finding_ids=[]`
- 远端结果：
  `/gsdata/home/wza/image-factual-verifier-v4/runs/replays/surgery-snapshot53-composite-7e31a0a/result.json`

没有 composite 是正确结果，因为 source Evidence 没有提供 tray/specimen 对立事实；
但触发一次与 source property 不一致的复看属于 trigger precision 假阳性。结果停留在
非终局状态，因此 `compile_error` 是预期的终局编译保护，不应补造 verdict。

### 4.4 Bolt scoreboard：当前基线未触发复看

- snapshot：`snapshot-00000024.json`
- source Evidence：`evidence-7037ef4fc6ae8a77b2ca`
- 当前 `d7f4e69` 结果：
  `/gsdata/home/wza/image-factual-verifier-v4/runs/replays/bolt-snapshot24-composite-d7f4e69-r2/result.json`
- 当前结果：未请求复看；`composite_success=false`；最终 `fake` 的 basis 只有
  `finding-40535f063c951e8db4f6` 和 source Evidence。

一次更早的 exploratory replay 使用同一 snapshot、Evidence ID 和 sampling seed，
当时 clean clone 实际为 `7e31a0a`，尽管目录名误写成 d7。该次运行正确询问
`Does the scoreboard display the time '9.58' or '9.98'?`，并得到 pixel Evidence
`evidence-5b61d69fdbee6b6d7782`，明确观察到 `9.98`；但 Decision 2 仍未创建
assessment、discrepancy 或 composite。结果保留在：
`/gsdata/home/wza/image-factual-verifier-v4/runs/replays/bolt-snapshot24-composite-d7f4e69/result.json`。

`d7f4e69` 相对 `7e31a0a` 只增加结果报告字段，Agent runtime 没有变化。因此两次结果
差异应记录为模型/服务采样敏感性，不应解释为代码回归。canonical 计数采用当前 head
的 r2 结果。

### 4.5 Sarcophagus：正确复看，消费失败

- snapshot：`snapshot-00000029.json`
- source Evidence：`evidence-7883167a6587afa9c8ba`
- visual Evidence：`evidence-4a7270c8efaa6f603afe`
- 远端结果：
  `/gsdata/home/wza/image-factual-verifier-v4/runs/replays/sarcophagus-snapshot29-composite-d7f4e69/result.json`
- 结果：`composite_success=false`，无终局 verdict。

visual Evidence 以 `0.99` confidence 记录整具人形被破旧绷带包裹，且没有长棕发或现代
衣着。Decision 2 虽然执行并接受一个 `insufficient` assessment，却只引用 source
Evidence，并把 remaining gap 写成“需要视觉协调”。这说明正确视觉结果已经存在于
canonical state，但没有被 Decision 的 provenance selection 消费。

## 5. 首轮计数

以五条 canonical trace 为分母：

- 请求 visual reinspection：4/5；
- 复看问题正确绑定 source-side 鉴别属性：2/5（Diana、Sarcophagus）；
- focused visual observation 正确回答目标问题：2/5；
- 复看结果实际改变后续 Decision：1/5（Diana）；
- 严格 composite success：1/5；
- 最终 evidence-determined verdict：3/5，其中只有 Diana 的 basis 包含完整
  source-pixel composite 链。

最早断裂类型计数：

- 完整成功：1；
- Evidence-to-Vision trigger failure：1；
- question generation / source-question alignment failure：2；
- post-reinspection state consumption / provenance selection failure：1。

## 6. 排除的候选

### Maid Marian

`case_9647ef7ca00bfbb8` 的所有旧 attempt 只包含 Ribbons / East-West 钢带雕塑相关
Evidence，没有“雕像在砂岩石上而非石旁”这一 source-side 空间事实。它不能作为
空间关系正例。

### Rugby / EAKINS

`case_9b223c87c2fa993c`、attempt `064804-a8de38` 的有效 source Evidence 是
“Jan 1991、St Albans team”。它没有引入 EAKINS 球标；而 EAKINS 在初始图像账户和
pixel/OCR anchors 中已经存在。因此该样本不满足“新知识揭示此前未建模的视觉区别”，
不应作为第二个正例。

## 7. Go / no-go

- **No-go：** 不引入 belief graph、自由改写 interpretation state 或更大的控制流。
- **Go：** 进入最小改动实验设计，但在实现前先把未修改基线固定为本文件中的五条
  canonical replay。
- 第一优先级是 source-specific question gating：复看请求必须从 reviewed Evidence
  中指出一个具体可见 property，并让问题直接列出 source-side 与 image-side 候选值。
- 第二优先级是 post-reinspection consumption：当 resolved reinspection 已生成 pixel
  Evidence 时，下一次 Decision 对相关 Claim 的 assessment/discrepancy 必须显式选择
  该 pixel Evidence；缺失时不能把“尚待视觉协调”作为理由继续。
- 最小消融应比较：原样 v4、仅 question gating、question gating 加 post-reinspection
  consumption。仍不增加新的 belief 数据结构。

## 8. 下一步

在改 Agent 前，再补两类轨迹可提高诊断覆盖但不是首轮完成的阻塞项：

1. support-alignment：`case_720d256cc8126414` 的 mummy mouth / gold foil 样本，验证
   source 与 pixels 同向时是否会产生不必要复看或错误 composite；
2. visual-ambiguous：选择一个分辨率不足、正确答案应为 ambiguous 的局部物种或产品
   属性样本，验证系统是否保留不确定性。

完成这两类补充后，再实现最小 intervention，并在同一 frozen snapshot 集合上做成对
replay；不要用 free-running rollout 的搜索随机性替代机制消融。

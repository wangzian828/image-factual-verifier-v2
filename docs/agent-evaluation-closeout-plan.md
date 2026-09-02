# Agent 与评测收尾执行计划

**状态：新版 Agent-100 真实重跑、逐条审阅和文档收尾已完成，停在大规模教师 rollout 启动前。** 本文是当前工作的连续执行记录。每次涉及 Agent 的代码、prompt、schema、
harness、审计口径或运行配置变更，必须先在第 4 节追加一行，再实施、测试、提交和部署。

Gemini 3.7 的当前在线可用性不影响本文档和本地回归的完成。依赖 3.7 的新 smoke、
judge 或 rollout 若遇到 API 不可用，只能暂停该运行；已经完成的历史 run、指标和审阅结论
不得覆盖或删除。

## 1. 目标与停止边界

完成以下工作：

1. 测出 Gemini 3.7 的当前可用并发；
2. 完成全量 direct QA 的 3.7 补跑与 private-gold judge；
3. 修复 unified Agent 当前已确认的参考图、候选处理、ReAct 指引与审计缺口；
4. 用修复后的 Agent 重跑并审计测试集 100 条；
5. 整理实验记录、SFT 上下文/导出和 PSD/RL 的前置验证。

**明确停止点：** 完成上述工作后，整理出可启动大规模教师轨迹 rollout 的命令、数据集、
并发与验收条件，但**不得实际启动**训练集的大规模教师轨迹 rollout。必须等待人工复核和
明确确认。

## 2. 不可违反的边界

- 不把历史 3.7 压测或 smoke 结果当作当前 API 在线保证；若 3.7 当前不可用，依赖它的新
  运行暂停，文档整理、本地测试和人工复核继续。
- 仅在本地 Windows 工作树修改源码；本地测试后提交、推送；gpu-13 只 fast-forward 和运行
  已提交代码。
- 统一测试集不能进入 teacher rollout、SFT 或 RL；private gold 只能在 rollout/QA 输出后进入
  judge，不能进入模型输入。
- 不恢复旧 v4 兼容路径；当前主链只维护 unified-react-v1。
- 不覆盖或混入用户现有未提交改动，尤其是 `src/trajectory/exporter.py`、`docs/operations/gpu07.md`
  及其对应 README hunk。
- Agent 工程错误与外部访问失败分开：网页、SSL、验证码、图片下载不可用应记录为
  `external_unavailable`；模型/工具 contract 矛盾才是工程错误。

## 3. 当前基线与运行状态

| 项目 | 当前状态 |
|---|---|
| 本地/服务器分支 | `codex/gpu13-canary-20260804-plan-relaxation-01` |
| 服务器工作树 | `/gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01` |
| 当前运行时代码 commit | `55ef0b3 Harden Gemini scene perception recovery` |
| 最新仓库 HEAD（含文档收尾） | `10bb1e1 docs: finalize agent100 closeout status` |
| gpu-13 checkout | 已同步到 `10bb1e1`，工作树干净；运行时代码仍为 `55ef0b3` |
| 测试集 | 1,684 条，real 447 / fake 1,237 |
| 3.1 Pro full direct QA | 1,682 完成、2 工程错误；judge 已完成 |
| 3.7 full direct QA | 1,684 完成、0 工程错误；private-gold judge 已完成 |
| 3.7 Agent-100（旧版） | 100 条与 report sidecar 已完成；仅作为旧基线 |
| 新版 Agent-10 smoke | 3.1 Pro、3.7 Flash 各 10/10 完成、0 provider 工程错误；逐条审阅见 2026-09-02 报告 |
| 新版 Agent-100 | 99/100 成功、1 条 provider `content_blocked`；99 条已完成统一 3.7 Flash private-gold judge |
| 大规模教师 rollout | 禁止启动，等待本计划全部完成后的人工确认 |

## 4. Agent 变更日志（从现在起持续追加）

| 时间（UTC） | 类别 | 文件/范围 | 改动与原因 | 验证/部署 | commit |
|---|---|---|---|---|---|
| 2026-08-30 20:55 | 基准设施 | `scripts/benchmark_gemini_interactions_concurrency.py` | 新增可复用 Gemini 并发基准；混合结构化文本、原生工具调用、图片结构化请求；强制零重试，并显式设置 request gate 与 HTTP 连接数，避免 256/512 档被默认 128 连接限制悄然串行化。不是 Agent 语义改动。 | 本地 `py_compile`、`--help`；gpu-13 fast-forward 后 2 并发/4 请求真实 smoke 为 4/4 成功。 | `f383cc0` |
| 2026-08-30 21:12 | 已实施：ReAct prompt | `src/orchestrator/unified_prompts.py` | 将“反向搜图得到匹配后”改为“返回未验证候选”；补一条自然的具体问题导向 query 指引。未新增固定槽位、query 拒绝器或真假倾向。 | 定向断言通过；已随 `5c63e49` 推送并部署。 | `5c63e49` |
| 2026-08-30 21:12 | 已实施：参考图 adapter | `src/orchestrator/unified_react.py` | 从 runtime `InvestigationDiscovery(reference_image_url, candidate_url)` 建立参考图→候选页面映射；compare 调用没有显式 `source_page_url` 时自动绑定。成熟 compare 工具不改。 | adapter 注入/显式覆盖单测通过；已随 `5c63e49` 推送并部署。 | `5c63e49` |
| 2026-08-30 21:12 | 已实施：候选 URL 与下载诊断 | `src/tools/reverse_image_search.py`、`src/tools/crop_and_search.py`、`src/tools/compare_reference.py` | 接受合法 HTTP(S) 的无扩展名图片候选，按响应 `Content-Type` 验图；把直接下载、URL 变体、候选页、页面图片提取、HTTP/网络/策略失败写入 subcalls。 | extensionless 与诊断 subcall 单测通过；已随 `5c63e49` 推送并部署。 | `5c63e49` |
| 2026-08-30 21:12 | 已实施：审计输出上限/主分类 | `scripts/audit_direct_qa_baseline.py`、`scripts/audit_agent_private_gold.py`、`src/eval/private_gold_metrics.py` | private-gold judge 默认输出上限升至 8,192；Agent 主三分类改为决定性、落地依据 / 正确但不足 / 错误，旧严格覆盖度仅保留诊断。 | Agent/Direct 分类差异单测通过；全量 3.7 judge 已完成。 | `5c63e49` |
| 2026-08-30 21:12 | 已核实：无需重复改 | `src/orchestrator/task_store.py`、`src/tools/compare_reference.py` | 当前代码已有失败/低相关候选的有界 sibling 控制；compare 已从 typed `differences` 派生并保留 `edit_evidence_*` 的原始字段与修复记录。先以真实 smoke 验证，不重复造同类补丁。 | 新版 10 条 smoke 中 strict audit 通过，未出现 compare contract 工程错误。 | 既有代码 |
| 2026-08-30 21:50 | 已实施：审计去重 | `scripts/audit_direct_qa_baseline.py` | 3.7 direct QA 采用追加式补跑后，审计器按 `case_id` 只取最后一条结果，再进入 judge，避免旧失败记录污染最终统计。 | 重复结果回归测试通过；3.7 全量 private-gold audit 已完成。 | `ea4c7a4` |
| 2026-08-31 | 已实施：统一 prompt 语言与文档 | `src/orchestrator/unified_prompts.py`, `src/orchestrator/context_workspace.py`, `docs/active-agent-system-prompts.md` | 当前主 Agent 的四类 runtime prompt 和 stage objective 统一为英文；中文文件降为阅读对照，不再作为运行时输入。新增从源码生成精确 prompt 备份的脚本。未改变工具契约、状态机、private-gold 隔离或判断规则。 | 本地 prompt 导出、compileall、diff-check 和真实 smoke 均已验证。 | `b68ce31` |
| 2026-08-31 | 已实施：轨迹可读视图 | `scripts/trajectory/render_sft_episodes_readable.py`, `docs/trajectory-artifact-guide.md` | 保留 canonical `trajectory_sft.jsonl` 不变，新增逐 episode 的原始 JSON 与 Markdown 审阅视图，明确 manifest/index 不是轨迹。 | 已生成 10 个 episode 目录；当前 smoke 的 reasoning/action-only 分桶已在服务器复核。 | `5c63e49` |
| 2026-08-31 | 已修复 Gemini follow-up 与跨 action session 问题 | `src/orchestrator/stage_runner.py`, `src/orchestrator/pipeline.py`, `scripts/audit_real_trace.py`, `src/trajectory/exporter.py`, `src/trajectory/report_history.py`, `docs/gemini-interaction-sequence.md` | 保留候选图下载、校验、压缩和 `function_result` 纯文本；恢复同一 ReAct episode 的 provider-side `InteractionSession`。工具结果只在下一请求显式前递一次，原图只在根请求上传，后续依靠会话历史；多个 `user_input` 合并。审计改为检查父链连续及工具结果前递；SFT/report 继续按 canonical 时间顺序单次投影，不重复展开 provider 历史。 | 本地/服务器相关门禁通过；新版真实 smoke 10/10、strict audit 10/10、0 工程错误。 | `c87f36b`、`85dab10`、`99609df`、`922d9d2` |
| 2026-08-31 | 已更新回归断言 | `test_native_structured_output.py` | 将共享 InteractionSession 的跨阶段测试改为验证后续阶段复用 provider 历史中的原图，只发送新的文本输入；不再把重复上传原图当作正确行为。 | 本地/服务器定向测试通过。 | `f88352a` |
| 2026-08-31 | 已适配真实 canary 验收 | `scripts/run_real_canary.py`, `test_real_canary_cli.py` | 当前主流程是 claimless `react_runtime`：不再要求 ImageClaim、claim_ids 或旧的终止原因；只对当前 ReAct schema 检查有动作、视觉记忆、统一 judgment basis 和 fact-check report。旧图谱 trace 仍保留原校验。 | 本地/服务器定向测试通过；用当前 checkout 复核既有 smoke trace 通过。 | `62f6b9c` |
| 2026-08-31 | 收尾验证与实验记录 | `docs/agent-evaluation-closeout-plan.md`, `docs/reports/2026-08-30-gemini-direct-qa-experiment-record.md` | 写入 3.7 全量 direct QA/judge、10 条新版 Agent smoke、SFT 分桶、128K 长轨迹和“未启动 Agent-100/教师 rollout”的决策。 | 本地 77 项当前门禁、服务器 103 项定向门禁通过；服务器无 rollout/audit 残留进程，CLOSE-WAIT=3。 | 本次文档提交 |
| 2026-09-01 | 已实施：ReAct 调查状态由模型维护 | `src/orchestrator/react_runtime.py`, `src/orchestrator/unified_prompts.py`, `test_react_runtime.py`, `test_prompt_boundaries.py` | 将 `investigation_progress.status` 定义为 `investigating`、`decision_capable_support`、`decision_capable_refute`。runtime 只校验、保存和传回这个状态；不根据工具名、`stance`、`directness`、`relevance` 或 `evidence_class` 推断它，也不因此动态增删调查工具。既有 Evidence 归档语义保持不变。主动结束仍需模型声明方向性状态；24 次动作上限和 Judgment 流程不变。 | 定向 pytest、compileall、git diff --check 通过；已提交并部署；后续 3.1 Pro / 3.7 Flash smoke 均完成。 | `62f6b9c` |
| 2026-09-02 | 已实施：`perceive_scene` 请求恢复 | `src/tools/perceive_scene.py`, `src/tools/visual_common.py`, `src/orchestrator/stage_runner.py`, `scripts/server/start_gemini_eval_gpu13.sh` | 视觉请求默认使用压缩 JPEG；对可恢复的 Gemini 400、传输失败和超时追加一次更小图片/宽松 schema 的恢复请求；移除 provider 不接受的 schema 约束；工具动作边界调整为 210 秒。失败仍记录为工具错误，不伪装成成功。 | 本地 468 项测试通过；gpu-13 fast-forward 到 `55ef0b3`；3.1 Pro 与 3.7 Flash 各完成 10/10、0 provider 工程错误。逐条质量问题见 `docs/reports/2026-09-02-perceive-scene-recovery-and-smoke.md`。 | `55ef0b3` |
| 2026-09-02 | 已实施：测试 100 条正式 release 重建 | `scripts/prepare_agent_test_release.py` 及服务器生成 release | 复用原 Agent-100 的精确 100 个 case，排除其余 1,584 个测试 case；补齐顶层 `training_prohibited=true`，保证 runtime 只含 case/image/hash，gold 与来源策略仍在 evaluator-private。 | 100/100 case、5 个构造子路线各 20 条；首次启动因旧 release 元数据缺失而拒绝，失败目录保留；正式 release 已创建。 | `55ef0b3` |
| 2026-09-02 | 文档收尾与状态统一 | `docs/agent-evaluation-closeout-plan.md`, `docs/reports/2026-09-02-*.md`, `docs/plans/2026-09-01-tool-parity-and-react-observation-plan.md` | 统一 runtime commit、仓库 HEAD、服务器状态、468 项测试结果、3.7 在线可用性边界和大规模 rollout 停止点；把已完成工作从“待提交/待部署”改为可核对的完成状态。 | 文档检查、链接/路径核对、`git diff --check`；不启动任何新 rollout。 | `10bb1e1` |

后续每一条 Agent 改动必须记录：修改前行为、修改后行为、为何不改变 private-gold
隔离/成熟工具契约、对应测试、真实 smoke case、commit 和服务器部署状态。

## 5. 执行顺序

### A. Gemini 3.7 当前容量定标

1. 使用基准脚本，对 `gemini-3.7-flash` 运行 64、256、512 三档；每档 512 请求、无重试。
2. 使用同一张测试集中位大小图片，profile 为 `mixed`，`thinking_level=high`，
   `max_output_tokens=2048`，使请求形状接近 QA judge 与 Agent 子调用。
3. 每档记录：原始成功率、排除内容拦截后成功率、成功返回/min、全部请求端到端 p90、
   成功请求 p90、错误类型。
4. 由结果分别指定 QA judge 的请求并发，以及 Agent rollout 的 case 并发。二者不能相同
   视为理所当然，因为一个 case 会产生多次 Gemini 调用。

以下是已落盘的历史容量基准，不代表 2026-09-02 之后 Gemini 3.7 仍然在线。

| API 并发 | 请求数 | 成功 | 原始成功率 | 排除内容拦截后 | 成功返回/min | p90 | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 512 | 510 | 99.61% | 99.61% | 196.83 | 77.53s | 已完成；2 个 Gemini HTTP 错误 |
| 256 | 512 | 484 | 94.53% | 94.53% | 382.11 | 31.55s | 已完成；28 个 `ReadError` |
| 512 | 512 | 502 | 98.05% | 98.05% | 372.41 | 39.04s | 已完成；9 个 `ReadError`、1 个 Gemini HTTP 错误 |

定标结论：256–512 的吞吐已进入约 370–382 successful returns/min 的平台区，且本轮没有
内容拦截。生产不直接采用 256/512：先以 **QA judge 请求并发 64**、**Agent case 并发 16**
作为保守起点，分别通过真实 judge 小样本和 10 条 Agent smoke 后再决定是否提高。

### B. 完成 Gemini 3.7 full direct QA 与 judge（已完成）

1. 等待已有自动恢复补齐 1,684 条，检查最新记录是否每个 case 恰好一个 terminal 状态。
2. 重新生成 full direct-QA 的 Accuracy、BACC、real/fake 召回和混淆矩阵。
3. 将 direct-QA judge 的输出上限固定为 8,192 后再启动 3.7 judge，防止 `high` thinking
   消耗 2,048 上限造成结构化 JSON 截断。
4. 使用 A 中测出的安全并发，运行 1,684 条 private-gold judge；输出三分类、工程错误和
   `reason_quality` / `fact_match` 诊断。
5. 与 3.1 Pro full result 同口径比较，主看 BACC 与两类召回；Accuracy 不能单独定优劣。

### C. unified Agent 修复（已完成）

以下每项先写第 4 节变更日志，再改代码：

1. **反向候选语义。** ReAct prompt 明确反向搜图返回的是未验证候选，不等于图片已匹配。
2. **搜索问题表达。** ReAct prompt 只增加简短自然指引：每次搜索要回答图片中的具体人、
   物、事件或关系，不要反复改写泛化场景词。不得引入固定槽位、强制换 query、查询拒绝器，
   或证据不足时强判 fake。
3. **参考图页面回退。** 将 `InvestigationDiscovery` 中配对的候选页面 URL 与参考图片 URL
   传给 `UnifiedReactToolAdapter`；模型未显式给 `source_page_url` 时，compare 自动注入配对页。
   不改成熟 compare 工具的下载/视觉契约。
4. **无扩展名图片 URL。** 反向搜图不因 URL 无 `.jpg/.png` 等扩展名提前过滤；compare 下载器
   按真实 HTTP `Content-Type` 判断图片。
5. **参考图失败可观测性。** 记录直链、URL 变体、候选页面、页面图片提取、HTTP、策略阻断等
   具体失败层级，不再只写笼统的 access failed。
6. **候选重复控制。** harness 显式记忆已比较、低相关和不可访问候选；后续避免无意义重复访问。
7. **视觉字段矛盾。** 对 `edit_evidence_present=false` 但 `edit_evidence_strength!=none` 的模型
   输出进行受审计的保守归一化或一次重试；未恢复的 contract 矛盾保留为工程错误，不把它伪装成
   普通外部访问失败。

### D. Agent private-gold 审计与报告（旧版 Agent-100 基线已完成）

1. Agent 100 条的主三分类采用：正确且有决定性、落地的实际依据；正确但依据不足；判断错误。
2. 旧的“严格 private target 全覆盖”计数只保留为内部诊断，不作为主表或 gate。
3. Agent 的第一类要求 selected Evidence、verdict basis 和 fact-check report 一致；direct QA
   的第一类只评价其图片直答的 `core_fact/reason`，不能称为检索证据。
4. 所有 private-gold judge 默认输出上限至少 8,192；审计结果 JSON 被截断时不能计为语义失败。

### E. 真实验收与新 Agent-100（诊断性重跑）

1. 为 C/D 的每项补定向单测；本地运行相关 pytest、`compileall` 与 `git diff --check`。
2. 每个提交 push 后，gpu-13 fast-forward；服务器运行定向测试。
3. 先用 10 条真实 Gemini 3.7 smoke 验证 trace：候选标识、页面回退、失败记录、工具 contract、
   终止与 report 均正常。
4. 先用 10 条真实 smoke 验证工程稳定性和调查轨迹；本次按人工要求，即使 smoke 的
   accuracy 不能作为质量结论，也继续把同一固定 100 条作为诊断性 Agent-100 跑完。
   该 100 条不进入训练，只用于逐条轨迹审阅和 private-gold judge。
5. 旧 Agent-100、3.7 direct-QA-100、3.1 Pro direct-QA-100 的历史对照继续保留；
   新版 Agent-100 的最终分类必须等待所有工程补跑和 judge 完成后写入。

### F. 文档、SFT 与 PSD/RL 前置收尾（已完成）

1. 更新 `docs/reports/2026-08-30-gemini-direct-qa-experiment-record.md`，补齐 3.7 full judge
   和新 Agent-100 的最终指标；文档只保留当前认可的主口径。
2. 审查真实 Agent 的 provider input token ledger，区分 runtime 上下文膨胀与 SFT exporter
   重复保存完整状态；完整 canonical archive 不删除。
3. 在不覆盖用户已有 `src/trajectory/exporter.py` 改动的前提下，完成 Qwen 原生格式导出与
   紧凑 handoff 复核：每轮保留 thought、工具调用、工具观察、必要状态增量，不重复嵌入完整
   next-stage packet。
4. 复核 PSD/RL 的前置代码、private-gold 隔离、训练输入格式与 smoke；不在本阶段启动正式 RL。
5. 汇总为“可启动大规模教师 rollout”的审阅包：确定输入训练集、并发、retry 队列、SFT 分桶、
   验收标准和启动命令。到此停止，等待人工确认。

## 6. 实际执行结果与停止决策

历史记录曾执行到人工复核边界；2026-09-02 已按新的人工指令继续进行诊断性 100 条重跑：

1. Gemini 3.7 并发定标、全量 direct QA、全量 private-gold judge 均完成；
2. unified-react-v1 的代码、工具结果前递、InteractionSession、原图/候选图传递、
   canary 验收和下游 SFT 分桶均已在服务器复核；
3. 新版 10 条 Agent smoke 工程上通过；3.1 Pro 为 8/10、3.7 Flash 为 4/10，
   其中多条仍有视觉异常越界或证据不足早停，具体问题不在本轮擅自修复；
4. 诊断性新版 Agent-100 已完成，不作为训练数据，不进入 SFT/RL；
5. 训练集大规模教师 rollout、正式 SFT/RL 训练均未启动。

可启动大规模教师 rollout 的输入和边界已经固定：

- 输入：`/gsdata/home/wza/image-factual-verifier-v2-data/datasets/route-aware-hrc-stage2-10563-final-organized-20260824-r2/unified-dataset/train-manifest.jsonl`
- 规模：8,490 条，测试集 1,684 条不进入 rollout/SFT/RL；
- 生产入口：`scripts/trajectory/run_teacher_rollout_autopilot.py`；
- 建议起始 case 并发：16；QA judge 请求并发：64；
- 工程错误自动回队列，外部不可用单独记录，不把测试集 private gold 注入 rollout；
- 启动前仍需人工确认。

## 7. 完成判定

只有同时满足以下条件，才可把本计划标为完成：

- A 的 64/256/512 结果已落盘并据此记录生产并发；
- 3.7 full direct QA 和完整 judge 已结束并有最终汇总；
- C/D 全部改动有日志、测试、提交和 gpu-13 验证；
- 新 Agent-100 已按 2026-09-02 人工要求启动，完成后必须有逐条轨迹问题清单和统一 private-gold 三分类；
- 实验文档、SFT 导出审查和 PSD/RL 前置验证已更新；
- 没有启动大规模教师 rollout。

## 8. 2026-09-02 工具观察续做记录

本轮针对 WebWatcher 风格的工具观察与上下文传递已部署服务器并完成收尾：

- `visit` 保留选中原文段落及必要的前置完整段落；模型看到的是受限的完整来源上下文，
  不是被截断的半句。
- `text_search` 的空结果明确标记为 `empty_results`，不会伪装成可用证据。
- 多页面 `visit` 的每个页面都进入统一 Evidence ledger；不再只保留第一张页面。
- `compare_with_reference` 对无效图片、登录页、验证码页和 SVG 返回
  `invalid_reference` / `external_unavailable`，不生成比较证据。
- 当前 ReAct 下一轮收到完整 canonical `function_result`；状态只保留索引和有限的近期证据，
  不重复叠加累计 workspace。
- 原始 provider/tool 结果仍写入 archive；`validated_claim_state` 和
  `agent_control_state` 不再进入当前 ReAct 模型上下文。
- 新增回归测试覆盖上述边界；服务器全量测试为 `468 passed`，`compileall` 和
  `git diff --check` 已通过。

上述工具观察改动已由 `55ef0b3` 部署并完成服务器回归；3.1 Pro / 3.7 Flash smoke
的具体结果和逐条问题已写入：

`docs/reports/2026-09-02-perceive-scene-recovery-and-smoke.md`

## 9. 2026-09-02 当前执行状态

- 3.1 Pro smoke：10/10，0 provider 工程错误，8/10 对 private gold；
- 3.7 Flash smoke：10/10，0 provider 工程错误，4/10 对 private gold；
- 两组 smoke 均出现外部访问失败记录，但没有 `perceive_scene` 400/超时；
- 发现的问题统一作为诊断记录，当前不擅自改变 Agent 结构或 prompt；
- 新版 Agent-100 正式输入：
  `/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini37-agent-test100-newagent-release-20260902-r55ef0b3/runtime-release/runtime_input/cases.jsonl`
- 新版 Agent-100 rollout 输出：
  `/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini37-agent-test100-newagent-rollout-20260902-r55ef0b3/`
- rollout 并发：10；工程失败自动进入现有 attempt 队列，最多 4 次；最终 99 条成功，
  1 条 Gemini `content_blocked`；
- 统一 private-gold Agent judge 已完成：34 条正确且理由充分、40 条正确但依据不足、
  25 条判断错误；逐条审阅产物见 `docs/reports/2026-09-02-agent100-newagent-rollout-and-review.md`；
- 该 100 条不进入训练；
- 大规模训练集教师 rollout 仍未启动。

当前没有残留 rollout、judge 或旧 Jupyter kernel 进程。3.7 若暂时不可用，只影响新的
3.7 实时请求，不影响上述归档结果、文档维护或本地测试；在人工复核完成前不会因此启动
8,490 条教师 rollout。

## 10. 文档收尾结论

- 当前有效代码基线：运行时代码 `55ef0b3`，仓库/服务器最新 HEAD `10bb1e1`。
- 当前有效 Agent：`unified-react-v1`；旧 v4 只作为历史归档，不作为兼容运行路径。
- 当前训练输入仍固定为 8,490 条 `train-manifest.jsonl`；1,684 条测试集不进入
  teacher rollout、SFT 或 RL。
- 新版 Agent-100 仅用于调查质量诊断和统一 private-gold 审计，不进入训练。
- 大规模教师 rollout 的命令、数据边界、重试和分桶规则已记录，但在人工复核完成前不启动。

# Agent 与评测收尾执行计划

**状态：进行中。** 本文是当前工作的连续执行记录。每次涉及 Agent 的代码、prompt、schema、
harness、审计口径或运行配置变更，必须先在第 4 节追加一行，再实施、测试、提交和部署。

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

- 当前运行中的 Gemini 3.7 direct-QA 补跑不停止；压测反映其并行存在时的真实容量。
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
| 最新服务器 commit | `f383cc0 feat: add Gemini concurrency benchmark` |
| 测试集 | 1,684 条，real 447 / fake 1,237 |
| 3.1 Pro full direct QA | 1,682 完成、2 工程错误；judge 已完成 |
| 3.7 full direct QA | 自动补跑中；完成数随恢复变化，尚未做全量 judge |
| 3.7 Agent-100（旧版） | 100 条与 report sidecar 已完成；仅作为旧基线 |
| 大规模教师 rollout | 禁止启动，等待本计划全部完成后的人工确认 |

## 4. Agent 变更日志（从现在起持续追加）

| 时间（UTC） | 类别 | 文件/范围 | 改动与原因 | 验证/部署 | commit |
|---|---|---|---|---|---|
| 2026-08-30 20:55 | 基准设施 | `scripts/benchmark_gemini_interactions_concurrency.py` | 新增可复用 Gemini 并发基准；混合结构化文本、原生工具调用、图片结构化请求；强制零重试，并显式设置 request gate 与 HTTP 连接数，避免 256/512 档被默认 128 连接限制悄然串行化。不是 Agent 语义改动。 | 本地 `py_compile`、`--help`；gpu-13 fast-forward 后 2 并发/4 请求真实 smoke 为 4/4 成功。 | `f383cc0` |
| 2026-08-30 21:12 | 已实施：ReAct prompt | `src/orchestrator/unified_prompts.py` | 将“反向搜图得到匹配后”改为“返回未验证候选”；补一条自然的具体问题导向 query 指引。未新增固定槽位、query 拒绝器或真假倾向。 | 定向 prompt 断言通过；真实 smoke 待部署后运行。 | 待提交 |
| 2026-08-30 21:12 | 已实施：参考图 adapter | `src/orchestrator/unified_react.py` | 从 runtime `InvestigationDiscovery(reference_image_url, candidate_url)` 建立参考图→候选页面映射；compare 调用没有显式 `source_page_url` 时自动绑定。成熟 compare 工具不改。 | adapter 注入/显式覆盖单测通过；真实 reverse-search smoke 待运行。 | 待提交 |
| 2026-08-30 21:12 | 已实施：候选 URL 与下载诊断 | `src/tools/reverse_image_search.py`、`src/tools/crop_and_search.py`、`src/tools/compare_reference.py` | 接受合法 HTTP(S) 的无扩展名图片候选，按响应 `Content-Type` 验图；把直接下载、URL 变体、候选页、页面图片提取、HTTP/网络/策略失败写入 subcalls。 | extensionless 与诊断 subcall 单测通过；真实下载回退 smoke 待运行。 | 待提交 |
| 2026-08-30 21:12 | 已实施：审计输出上限/主分类 | `scripts/audit_direct_qa_baseline.py`、`scripts/audit_agent_private_gold.py`、`src/eval/private_gold_metrics.py` | private-gold judge 默认输出上限升至 8,192；Agent 主三分类改为决定性、落地依据 / 正确但不足 / 错误，旧严格覆盖度仅保留诊断。 | Agent/Direct 分类差异单测通过；真实审计待部署后运行。 | 待提交 |
| 2026-08-30 21:12 | 已核实：无需重复改 | `src/orchestrator/task_store.py`、`src/tools/compare_reference.py` | 当前代码已有失败/低相关候选的有界 sibling 控制；compare 已从 typed `differences` 派生并保留 `edit_evidence_*` 的原始字段与修复记录。先以真实 smoke 验证，不重复造同类补丁。 | 待真实 smoke 复核。 | 既有代码 |
| 2026-08-30 21:50 | 待实施：审计去重 | `scripts/audit_direct_qa_baseline.py` | 3.7 direct QA 采用追加式补跑后，审计器直接读取全部 `results.jsonl`，会重复审计同一 case 的旧失败记录和新成功记录；改为按 `case_id` 只取最后一条结果，再进入 judge。 | 新增重复结果回归测试；用 3.7 全量 QA 重新执行一轮干净 private-gold audit。 | 待提交 |

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

当前结果（与正在运行的 3.7 direct-QA 补跑并行）：

| API 并发 | 请求数 | 成功 | 原始成功率 | 排除内容拦截后 | 成功返回/min | p90 | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 512 | 510 | 99.61% | 99.61% | 196.83 | 77.53s | 已完成；2 个 Gemini HTTP 错误 |
| 256 | 512 | 484 | 94.53% | 94.53% | 382.11 | 31.55s | 已完成；28 个 `ReadError` |
| 512 | 512 | 502 | 98.05% | 98.05% | 372.41 | 39.04s | 已完成；9 个 `ReadError`、1 个 Gemini HTTP 错误 |

定标结论：256–512 的吞吐已进入约 370–382 successful returns/min 的平台区，且本轮没有
内容拦截。生产不直接采用 256/512：先以 **QA judge 请求并发 64**、**Agent case 并发 16**
作为保守起点，分别通过真实 judge 小样本和 10 条 Agent smoke 后再决定是否提高。

### B. 完成 Gemini 3.7 full direct QA 与 judge

1. 等待已有自动恢复补齐 1,684 条，检查最新记录是否每个 case 恰好一个 terminal 状态。
2. 重新生成 full direct-QA 的 Accuracy、BACC、real/fake 召回和混淆矩阵。
3. 将 direct-QA judge 的输出上限固定为 8,192 后再启动 3.7 judge，防止 `high` thinking
   消耗 2,048 上限造成结构化 JSON 截断。
4. 使用 A 中测出的安全并发，运行 1,684 条 private-gold judge；输出三分类、工程错误和
   `reason_quality` / `fact_match` 诊断。
5. 与 3.1 Pro full result 同口径比较，主看 BACC 与两类召回；Accuracy 不能单独定优劣。

### C. unified Agent 修复

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

### D. Agent private-gold 审计与报告

1. Agent 100 条的主三分类采用：正确且有决定性、落地的实际依据；正确但依据不足；判断错误。
2. 旧的“严格 private target 全覆盖”计数只保留为内部诊断，不作为主表或 gate。
3. Agent 的第一类要求 selected Evidence、verdict basis 和 fact-check report 一致；direct QA
   的第一类只评价其图片直答的 `core_fact/reason`，不能称为检索证据。
4. 所有 private-gold judge 默认输出上限至少 8,192；审计结果 JSON 被截断时不能计为语义失败。

### E. 真实验收与新 Agent-100

1. 为 C/D 的每项补定向单测；本地运行相关 pytest、`compileall` 与 `git diff --check`。
2. 每个提交 push 后，gpu-13 fast-forward；服务器运行定向测试。
3. 先用 10 条真实 Gemini 3.7 smoke 验证 trace：候选标识、页面回退、失败记录、工具 contract、
   终止与 report 均正常。
4. 再对固定测试集 100 条各跑 **一条** 新 Agent 完整轨迹；并发由 A 决定。这里不做四选一，
   以便干净比较新旧 Agent。
5. 对这 100 条运行 Gemini 3.7 private-gold Agent judge，输出 Accuracy、BACC、两类召回、
   三分类、工程错误和逐条失败原因。
6. 对比旧 Agent-100、3.7 direct-QA-100、3.1 Pro direct-QA-100，并人工抽看全部 100 条
   新 Agent trace 的调查过程是否合理。

### F. 文档、SFT 与 PSD/RL 前置收尾

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

## 6. 完成判定

只有同时满足以下条件，才可把本计划标为完成：

- A 的 64/256/512 结果已落盘并据此记录生产并发；
- 3.7 full direct QA 和完整 judge 已结束并有最终汇总；
- C/D 全部改动有日志、测试、提交和 gpu-13 验证；
- 新 Agent-100 轨迹、report、judge、逐条检查和对比表均已完成；
- 实验文档、SFT 导出审查和 PSD/RL 前置验证已更新；
- 没有启动大规模教师 rollout。

# Visual Fact Discrepancy Agent v4 实施计划

日期：2026-07-17
状态：执行中
设计合同：`2026-07-17-discrepancy-first-v4.md`
交互顺序：`docs/gemini-interaction-sequence.md`

> 本文保存用户确认的独立执行计划。详细服务器命令与验收边界以本次接手消息为准；实施时不得跳过确定性门禁直接调用真实 Gemini。

## 当前执行进度（2026-07-18）

- Phase 0～7：本地实现完成。默认 workflow/release policy 已切换为
  `discrepancy-first-v4`；公开 `Orchestrator.run()` 不再执行 v3。冻结 v3 仅由测试专用回放
  harness 调用旧 schema/reducer。
- Phase 1 确定性门禁：原子 Planning/Decision reducer、非法输出零修改、稳定 ID、
  序列化往返、Evidence/Claim/Hypothesis 所有权、单次视觉复查和 post-verdict 禁止均有测试。
- Phase 2～5：Image Account Planning 原图 root、claim/hypothesis ReAct、稀疏
  Discrepancy Decision、v4 Coverage/basis/Judgment 已通过完整 mock Interactions 轨迹和
  默认 workflow canonical trace strict audit。
- Phase 6：v4 strict audit、`ifv-policy-v2`、Evidence-chain/discrepancy/stop-quality
  scoring 和训练排除门禁已实现。
- Phase 8：完成。已从 gpu-13 只读定位并白名单脱敏四条冻结 v3 成功轨迹；回放 fixture
  不含凭据、private gold 或 provider interaction ID。当前 v4 reducer 对 Andreea、Queen、
  Pillars 和 Monarch 双跑状态完全一致，预期 verdict、Evidence 所有权、discrepancy 对齐、
  及时停止、post-verdict 禁止和 strict audit 均通过。
- Phase 9：未执行。四条历史回放门禁已放行，下一步是提交/同步后运行单条真实 Gemini canary。
- Phase 10：核心文档已更新；尚未提交代码。当前本地门禁为 `370 passed`、
  `compileall` 通过、`git diff --check` 通过。

## 目标

将 v3 的单一核心事实核查运行时迁移为 v4 discrepancy-first 调查链：

```text
原图
→ Image Account
→ 1～3 条高显著性 ImageClaim
→ SearchHypothesis
→ 搜索与视觉 Evidence
→ Gemini 多模态 Discrepancy Decision
→ fake | real | unverifiable
```

核心问题：图像传达的主要事实中，是否存在由证据支持、与图像可见内容绑定的实质性错误或篡改？

## 强制边界

- v3 由 `runtime-v3-final-20260717` 冻结，不再修改。
- v4 分支：`codex/image-factual-verifier-v4`。
- 旧字段只作为迁移残留，不建立 v2/v3 runtime 兼容层。
- Prompt 负责语义判断；确定性代码负责协议、引用、预算、原子更新和终止安全。
- 搜索标题和 snippet 只是 Discovery；可追溯正文、同图比较等才是 Evidence。
- provider/协议故障是工程失败，不得伪装成 `unverifiable`。
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

Planning reducer 必须验证 Claim key、视觉锚点、高显著性 Claim、Hypothesis 引用和预算，创建稳定 ID 与 claim-owned task，并在完整验证后一次性写入。

Decision reducer 必须验证 Claim、Evidence、Hypothesis、视觉锚点及调查归属；验证新增 Hypothesis 和视觉复查预算；discrepancy 必须引用 Evidence；verdict 必须满足前置条件；完整验证后一次性提交。

退出条件：原子性、非法输出零修改、序列化往返稳定。在此之前禁止接真实 Gemini。

### Phase 2：Image Account Planning

- 用 `ImageAccountPlanningOutput` 替换 Target Planning。
- Planning 主 Interaction root 上传原图一次。
- 输入 Perception、OCR、VisualFact、retrieval anchors。
- 生成 1～3 条 ImageClaim；外部识别仅生成 SearchHypothesis。
- 后续通过 `previous_interaction_id` 继承原图上下文。

### Phase 3：Claim/Hypothesis 调查循环

- ResearchTask 必须归属 Claim 和 Hypothesis。
- ReAct 每次只选择一个有限动作。
- 搜索发现可更新 Hypothesis，不自动扩大 ImageClaim。
- 禁止相同语义、工具、目标的重复路线。

### Phase 4：稀疏多模态 Discrepancy Decision

触发：直接支持/反驳 Evidence、同图或同事件比较、累计实质性 Evidence 的 checkpoint、unresolved 终止前。不得每条 Evidence 都调用。

Gemini 可更新 ClaimAssessment、建立 MaterialDiscrepancy、退休/新增有限 Hypothesis、请求一次聚焦视觉复查、提议 verdict。

### Phase 5：停止、Coverage 和 verdict basis

- `fake`：decisive discrepancy + 有效 Evidence + 高显著性 Claim + 视觉锚点，满足即停。
- `real`：所有 high-salience Claim supported；无 established/unresolved decisive discrepancy；Gemini 明确提议 real。
- `unverifiable`：关键 Claim insufficient/conflicted；无 decisive discrepancy；有意义路线耗尽；Gemini 明确提议。
- 保留 24 action、重复路线、Hypothesis、视觉复查预算和 post-verdict 禁止。

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

### Phase 10：文档、清理和提交

更新 `docs/architecture.md`、`docs/agent-prompt-and-runtime-guide.md`、`docs/gemini-interaction-sequence.md`、`AGENTS.md`、README。最终运行：

```bash
python -m pytest -q
python -m compileall -q src scripts
git diff --check
```

单元测试通过不等于 v4 完成；还需真实 canary 和 strict audit。

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

正式执行链不依赖单一 core fact；Planning/Decision/调查/停止可追溯且有界；fake、real、unverifiable 前置条件明确；历史回放通过；至少一条真实 canary 及另外 3～4 条异构 case 经人工和 strict audit 验收；无样例专用规则；可稳定导出教师轨迹与训练数据。在此之前不进入 Qwen-VL 批量轨迹生产或训练。

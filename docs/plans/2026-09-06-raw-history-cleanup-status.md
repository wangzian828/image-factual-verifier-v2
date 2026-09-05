# Raw-history Runtime 清理进度

更新时间：2026-09-06

## 当前边界

当前分支为 `codex/gpu13-canary-20260804-plan-relaxation-01`。生产入口是：

```text
src/orchestrator/pipeline.py
  -> src/orchestrator/react_runtime.py
  -> src/orchestrator/stage_runner.py
  -> provider-side cumulative interaction history
  -> src/orchestrator/react_runtime.py 的 raw observation basis
  -> unified Judgment
```

当前 runtime state 只保留 `schema_version`、case/image identity、objective、动作计数、
停止原因和完成说明。工具观察只从 `state.all_steps` 的原始 action/result 读取。

## 已完成检查点

- `383d6a0`：raw-history runtime 大规模迁移；SFT、审计、导出和 private-gold 投影切换到
  新 schema 的第一版。
- `131d167`：SFT raw-observation eligibility gate；空搜索可记录但不能单独支持 fake，
  `finish_investigation` 不作为可引用 observation。
- `c4c9141`：过程评分切换为 raw-history-only，删除旧 graph/reducer scorer。
- 工作树未提交清理：`StageRunner` 移除旧 `StageHandoffPacket` 生成/渲染，查看器和 readable
  renderer 不再展示每步旧 `investigation_state_update`。

## 残留盘点

### 当前入口内已不再使用

- `src/orchestrator/unified_react.py`：旧 graph/reducer 状态、工具 adapter 和 delta 生成。
- `src/orchestrator/task_store.py`：旧 VisualFact/ResearchTask/Evidence 图存储与路线控制。
- `src/orchestrator/progress_control.py`：旧 graph progress ledger。
- `src/orchestrator/discrepancy_coverage.py`：旧 Coverage/Discrepancy reducer 语义。
- `src/orchestrator/context_workspace.py`：旧 stage handoff/workspace packet。
- `src/orchestrator/unified_context.py`：旧 graph 上下文编译器。

这些模块仍被历史测试、replay/backfill 脚本或文档引用；删除前必须先给历史工具明确归档
边界，不能恢复它们到生产入口。

### 仍需处理的代码级残留

- `scripts/audit_real_trace.py` 仍包含不可达的 v3/v4 审计函数；active `audit_trace()` 已只
  调用 raw-history 分支，但需要迁移/移出历史测试后再删除死代码。
- `src/trajectory/exporter.py` 仍保留旧 v4 chain helper；active raw-history quality gate 已
  独立，需进一步确认 reference-chain evaluator 的依赖后再收缩。
- `src/trajectory/reference_chain.py` 仍从 `scoring.py` 导入若干通用匹配 helper；这是评估
  旁路，不是 runtime reducer，但需要单独验证不会把旧 graph 字段带入 active release。
- `StageRunner` 的非 Interactions fallback 仍有 `evidence_so_far` 文本摘要逻辑；需确认它只
  属于 provider fallback，不能混入 raw-history canonical state 或替代原始结果。
- viewer/readable renderer 仍保留旧 trace 的兼容展示分支；当前 raw-history 分支不读取旧
  graph state。

## 测试状态

- `python -m compileall -q src scripts training/ifv_training`：通过。
- raw-history/SFT/export 聚焦回归：`36 passed, 2 failed`；2 个失败夹具仍写旧
  `ifv-unified-react-v1` schema，待迁移到 raw-history fixture。
- 直接运行 `test_react_runtime.py` 会因测试引用已删除的 `reduce_react_action` 导入失败；
  该文件属于旧 reducer 契约，当前默认回归已移出，不能通过恢复旧 API 修复。
- native StageRunner 仍有预置 response 耗尽后出现额外 correction request 的测试失败，需单独
  区分 fixture 旧假设与当前 structured-output 行为。

## 后续工作

1. 迁移 active exporter/eval/native StageRunner fixtures，不恢复旧 reducer API。
2. 收缩 raw-history strict audit 和 exporter 的不可达 v3/v4 分支。
3. 完成服务器一键 teacher rollout → 10 条训练 smoke → strict audit → frozen SFT eligibility
   → provider-neutral export → ms-swift policy/perception 双包 audit。
4. 制作本地 Direct QA 对比实验包，包含测试图片、manifest、private gold/sidecar、prompt、
   runner/audit、Windows/Linux 启动脚本、`.env.example`、README 和哈希 manifest。
5. 对 10 条训练 trace 逐条检查 verdict、warning、strict audit、SFT eligibility，以及连续
   无相关搜索是否完整保留。

# Image Factual Verifier

当前生产策略是 `unified-react-v1`：一个大的 raw-history ReAct loop，模型每轮选择一个
真实工具动作；runtime 只负责权限、动作预算和终止条件，原始观察保留在 canonical trace。

```text
空 runtime state
  -> provider-side cumulative InteractionSession
  -> ReAct：thought -> 一个 native tool -> raw observation
  -> Judgment 读取完整保留的 tool history
  -> canonical trace
```

当前代码入口：

- Agent：`src/orchestrator/pipeline.py`
- raw-history runtime：`src/orchestrator/react_runtime.py`
- provider interaction runner：`src/orchestrator/stage_runner.py`
- 当前 prompt（英文）：`src/orchestrator/unified_prompts.py`
- prompt 备份：`docs/active-agent-system-prompts.md`（精确英文）、`docs/active-agent-system-prompts-zh.md`（中文对照）
- 历史 graph 上下文：`src/orchestrator/unified_context.py`（当前入口不调用）
- 教师 rollout：`scripts/trajectory/run_teacher_rollout_autopilot.py`
- 轨迹审计：`scripts/audit_real_trace.py`
- SFT 导出：`src/trajectory/exporter.py`

旧 v3/v4 graph、reducer、handoff 和 replay 代码、测试及研究记录已在
`legacy-v4-pre-unified-cleanup-20260827` 标记及历史文档中保留，不属于当前生产入口。

当前 raw-history 清理进度见
`docs/plans/2026-09-06-raw-history-cleanup-status.md`。

本地校验：

```powershell
python -m compileall -q src scripts training/ifv_training
python -m pytest -q test_unified_react.py test_workflow_lifecycle.py
python -m pytest -q training/tests
```

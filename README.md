# Image Factual Verifier

当前生产策略是 `unified-react-v1`：一个大的 ReAct loop，模型每轮选择一个真实工具动作；
runtime 负责状态、权限、预算、结果归并和终止条件。

```text
空 workspace
  -> perceive_scene / ocr_with_position（模型选择顺序）
  -> 首个调查工具 + investigation_intent
  -> ReAct：thought -> 一个 native tool -> observation/state delta
  -> 低频 Reflection / Discrepancy Decision
  -> Judgment
  -> canonical trace
```

当前代码入口：

- Agent：`src/orchestrator/pipeline.py`
- ReAct reducer：`src/orchestrator/unified_react.py`
- 当前 prompt：`src/orchestrator/unified_prompts.py`
- 紧凑上下文：`src/orchestrator/unified_context.py`
- 教师 rollout：`scripts/trajectory/run_teacher_rollout_autopilot.py`
- 轨迹审计：`scripts/audit_real_trace.py`
- SFT 导出：`src/trajectory/exporter.py`

旧 v4 代码、测试和研究记录已在 `legacy-v4-pre-unified-cleanup-20260827` 标记及历史文档中
保留，不属于当前生产入口。

本地校验：

```powershell
python -m compileall -q src scripts training/ifv_training
python -m pytest -q test_unified_react.py test_workflow_lifecycle.py
python -m pytest -q training/tests
```

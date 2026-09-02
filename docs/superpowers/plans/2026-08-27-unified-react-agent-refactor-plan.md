# unified-react-v1 主流程重构记录

日期：2026-08-27
状态：历史重构计划，已完成并由
`docs/superpowers/plans/2026-08-31-unified-react-runtime-refactor-plan.md`
和 `docs/agent-evaluation-closeout-plan.md` 收尾。本文保留当时的设计过程，
不作为当前运行时规范；当前以 `unified-react-v1` 的活动文档为准。

## 目标

把 IFV 收敛为一个大的 ReAct Agent：模型每轮决定一个工具动作，runtime 负责工具权限、状态、
预算、去重、结果归并和终止条件。路线变化不再通过独立的 Replan 请求表达。

## 当时的设计流程

```text
空 workspace
  -> ReAct 选择 perceive_scene / ocr_with_position
  -> ReAct 首次调查动作携带 investigation_intent
  -> reducer 创建一个 target_fact 和 2–3 条候选 route/task（provider 只提交一个主 route）
  -> ReAct：thought -> 一个 native tool -> observation/state delta
  -> 低频 unified_reflection
  -> 低频 unified_discrepancy_decision
  -> unified_judgment
```

## 已实现

1. `Orchestrator.run()` 只进入统一 ReAct 主流程；
2. scene 和 OCR 由模型选择顺序，但依赖未满足前不开放外部工具；
3. 首个调查动作通过 `investigation_intent` 建立正向 target fact 和 2–3 条候选路线；
4. 后续换 query、候选页或视觉方向直接成为下一轮 action；ReAct 可在未完成 route/task 之间切换；
5. 新增紧凑上下文 renderer，完整 workspace 只在 archive 保存；
6. prompt、schema、audit、exporter、reward 使用 `unified-react-v1`；
7. SFT 导出使用 Qwen `<think>` + 原生 `<tool_call>`，无 provider thought 的轨迹进入 action-only；
8. 当前活动文档已改为统一流程；
9. 旧工作树已打标签 `legacy-v4-pre-unified-cleanup-20260827`，旧材料不参与当前生产。

## 不改动

成熟工具及其 provider、图片上传/压缩、OCR、网页提取、缓存和重试契约不在本次主流程重构中
改动。若工具自身仍有问题，单独按工具 issue 处理。

## 验收顺序

1. 本地 compileall、focused tests、training tests；
2. 检查当前生产代码和活动文档不再引用旧主流程；
3. 提交并 push 当前分支，按运维文档更新服务器 checkout；
4. 服务器真实 Gemini 并发 10 跑 10 条；
5. 逐条检查 scene/OCR action、thought、工具调用、state delta、工程错误和 strict audit；
6. smoke 通过后才恢复全量 teacher rollout。

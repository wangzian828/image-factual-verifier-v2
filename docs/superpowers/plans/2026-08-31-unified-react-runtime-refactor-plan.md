# 统一 ReAct 主流程重构计划

日期：2026-08-31

## 目标

把旧的“先生成 Claim/route/task，再围绕 Claim 调查”改成一个标准的
image-grounded ReAct loop：

```text
原图 + 固定事实核查任务
  → thought
  → 一个工具调用
  → 工具结果进入紧凑记忆
  → 下一轮继续选择动作
  → Judgment 输出二分类和 fact-check report
```

## 已完成

- [x] 新增 `UnifiedReactState`、runtime tool adapter 和 reducer。
- [x] `pipeline.py` 主入口切换到统一 ReAct；视觉工具不再固定顺序。
- [x] 每轮请求重新附加压缩原图，不把图片或完整 workspace 叠加进历史。
- [x] 统一记录 `visual_memory`、Discovery、Evidence、Failure、访问/query 历史、
      action budget 和 state delta。
- [x] 隐藏成熟工具的内部字段；模型只看到公开参数。
- [x] 反向搜图候选保持 `unverified`；无效参考图不会进入有效比较。
- [x] 外部访问失败与工程契约错误分开记录。
- [x] 新 trace 审计分支完成，禁止当前 state 混入旧 Claim graph。
- [x] exporter 保留完整 episode，按 Qwen 格式输出 `<think>` 和 tool call。
- [x] semantic reward 与 Agent private-gold projection 已支持无 Claim 的当前 state。
- [x] 修复 SFT/reward 对 runtime `excerpt`、`evidence_class`、`open_questions` 的读取。
- [x] 重新生成英文 prompt 备份并同步中英文运行文档。
- [x] 本地回归：`pytest -q` 通过；`compileall` 和 `git diff --check` 通过。

## 当前收尾

- [x] 提交并 push 当前工作树，记录最终 commit：运行时代码
      `55ef0b3`，当前文档 HEAD `10bb1e1`。
- [x] 按 gpu13 运维文档更新服务器 checkout，确认分支、commit 和
      `OMP_NUM_THREADS=1`。
- [x] 用真实 Gemini 跑之前约定的 10 条 smoke，检查工程错误、工具路线、视觉
      上下文、最终 report、trace audit 和 private-gold 结果。
- [x] smoke 完成并记录；结果进入人工复核边界。大规模教师 rollout 尚未恢复，
      仍需人工确认。

如果 Gemini 3.7 当前不可用，只暂停新的 3.7 实时调用，不影响上述已完成记录、
本地测试和文档收尾。

## 约束

- 当前 active runtime 不创建或依赖 `target_facts`、`search_hypotheses`、`tasks`
  和 claim ownership。
- 旧模型/旧审计代码只作为明确标记的 legacy 回放边界，不得被主流程调用。
- 不修改成熟工具的内部语义；只在 runtime adapter 和 reducer 做边界适配。
- 真实 smoke 期间不把测试集混入训练集，也不启动大规模教师 rollout。

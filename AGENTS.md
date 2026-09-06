# IFV 开发约定

## 当前入口

当前工作树维护 `unified-react-v1` raw-history runtime：

```text
公开 case -> 机械 runtime state -> provider interaction history
-> unified ReAct（每轮一个 native tool call）
-> 原始 tool result 追加到 canonical trace -> Judgment 读取 raw history
```

当前入口和职责：

- `src/orchestrator/pipeline.py`：统一 Agent runtime；
- `src/orchestrator/react_runtime.py`：runtime 状态、工具适配、Judgment 上下文；
- `src/orchestrator/stage_runner.py`：provider interaction 与工具 roundtrip；
- `src/orchestrator/unified_prompts.py`：当前 agent prompts；
- `scripts/audit_real_trace.py`：trace strict audit；
- `src/trajectory/exporter.py`：Qwen SFT 导出；
- `scripts/trajectory/run_teacher_rollout_autopilot.py`：批量 rollout。

历史 graph/reducer、handoff packet、固定 Planning/Replan/Reflection、隐式 perception
预执行和 `src/orchestrator/unified_context.py` 不属于当前入口。不要为了旧兼容重新接入；
路线调整直接表现为下一轮 ReAct 工具调用，成熟工具契约保持不变。

## 不可破坏边界

- 不创建或依赖 `target_facts`、`image_claims`、旧 graph 节点或内部三态语义；
- 模型不能读取 gold、标签、构造信息或 judge 结果；
- 每轮只接受一个 native tool call；runtime 只负责机械计数、停止原因和完成说明；
- 成功、空结果、外部不可用和 malformed tool result 都原样保留在 `state.all_steps`；
- 工具错误、访问失败和空搜索不能被静默丢弃或改写成事实结论；
- archive 保留原始请求、响应和状态，并复用 provider-side interaction history；
- 训练数据、图片、轨迹、日志和 checkpoint 留在服务器；本地工作树只修改代码和文档，
  除非任务明确要求制作迁移或对比包。

## 工作方式

- 只读取当前任务需要的文件和文档；不要把完整仓库扫描写成每次任务的前置条件；
- 小改动先做针对性验证；只有受影响时才扩大到全链路测试；
- 对明确隔离、可丢弃的本地测试，可直接运行并修复请求变更造成的失败；
- 涉及 runtime、工具契约、trace、SFT 或 prompt 的变更，验证对应链路；
- 不修改历史 fixture、评测基线或旧实现，除非任务明确要求；不为旧兼容层堆新分支。

## 交付标准

普通代码变更至少运行：

```powershell
python -m compileall -q <受影响的源码目录>
git diff --check
```

涉及真实 runtime、轨迹或 SFT 发布时，再做真实 provider smoke、逐条 strict audit 和目标
processor 验证；mock 测试不能替代真实验收。提交、分支和服务器数据位置由任务决定，不自动
创建提交或分支。

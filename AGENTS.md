# IFV 当前开发约定

## 当前主流程

当前工作树只维护 `unified-react-v1`：

```text
公开 case
  -> 空 workspace
  -> unified ReAct（每轮 thought + 一个工具调用）
  -> reducer 写入 observation / state delta
  -> 低频 Reflection / Discrepancy Decision
  -> Judgment
```

不要恢复固定 Planning、Query Replan、Route Replan 或隐式 perception 预执行。路线调整直接
表现为下一轮 ReAct 工具调用；成熟工具本身的输入、输出、上传、超时和重试契约不改。

## 当前字段与边界

- 当前目标字段是 `target_facts`，不要新增 `image_claims` 别名；
- 模型不能读取 gold、标签、构造信息或 judge 结果；
- 搜索摘要和反向搜图结果是 Discovery，不是 Evidence；
- 外部不可访问、空结果和 malformed tool result 都记录为可恢复观察失败；只有
  原图/case/持久化状态或 worker 级无法恢复故障才是工程错误；
- 每轮只接受一个 native tool call；状态只能由 reducer 写入；
- 完整 archive 保留原始请求、响应和状态，模型上下文只使用紧凑 handoff。

## 修改前检查

```powershell
git branch --show-current
git status --short --branch
git log -1 --oneline --decorate
```

工作树必须是 `codex/gpu13-canary-20260804-plan-relaxation-01`。旧实现已在
`legacy-v4-pre-unified-cleanup-20260827` 标记；旧研究材料只作历史参考，不进入当前入口。

## 主要入口

- `src/orchestrator/pipeline.py`：统一 Agent runtime；
- `src/orchestrator/react_runtime.py`：当前 ReAct 状态、动态工具 adapter 与状态登记；
- `src/orchestrator/unified_react.py`：未激活的历史图状态实现，不作为当前入口；
- `src/orchestrator/unified_prompts.py`：当前四类英文 agent prompt；
- `src/orchestrator/unified_context.py`：紧凑上下文；
- `scripts/audit_real_trace.py`：统一 trace strict audit；
- `src/trajectory/exporter.py`：完整 episode 的 Qwen SFT 导出；
- `scripts/trajectory/run_teacher_rollout_autopilot.py`：批量 rollout。

## 验收

```powershell
python -m compileall -q src scripts training/ifv_training
python -m pytest -q test_unified_react.py test_workflow_lifecycle.py
python -m pytest -q training/tests
git diff --check
```

代码通过后还要用真实 Gemini 做 10 条并发 10 smoke，并逐条审计 trace；mock 测试不替代真实
验收。

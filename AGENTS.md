# IFV 当前开发约定

## 当前主流程

当前工作树只维护 `unified-react-v1` 的 raw-history runtime：

```text
公开 case
  -> 空 runtime state（只含机械字段）
  -> provider-side cumulative InteractionSession
  -> unified ReAct（每轮 thought + 一个工具调用）
  -> 原始 tool result 追加到 canonical trace
  -> Judgment 读取保留的 raw tool history
```

不要恢复固定 Planning、Query Replan、Route Replan、旧 graph reducer、handoff packet
或隐式 perception 预执行。路线调整直接表现为下一轮 ReAct 工具调用；成熟工具本身的
输入、输出、上传、超时和重试契约不改。

## 当前字段与边界

- 当前 raw-history runtime 不创建 `target_facts`、`image_claims` 或旧 graph 节点；历史
  graph 模块中的同名字段不属于当前入口；
- 模型不能读取 gold、标签、构造信息或 judge 结果；
- 每轮只接受一个 native tool call；runtime 只维护动作计数、停止原因和完成说明；
- 成功、空结果、外部不可用和 malformed tool result 都原样保留在 `state.all_steps`；
  工具错误或空搜索不是事实证据；
- 完整 archive 保留原始请求、响应和状态，后续请求复用 provider-side interaction history，
  不再生成旧 `StageHandoffPacket`。

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
- `src/orchestrator/react_runtime.py`：当前 raw-history runtime、机械状态和工具适配；
- `src/orchestrator/unified_react.py`：未激活的历史 graph/reducer 实现，不作为当前入口；
- `src/orchestrator/stage_runner.py`：provider interaction、原始请求/响应和工具 roundtrip；
- `src/orchestrator/unified_prompts.py`：当前四类英文 agent prompt；
- `src/orchestrator/unified_context.py`：历史 graph 上下文实现，当前 raw-history 入口不调用；
- `scripts/audit_real_trace.py`：统一 trace strict audit；
- `src/trajectory/exporter.py`：完整 episode 的 Qwen SFT 导出；
- `scripts/trajectory/run_teacher_rollout_autopilot.py`：批量 rollout。

## 验收

```powershell
python -m compileall -q src scripts training/ifv_training
python -m pytest -q test_sft_eligibility.py test_unified_react_scoring.py test_workflow_lifecycle.py
python -m pytest -q training/tests
git diff --check
```

代码通过后还要用真实 Gemini 做 10 条并发 10 smoke，并逐条审计 trace；mock 测试不替代真实
验收。

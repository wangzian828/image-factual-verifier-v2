# 项目约定

遵循 `AGENTS.md`。当前唯一生产 Agent 是 `unified-react-v1`，使用一个统一 ReAct loop：
每轮 `thought -> 一个 native tool -> observation/state delta`。

状态由 runtime/reducer 管理，模型不直接写 state，不读取 gold。视觉感知和 OCR 是模型可以
选择的工具；二者完成前不开放外部调查工具。换 query 或调查方向直接进入下一轮 ReAct，不
新增独立 Replan 请求。

当前 prompt、上下文和字段分别以以下文件为准：

- `src/orchestrator/unified_prompts.py`
- `src/orchestrator/unified_context.py`
- `src/orchestrator/investigation_models.py`
- `src/orchestrator/unified_react.py`

修改后先跑 focused tests、compileall 和 `git diff --check`，完成后做真实 Gemini 10 条并发
10 smoke。凭据只能来自环境变量或未跟踪的本地配置。

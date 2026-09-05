# Unified ReAct Harness 可恢复错误修复计划

日期：2026-09-05

## 目标

让 Agent 在单个工具失败、返回空结果、返回坏结构或出现一次协议错误时继续调查，
而不是让 worker 直接把整条轨迹判成工程错误。

本次修复只处理 harness、runtime 状态和审计语义，不修改：

- Agent 的事实判断任务；
- Agent prompt 的事实边界；
- 成熟工具本身的业务实现；
- private-gold 标签和 judge 口径；
- 现有 24 次 ReAct 动作预算；
- 最终二分类 Judgment。

## 当前已确认的问题

当前生产入口是：

```text
src/orchestrator/pipeline.py
  _run_react_runtime_policy
    StageRunner
    observation_callback
    react_runtime 的动作状态处理
```

`src/orchestrator/unified_react.py` 是旧的复杂路径，不作为本次主流程基线。

已确认的主要缺口：

1. `perceive_scene` 会在统一失败处理之前调用 `_parse_perception_result()`。
   如果工具返回空 JSON、坏 JSON 或异常结构，解析异常会直接穿透 callback。

2. 普通工具错误虽然已经能被识别为
   `external_unavailable` 或 `provider_error`，但入口层只要在前置解析阶段抛异常，
   就不会进入这套记录逻辑。

3. `engineering_error`、可恢复错误和外部不可用在部分路径上没有统一的生命周期语义，
   容易被外层误认为必须停止。

4. 状态更新和工具结果解析不是事务边界。状态更新中出现异常时，可能既没有安全回滚，
   也没有向下一轮提供结构化观察。

5. 测试主要覆盖正常结果和局部 helper，没有覆盖“工具执行失败后继续到下一动作，
   最后进入 Judgment”的完整链路。

## 目标错误分类

| 类别 | 例子 | 是否消耗调查动作 | Agent 是否继续 |
| --- | --- | ---: | --- |
| `success_with_content` | 有效搜索结果、有效页面段落、有效视觉观察 | 是 | 是 |
| `success_empty` | 搜索无结果、无 OCR 文本、无候选图 | 是 | 是 |
| `external_unavailable` | SSL、超时、验证码、403/404、图片下载失败、429 | 是 | 是 |
| `provider_error` | VLM/搜索 provider 返回可记录的失败 | 是 | 是 |
| `malformed_tool_result` | 空对象、坏 JSON、缺少业务字段但仍可记录 | 是 | 是 |
| `recoverable_state_error` | 本轮状态合并失败但上一状态仍完整 | 是 | 是 |
| `protocol_correction` | 模型工具参数或输出格式不符合 schema | 否 | 同轮纠正，必要时下一轮继续 |
| `fatal_runtime_error` | 原图、case 身份、持久化状态均无法恢复；worker/进程级故障 | 不适用 | 仅终止当前 case |

“工程错误”只保留给真正无法恢复的 runtime 故障。一次模型 schema 错误、工具坏结果
或状态合并异常不能直接让 worker 崩溃。

## 设计原则

### 1. 先统一归一化，再做专用解析

每个已执行工具结果先经过统一入口：

```text
raw provider result
    ↓
result envelope normalization
    ↓
success / empty / external_unavailable / malformed / provider_error
    ↓
tool-specific parser only on usable payload
    ↓
state update
    ↓
next ReAct request
```

专用解析器不能再决定整个 worker 是否退出。

对于 `perceive_scene`：

- 有效 JSON 才调用感知模型解析器；
- 空对象、坏 JSON、provider 异常统一转为结构化失败观察；
- 保留原始错误摘要和 call ID；
- 不覆盖上一轮仍有效的视觉记忆；
- 允许 Agent 改用 `focused_visual_inspection`、`crop_and_inspect`、OCR 或搜索。

### 2. 外部不可用不改变 ReAct 生命周期

SSL、验证码、超时、下载失败、429 和空搜索结果都要：

- 写入当前轨迹的 failure ledger；
- 在下一轮以简短、明确的工具结果进入模型上下文；
- 标记当前候选或 URL 不可用，避免无意义重复；
- 保持剩余工具和动作预算；
- 继续调查；
- 如果后续没有路线或达到 24 次动作，进入现有 Judgment。

不能设置 `stop_reason=engineering_error`，也不能抛出未捕获异常。

### 3. 协议错误与工具错误分开

- 模型还没有成功执行工具前的 schema 错误：不计调查动作，沿用现有 bounded correction。
- 工具已经执行但返回坏结构：计一次已执行动作，记录结果后继续。
- 纠错预算耗尽：不伪造成功工具结果；结束当前 ReAct 段并回到外层循环，
  外层继续其它动作或按现有预算进入 Judgment。

### 4. 状态更新必须可回滚

每次动作使用临时副本执行：

```text
last_valid_state
    ↓ deep copy
candidate_state
    ↓ normalize + parse + update + validate
success → commit candidate_state
failure → discard candidate_state, keep last_valid_state,
          append recoverable failure record
```

回滚时必须保留：

- 原始工具结果摘要；
- tool name；
- function call ID；
- 错误分类；
- 是否恢复；
- 下一轮可用工具；
- 当前动作计数。

### 5. 终止只由统一生命周期决定

ReAct 只在以下情况停止：

- Agent 成功调用 `finish_investigation`；
- 达到现有 24 次动作预算；
- 没有可用工具；
- 无法恢复原图、case 身份或持久化状态；
- worker/模型服务/控制面本身不可用。

可恢复工具失败不能触发停止。

## 实施步骤

### Phase 1：建立统一结果边界

涉及：

- `src/orchestrator/pipeline.py`
- `src/orchestrator/react_runtime.py`
- 必要时 `src/orchestrator/tool_result.py`

工作：

1. 抽出统一的工具结果分类和 envelope。
2. 将 `perceive_scene`、OCR 和其它工具的前置解析改为安全解析。
3. 统一识别空结果、坏 JSON、provider error 和 external unavailable。
4. 保留原始错误但限制进入上下文的长度。
5. 不改变成熟工具的输入输出业务含义。

验收：

- 任意工具返回空对象不会抛出未捕获异常；
- `perceive_scene` 失败后仍能进入下一轮；
- 外部失败不会被标成 fatal engineering error。

### Phase 2：修复 ReAct 续跑和状态回滚

涉及：

- `src/orchestrator/pipeline.py`
- `src/orchestrator/react_runtime.py`

工作：

1. 将当前动作处理改为“归一化 → 安全解析 → 临时状态更新 → 提交”。
2. 解析或状态更新失败时保留上一份有效状态。
3. 为失败动作生成下一轮可见的简短观察。
4. 保证失败候选、失败 URL 和失败 query 不会被重复选择。
5. 确保动作计数、工具预算和 `stop_reason` 一致。
6. 代码内部不再使用语义上容易误解的旧 reducer 命名；如需保留兼容调用，
   仅保留薄的内部别名，不再把它描述成独立阶段。

验收：

- 工具失败后 Agent 至少能继续选择一个其它可用动作；
- 连续失败直到预算耗尽时仍能正常进入 Judgment；
- 成功恢复的轨迹不被标成 engineering error；
- 原图和已有有效观察不会因单次失败丢失。

### Phase 3：检查 StageRunner 协议边界

涉及：

- `src/orchestrator/stage_runner.py`
- active ReAct pipeline 调用点

工作：

1. 检查同轮 schema correction 的父子请求和工具结果前递。
2. 确保 forced boundary 不会制造“没有 accepted action”的假失败。
3. 区分：
   - 模型没有完成动作；
   - 工具动作已执行但结果不可用；
   - provider 请求本身失败。
4. 保留原始 rejected output 和 correction 信息供审计，
   但不让可恢复情况终止 worker。

验收：

- 单次 schema 错误可纠正；
- 连续纠错耗尽后能安全回到 ReAct 外层；
- 不出现 `ReAct did not complete one accepted action` 的假工程错误。

### Phase 4：审计和记录语义

涉及：

- `scripts/audit_real_trace.py`
- `scripts/monitor_runtime_events.py`
- 相关轨迹导出和状态记录代码

工作：

1. 将 recoverable failure、recovered warning 和 fatal engineering error 分开。
2. strict audit 不把正常外部不可用当作工程失败。
3. 如果最终成功进入 Judgment，记录：
   - `termination=success`；
   - `recoverable_failure_count`；
   - `unrecovered_external_failure_count`；
   - `fatal_engineering_error=false`。
4. 保留 raw error、工具名、请求 ID 和恢复路径。

验收：

- 轨迹可审计；
- 失败原因不会被“成功”字段覆盖；
- 外部失败不会污染工程错误统计。

### Phase 5：回归测试

新增或扩展测试，至少覆盖：

1. `perceive_scene` 返回 `{}`；
2. `perceive_scene` 返回坏 JSON；
3. OCR 返回空结果或残片；
4. `text_search` 返回空结果；
5. `visit` 返回 SSL/验证码/超时；
6. 参考图下载失败；
7. 工具结果字段缺失；
8. 工具结果解析异常；
9. 状态更新异常后回滚；
10. 失败后继续下一动作；
11. 失败累计到动作上限后进入 Judgment；
12. schema correction 后继续；
13. correction 耗尽后不伪造 accepted action；
14. 失败路径不丢失原图和上一轮有效工具结果；
15. HTTP/工具 session 在成功、异常和取消时都关闭。

门禁：

```text
pytest 全量
compileall
git diff --check
strict trace audit fixtures
```

### Phase 6：真实验证

严格按以下顺序：

1. 本地测试全部通过；
2. 提交并 push；
3. gpu-13 checkout fast-forward 到准确提交；
4. 先运行一个包含已知外部失败风险的单 case；
5. 再运行当前 10 条 smoke；
6. 逐条检查工具失败后的下一轮是否继续；
7. 检查最终是否正常进入 Judgment；
8. 运行 strict audit；
9. 统计：
   - terminal success；
   - recoverable failures；
   - recovered failures；
   - fatal engineering errors；
   - 未完成 case；
   - 平均动作数和工具调用数。

本阶段不启动全量教师 rollout，不重跑大规模 SFT/RL。

## 成功标准

满足以下条件才算完成：

- 外部工具失败不再导致 Agent 提前停止；
- `perceive_scene` 空/坏结果不再直接升级为工程错误；
- 状态更新异常可以回滚并继续；
- 协议纠错、工具失败和 fatal runtime 故障统计分离；
- 真实 smoke 中不出现假 engineering error；
- strict audit 通过；
- 原始 trace、失败记录和恢复路径完整保留；
- 文档记录准确提交号、服务器路径和实验结果。

## 不在本轮解决的问题

- 不通过 prompt 重新设计事实核查语义；
- 不把 private-gold 传给 Agent；
- 不修改任何数据标签；
- 不为单个 case 增加专用分支；
- 不扩大工具预算；
- 不把外部不可用强行变成成功证据；
- 不因某次 smoke 的准确率变化直接调整 Agent 结构。

## 实施记录

### 本地实现（2026-09-05）

当前正式入口已再次确认：

```text
Orchestrator.run()
  -> _run_react_runtime_policy()
  -> src/orchestrator/react_runtime.py
```

版本名仍是 `unified-react-v1`。`_run_unified_react_policy()` /
`src/orchestrator/unified_react.py` 保留为未激活的历史图状态实现，不是本次修改对象。

已完成：

1. `reduce_react_action()` 将 `external_unavailable`、`provider_error`、
   `malformed_tool_result` 和 `success_empty` 都记录为可恢复的 failure ledger 项；
   工具 contract/解析异常不再自动产生 `engineering_error`。
2. `pipeline._apply_react_observation()` 成为当前 ReAct 的事务边界：
   - 先保留上一份 perception 和 runtime state；
   - `perceive_scene` / OCR 只在 envelope 可用时进行专用解析；
   - 解析失败会把模型可见结果替换成结构化 `MalformedToolResult`，原始结果仍由
     `StageRunner` artifact 保留；
   - 状态登记异常会回滚到上一份有效状态，并用同一动作的结构化失败结果重新登记。
3. 感知/OCR 的数值、bbox 和置信度解析改为防御式归一化；不合法 bbox 被丢弃而不是
   令 case 退出。
4. native protocol correction 耗尽时，active pipeline 识别
   `protocol_correction_exhaustion_boundary`，直接进入现有 Judgment，而不是抛出
   `ReAct did not complete one accepted action`。
5. `StageRunner` 不再因为恢复 cache / 普通 cache 中的坏工具 JSON 抛出；
   它会跳过坏缓存并重新执行工具。
6. strict trace audit 新增可恢复失败、外部不可用、坏工具结果和空成功结果的计数。
7. 当前 ReAct 英文 prompt 和中英文运行文档已同步：空结果和坏工具结果是未解决的
   失败观察，不是证据，也不会被描述为必然工程错误。

新增/更新本地验证：

- `test_react_runtime.py`：
  - 坏 JSON 的 `perceive_scene` 被记录为可恢复失败并消耗一次动作；
  - malformed tool contract 可恢复；
  - `success_empty` 不晋升为 Evidence；
  - protocol correction 耗尽进入现有 Judgment。
- 已通过：

```text
pytest -q test_react_runtime.py test_native_interactions.py \
  test_gemini_interactions_contract.py test_tool_contract_repairs.py

113 passed, 6 skipped
```

`test_audit_real_trace.py` 当前在收集阶段因已删除的历史模块
`src.orchestrator.coverage` 缺失而失败；该失败在本次改动前已存在，且不阻断当前
`test_react_runtime.py` 内的 strict audit fixture。真实 smoke 仍会使用
`scripts/audit_real_trace.py` 再验证。

### 待完成

1. 扩展本地回归与 compileall；
2. 提交、push，gpu-13 fast-forward；
3. 从统一测试集抽取此前未用于 smoke 的 10 个新 case，真实运行；
4. 对每条新 trace 检查外部/坏结果是否继续调查、是否进入 Judgment、是否通过 strict audit；
5. 将真实结果回填本计划和实验记录；在启动任何大规模 rollout 前停止。

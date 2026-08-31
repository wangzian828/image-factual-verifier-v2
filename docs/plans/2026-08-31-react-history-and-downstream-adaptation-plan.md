# ReAct 累计会话与下游适配计划

日期：2026-08-31  
范围：`unified-react-v1` 当前主路径  
暂停边界：不启动大规模教师 rollout。

## 目标

让 Agent 按成熟 search-agent 的方式工作：工具结果进入后续模型上下文，并由同一 episode 的
会话历史持续保留；同时避免重复上传原图、重复拼接旧状态和重复生成 SFT 样本。

## 执行项

1. **Runtime**

   - 为完整 ReAct episode 复用一个 `InteractionSession`。
   - 每轮只把上一轮新产生的 `function_result` 作为显式输入。
   - 根请求上传压缩原图；后续轮次使用 provider 历史，不重复上传原图。
   - 候选图、聚焦视觉结果和当前状态文字合并到一个 `user_input`。
   - 保留成熟工具的执行、摘要、下载和重试契约。

2. **Strict audit**

   - 检查统一 ReAct interaction ID 存在。
   - 第一条 interaction 必须无 parent，后续 interaction 必须连接上一条。
   - 如果上一条是成功工具动作，下一请求必须包含匹配的 `function_result.call_id`。
   - 保留工具失败与工程错误的区分。

3. **SFT / reward / report**

   - SFT 只从 `state.all_steps` 的真实时间顺序构造 episode。
   - 每个工具结果只导出一次，不把 provider 历史快照复制为额外样本。
   - 继续保留 provider 实际返回的 thought、工具调用、观察和 reducer delta。
   - reward/scoring 继续按 action 与最终结果计算，不把 interaction parent 当作新 action。
   - report history 和可读视图保留事件流，去除重复的累计 workspace。

4. **验证**

   - 运行交互、ReAct、审计、SFT、reward 相关本地测试。
   - 运行 `compileall` 和 `git diff --check`。
   - 提交并同步服务器 checkout。
   - 用 Gemini 3.7 重新跑原定 10 条 smoke，逐条检查：
     - 工具结果是否进入下一请求；
     - 原图是否只在根请求上传；
     - 候选图是否不再触发多 `user_input`；
     - interaction parent 链是否连续；
     - strict audit 是否通过；
     - SFT episode 是否完整且没有重复。

5. **后续决策**

   - 只有 10 条 smoke 的调查轨迹明显改善，才启动新的 100 条 Agent rollout 和统一 private-gold
     judge。
   - 在此之前不启动大规模教师轨迹生产。

## 当前进度

- [x] 对照 WebWatcher 确认累计历史模式。
- [x] 恢复 ReAct episode 的 `InteractionSession`。
- [x] 恢复每轮 `function_result` 前递。
- [x] 避免链式请求重复上传原图。
- [x] 合并 session handoff 中的多个 `user_input`。
- [x] strict audit 改为检查父链和工具结果前递。
- [x] 增加交互回归测试。
- [ ] 完成下游导出/审计全量门禁。
- [ ] 提交并同步服务器。
- [ ] 完成 10 条真实 smoke 与前后对照。

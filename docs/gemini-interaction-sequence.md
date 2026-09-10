# Gemini 交互顺序

当前 `unified-react-v1` 为每个 episode 创建一个持续到 Judgment 的
`InteractionSession`。

```text
根 ReAct request（固定 objective + 受控原图）
  → thought + 一个 native function_call
  → 本地执行工具并记录原始 result
  → function_result + previous_interaction_id
  → 下一次 ReAct request
  → ...
  → finish / budget / protocol boundary
  → 同一 session 中的 Judgment request
```

每个 ReAct action 只接受一个 native function。第 N 轮完成的 result 在第 N+1 轮
显式提交一次；更早内容由 provider 父链保留，不再手工重复。根原图也不逐轮上传。
工具新增的候选图、裁剪图或聚焦视图与相应 function result 一起进入紧邻的请求边界。

runtime context 只补充 objective、图片可用性和剩余动作预算。它不维护语义 workspace，
不压缩或分类历史观察。`finish_investigation` 只携带完成说明；达到 24 个 accepted
tool actions 或协议修复边界时也会进入最终 Judgment。

## 轨迹、审计和训练

- canonical trace 保存真实 thought、function call、原始 tool result、请求快照和
  provider interaction 父链；
- strict audit 检查每轮单工具约束、父链连续性及上一轮 function result 是否进入
  下一请求；
- Judgment 读取同一累计 history，并只从机械 locator 中引用成功 observation IDs；
- SFT exporter 按 `thought → tool_call → raw tool_response → next thought` 导出一个
  完整 episode，不重复展开 provider 已保存的历史；
- hidden reasoning 不进入训练数据，只有 API 返回的可读 thought 可以成为 reasoning
  supervision。

SSL、验证码、页面不可访问、空结果和 malformed result 都作为原始观察保留。它们不被
静默丢弃或升级为事实结论；只有不可恢复的 case、worker、provider 或持久化故障会使
episode 报错终止。

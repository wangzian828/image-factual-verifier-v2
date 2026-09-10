# 当前架构：`unified-react-v1`

## 主流程

```text
公开 case（原图、case_id、SHA-256）
  → 创建一个 provider InteractionSession
  → ReAct thought + 一个 native tool call
  → 原始 function result 追加到同一会话和 canonical trace
  → 重复，直到 finish、预算耗尽或协议边界停止
  → Judgment 在同一累计会话中读取完整 raw history
  → 输出 real/fake、fact-check report 和引用的 observation IDs
```

没有 graph、Claim/Task ownership、Planning/Replan/Reflection 阶段，也没有把工具结果
归类为 Discovery/Evidence 的 reducer。策略模型直接解释 provider 会话中保留的原始
观察；runtime 不替模型生成语义摘要或证据结论。

## 职责边界

| 部件 | 负责 | 不负责 |
| --- | --- | --- |
| ReAct policy | 解释 raw history、形成 thought、选择下一工具 | 修改机械状态、伪造工具结果或内部 ID |
| runtime adapter | 暴露公开 schema、注入受控图片、校验来源策略 | 改写工具结果语义 |
| 成熟工具 | OCR、感知、搜索、网页提取、图像检查和比较 | 最终二分类 |
| mechanical runtime | 动作计数、预算、停止原因、完成说明 | 证据分级、路线图或事实判断 |
| Judgment | 从完整 raw history 写二分类报告 | 新增调查、来源或观察 |

## 状态和观察

`UnifiedReactState` 只保存 schema 版本、case ID、图片 SHA-256、固定 objective、
动作数、停止原因和完成说明。每次工具调用的参数、原始结果、thought、provider
interaction ID 和传输元数据保存在 canonical steps 中；provider-side history 是
下一轮模型看到先前结果的主要通道。

Judgment context 只增加机械 observation locator（成功状态、工具名、call ID、query
和请求 URL），便于校验引用。它不复制、裁剪或重写观察正文。最终
`verdict_observation_ids` 只能引用成功的原始观察。

## 图片传递

- `direct_multimodal`：Interaction 根请求附加一次受控原图；后续轮次复用同一
  provider history，Judgment 也沿用该 session。
- `separate_vlm`：主 policy 不直接接收图片，视觉工具接收受控原图并返回观察。
- 工具新增的裁剪图或候选参考图只在对应 observation 边界附加。
- canonical text 不保存 base64；媒体通过外部 artifact 和 SHA-256 定位。

## 下游产物

- canonical trace：请求、响应、thought、native call、raw result 和 provider 父链；
- SFT：按时间顺序导出完整 episode，不按动作拆行，也不制造 reducer 摘要；
- strict audit：检查单工具轮次、父链、function result 回传和 Judgment 引用；
- reward/evaluation：从 raw actions、raw observations 和最终报告计算，不恢复旧图状态。

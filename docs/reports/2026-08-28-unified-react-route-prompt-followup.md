# unified-react 多路线与 Prompt 补强复测记录

日期：2026-08-28

## 代码

- commit：`a784092`
- 分支：`codex/gpu13-canary-20260804-plan-relaxation-01`
- GPU13 checkout 已更新到该 commit。
- 本地全量测试：393/393。
- GPU13 全量测试：393/393。

## 本次改动

1. 首次真实调查动作注册同一个 target fact 下 2–3 条不同候选路线；
2. 后续统一 ReAct 可在多个 route/task 之间切换；
3. `stop_route` 只有在当前 route 没有待检查候选和其他可执行 material step 时才允许；
4. 补强 ReAct Prompt 对候选核验、相似来源、参考图比较边界、决定性 Evidence、
   `unresolved/open_gaps` 和继续调查条件的要求；
5. 同步 investigation intent schema、model-visible context、中文 Prompt 备份和测试。

## Gemini 3.7 复测

使用正式训练池 runtime projection 的公开输入，运行目录：

- 并发 10：
  `/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/unified-react-v1-gemini37-route-prompt-smoke-20260828-r2-p10`
- 并发 4：
  `/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/unified-react-v1-gemini37-route-prompt-smoke-20260828-r3-p4`
- 并发 1 可用性验证：
  `/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/unified-react-v1-gemini37-route-prompt-availability-20260828-p1-2`

三批共 22 条请求均未进入 Agent：

- trace：22/22 error；
- Agent tool calls：0；
- Agent model calls：0；
- 主要错误：Gemini Interactions HTTP 500；
- provider message：`gemini-3.7-flash is currently experiencing high demand`；
- 少量请求为读取超时。

因此这三批不能用于评价 Prompt、路线结构或 SFT 质量。

## SFT judge 状态

本次新增失败 trace 没有完成轨迹，按设计不调用 SFT judge，避免把 provider 错误当成
SFT 质量拒绝。上一批历史有效 3.7 trace 的独立 judge 诊断也受到同一 Gemini 3.7
provider 拥塞影响，结果只作为 judge 链路诊断，不作为本次改动的效果结论。

## 当前结论

代码和确定性协议问题已验证通过；Prompt/多路线改动是否提升 `no_decisive_evidence`、
`poor_retrieval_quality` 和 `major_overclaiming`，必须在 Gemini 3.7 恢复后用同一批
输入重新跑有效轨迹，再执行 SFT judge 对比。

## 3.6 诊断与修复

3.6 单请求探测成功，但用当前多路线 wire schema 跑 10 条时，视觉 bootstrap 之后全部返回
HTTP 400 `invalid argument`。最小复现确认原因是 provider-facing 的嵌套 route 数组过复杂：
旧版单 `route` schema 可以正常进入调查动作。

已修复为：provider 只提交主 `route`，可选提交 `alternate_route_focuses`；runtime 在解析后
稳定补齐 2–3 条内部 route/task，并把实际调用工具绑定到主路线。这样保留多路线状态管理，
同时避免把 Gemini 3.6/3.7 的请求契约绑定到复杂嵌套数组。

修复后的本地测试：394/394。此前 3.6 失败批次：
`/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/unified-react-v1-gemini36-route-prompt-smoke-20260828-p10-r2`
（修复前，10/10 provider 400），不能用于质量比较。

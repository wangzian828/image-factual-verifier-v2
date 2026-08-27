# Teacher Rollout 自动流程

当前流程只服务 `unified-react-v1`：

```text
读取 case
  -> 并发启动 rollout
  -> 每条 trace 实时落盘
  -> 工程错误自动回队列重跑
  -> strict trace audit
  -> SFT eligibility / LLM judge
  -> accepted、holdout、action-only、RL candidate 分桶
```

重跑只针对未成功的 case，不覆盖已经成功的 episode。每条 trace 的原始请求、响应、错误和
重试关系都保留；分桶只建立清单，不删除轨迹。

建议生产前先做 10 条并发 10 的真实 Gemini smoke，检查：成功率、工程错误、每条是否有 scene/OCR
action、thought 捕获率、工具调用顺序和最终审计结果。通过后再启动全量 rollout。

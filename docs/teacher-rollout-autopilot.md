# Teacher Rollout 自动流程

当前流程只服务 `unified-react-v1`：

```text
读取 case
  → 并发 rollout
  → 每条 trace 实时落盘
  → 工程异常自动回队列重跑
  → strict trace audit
  → SFT eligibility / LLM judge
  → accepted / holdout / action-only / RL candidate 分桶
```

## 重跑规则

- 工程失败的 case 自动进入待跑队列，不覆盖原始尝试。
- 质量重跑按轮次保存；上一轮未通过 SFT 的 case 才进入下一轮。
- 同一 case 最终选择质量最高且通过准入的轨迹。
- 连续质量重跑仍未通过的 case 标为 hard case，保留全部尝试。
- 轨迹不会因为重跑而混入旧的初始池，也不会删除原始结果。

每轮独立保存：

```text
rollouts/<round>/
sft-eligibility/<round>/
classification/<round>/
```

## SFT 导出边界

通过审计的 reasoning 轨迹进入 `trajectory_sft.jsonl`；没有可读 provider thought
但动作可执行的轨迹进入独立 `action_only.jsonl`；有效的图片观察报告进入
`perception_trajectories.jsonl`。

导出前必须完成：

- trace 结构和工程错误审计；
- SFT judge；
- 图像路径和 SHA-256 校验；
- private 字段泄漏检查；
- 目标 Qwen processor 编码检查。

## 启动前 smoke

完整 rollout 前先用并发 10 跑少量真实 case，检查：

- trace 是否正常结束；
- 每轮 thought 和工具动作是否真实存在；
- 工具结果是否进入下一轮；
- 图片输入没有重复塞进文本历史；
- 没有 CLOSE-WAIT 持续增长；
- SFT 导出和 processor 审计通过。

本次代码收尾完成后停在全量教师 rollout 启动之前，不自动启动 8,490 条全量训练
数据生产。

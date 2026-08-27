# 当前运行发布契约

本项目只消费不可变的 image-only v0.3 release。数据 release 和 Agent 运行时使用两个独立
版本字段，不能混用：

```json
{
  "input_mode": "image_only",
  "decision_policy_version": "reinspect-v2",
  "runtime_contract_version": "ifv-image-only-runtime-v1"
}
```

- `decision_policy_version=reinspect-v2`：数据流水线 release 契约；
- `unified-react-v1`：Agent 的当前编排策略，写入 run manifest、canonical trace 和 state。

公开 case 只允许 `case_id`、`image_path`、`image_sha256`。gold、标签、人工答案、构造信息和
judge 结果不进入 Agent 输入。

发布前检查：

1. 图片路径位于 release 根目录内，且 hash 一致；
2. release manifest 的策略版本为 `reinspect-v2`；
3. source-access policy 与 release manifest 一致；
4. Agent trace 的策略版本为 `unified-react-v1`，并通过 `scripts/audit_real_trace.py`；
5. 测试集、训练集和 RL candidate 使用彼此独立的 manifest。

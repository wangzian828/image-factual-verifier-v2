# RL 与语义奖励

RL 使用 rollout 产生的完整 episode 和独立 reward ledger，不把 evaluator gold 注入 Agent
上下文。奖励评估在轨迹结束后进行，主要检查：

- 是否找到有效调查方向；
- 工具是否服务当前 target fact；
- Evidence 是否被正确使用；
- state 是否随观察合理更新；
- 最终 verdict 与可审计 basis 是否一致。

policy action 仍按 `unified-react-v1` 的单工具调用格式记录。reasoning SFT、action-only 和
RL candidate 可以来自同一原始轨迹，但通过不同 manifest 分开，不在数据文件中混成一个桶。

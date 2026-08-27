# 文档导航

这里记录当前 `unified-react-v1` 的运行方式、数据边界和训练产物。
历史实验记录保留用于复现，不属于当前生产入口。

## 从这里开始

- [Agent 结构与数据流](agent-structure.md)：在线 Agent、工具、状态所有权、教师轨迹与训练数据链路。
- [Agent prompt 与运行时说明（中文）](agent-prompt-and-runtime-guide-zh.md)：每个语义阶段的职责、输入和约束；[英文原文](agent-prompt-and-runtime-guide.md)。
- [当前主 Agent 实际 system prompt（中文备份）](active-agent-system-prompts-zh.md)：统一 ReAct、Reflection、Discrepancy Decision 和 Judgment 的实际提示词。
- [可恢复教师轨迹自动链](teacher-rollout-autopilot.md)：训练集投影、自动工程重试、SFT 审计、质量重跑与产物。
- [当前统一数据集构成](current-dataset-composition.md)：总量、来源、训练/测试切分、prompt 和测试图像来源数量。
- [gpu-13 运维配置档](operations/gpu13.md)：当前服务器工作树、数据集、启动方式和监控。

## 数据与训练

- [运行时 release 契约](runtime-release-contract.md)：运行时公开输入与 evaluator-private gold 的隔离。
- [SFT canonical release 工作流](sft-canonical-release-workflow.md)：从接受的完整教师轨迹到可训练 SFT 数据。
- [SFT 训练方式与数据筛选构造](sft-training-and-data-construction.md)：审计、分桶、reroll、完整轨迹构造和训练启动边界。
- [语义奖励说明](rl-semantic-reward.md)：RL 的诊断性语义奖励边界。

## 历史与排障

- [Gemini 交互与 prompt 顺序](gemini-interaction-sequence.md)
- [2026-08-23 资源生命周期事故](operations/2026-08-23-gemini-rollout-resource-lifecycle.md)
- [2026-08-24 CLOSE-WAIT 后续修复](operations/2026-08-24-close-wait-session-leak-followup.md)
- [2026-08-25 OSS 上传 session 生命周期事故](operations/2026-08-25-oss-upload-session-lifecycle.md)
- [gpu-13 版本登记](operations/gpu13-version-registry.md)

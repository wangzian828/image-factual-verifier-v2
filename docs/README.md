# IFV 文档导航

当前生产入口是 `unified-react-v1`。旧实验文档只用于复现历史结果，不代表当前运行协议。

## Agent 与数据

- [Agent 结构与训练接口](agent-structure.md)
- [SFT 训练与数据构造](sft-training-and-data-construction.md)
- [SFT Canonical Release 工作流](sft-canonical-release-workflow.md)
- [轨迹产物说明](trajectory-artifact-guide.md)
- [Teacher Rollout 自动流程](teacher-rollout-autopilot.md)
- [Qwen / ms-swift 部署说明](qwen-ms-swift-deployment.md)

## Prompt 与运行时

- [英文 Agent prompt 与 runtime 说明](agent-prompt-and-runtime-guide.md)
- [中文 Agent prompt 与 runtime 说明](agent-prompt-and-runtime-guide-zh.md)
- [当前英文 system prompt](active-agent-system-prompts.md)
- [当前中文 system prompt](active-agent-system-prompts-zh.md)
- [运行时 release 契约](runtime-release-contract.md)

## 数据集与实验

- [当前数据集构成](current-dataset-composition.md)
- [Agent 收尾计划](agent-evaluation-closeout-plan.md)
- [Gemini direct QA 实验记录](reports/2026-08-30-gemini-direct-qa-experiment-record.md)
- [新 Agent-100 rollout 与审计](reports/2026-09-02-agent100-newagent-rollout-and-review.md)
- [工具与 SFT 数据包审计](reports/2026-09-02-text-image-search-and-sft-packet-audit.md)

## 服务器运维

- [gpu-13 运维配置](operations/gpu13.md)
- [gpu-07 运维记录](operations/gpu07.md)
- [gpu-13 训练环境说明](../training/docs/gpu13.md)

训练数据、图片、rollout、日志和 checkpoint 放在服务器 `/gsdata`，不写入 Git checkout。
发布前必须同时通过结构审计和目标 Qwen checkpoint 的真实 processor 验证。

# IFV 文档导航

当前生产入口是 `unified-react-v1` raw-history runtime。这里仅保留当前协议、部署、
训练和运维文档；清理前的研究记录、阶段报告和旧架构计划可从 Git 标签
`pre-deep-cleanup-20260910` 恢复。

## Agent 与运行时

- [当前架构](architecture.md)
- [Agent 结构与训练接口](agent-structure.md)
- [英文 prompt 与 runtime 说明](agent-prompt-and-runtime-guide.md)
- [中文 prompt 与 runtime 说明](agent-prompt-and-runtime-guide-zh.md)
- [当前英文 system prompt](active-agent-system-prompts.md)
- [当前中文 system prompt](active-agent-system-prompts-zh.md)
- [Gemini 交互顺序](gemini-interaction-sequence.md)
- [Prompt 与 runtime 边界](prompt-boundaries.md)
- [运行时 release 契约](runtime-release-contract.md)
- [Agent 报告与 private-gold 审计](agent-fact-check-report-and-private-gold-audit.md)

## 数据、轨迹与训练

- [当前数据集构成](current-dataset-composition.md)
- [SFT 训练与数据构造](sft-training-and-data-construction.md)
- [SFT canonical release 工作流](sft-canonical-release-workflow.md)
- [轨迹产物说明](trajectory-artifact-guide.md)
- [Teacher rollout 自动流程](teacher-rollout-autopilot.md)
- [Teacher rollout API 交接](teacher-rollout-api-handoff.md)
- [教师轨迹与 SFT 交接](teacher-rollout-and-sft-handoff.md)
- [Teacher/SFT pipeline 入口](teacher-sft-pipeline-entrypoint.md)
- [Qwen / ms-swift 部署](qwen-ms-swift-deployment.md)
- [SFT 监控与语义看门狗](sft-monitoring-and-semantic-watchdog.md)

## 部署与运维

- [gpu-13 运维配置](operations/gpu13.md)
- [gpu-13 版本注册表](operations/gpu13-version-registry.md)
- [远程 Jupyter 运维](operations/remote-jupyter-operations.md)
- [gpu-13 训练环境](../training/docs/gpu13.md)
- [可迁移部署交接](portable-deployment-handoff.md)

训练数据、图片、rollout、日志和 checkpoint 留在服务器 `/gsdata`，不写入 Git
checkout。发布前同时执行结构审计、真实 provider canary 和目标 Qwen checkpoint
的 processor 验证。

# Search Agent Paradigm Research

本目录来自 2026-07-10 的 Search Agent 与单图事实核查调研，现已归档到项目仓库。

## 主要入口

- `to_human/search_agent_paradigm_report_zh.md`：完整中文报告。
- `to_human/search_agent_paradigm_report_zh.html`：HTML 阅读版。
- `to_human/agent_learning_guide.md`：供其他 Agent 学习和实现的指南。
- `literature/survey.md`：详细文献地图。
- `literature/references.json`：81 条论文机器可读元数据（含摘要）。
- `conversation_log.md`：保留用户关键原话和每轮研究结论入口的对话索引。
- `discussion_2026-07-10.md`：关于主动视觉调查、RL 和数据构造的项目讨论记录。
- `data_construction_review_2026-07-10.md`：对前期 Benchmark 构造方案的严格审计与 Investigation World 改造建议。
- `source_benchmark_plan_v0_2026-06-09.md`：前期 Benchmark 构造原案的只读归档副本。
- `closed_to_open_validation_2026-07-10.md`：闭合训练、冻结真实网页与 live-web 迁移的论文证据和评测协议。
- `data_pipeline_feasibility_2026-07-11.md`：互联网事实锚定合成训练管线与真实图片评测集的难度、成本和分阶段实施评估。
- `two_paper_split_2026-07-11.md`：任务/真实 Benchmark 与训练策略/Agent 两篇论文的贡献边界、数据隔离和依赖式路线。
- `image_only_task_definition_2026-07-11.md`：纯图片、图片加 claim 与 image-centric open-world investigation 三种任务定义的重合分析和 pilot 决策方案。

## 状态说明

这些文件是研究建议和未来设计候选。项目当前实际运行时契约仍以仓库根目录的 `AGENTS.md` 和 `docs/architecture.md` 为准。

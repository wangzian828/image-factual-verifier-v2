# 图像事实核查对比实验结果

## 实验设置

- 评测集：筛选后的统一测试集
- 样本数：1,527
- 评测日期：2026-09-09 至 2026-09-14
- 二分类任务：判断图像及其事件陈述为 `real` 或 `fake`
- 证据质量评估模型：Gemini 3.7 Flash，`thinking_level=low`
- Agent 结果的最终公平审核口径：v3 全材料输入，`max_output_tokens=32768`

## 主要结果

**表 1　不同方法在图像事实核查测试集上的性能比较**

| 方法 | 推理范式 | BAcc ↑ | Real Recall ↑ | Fake Recall ↑ | SESR ↑ |
|:---|:---:|---:|---:|---:|---:|
| **Gemini 3.7 Agent** | Agentic | **81.57** | 84.35 | 78.78 | 42.89 |
| Qwen3.5-9B Agent（第二轮 SFT：4,872 条） | Agentic | 79.95 | 83.02 | 76.87 | 36.54 |
| Qwen3.5-9B Agent（首轮 SFT：2,578 条） | Agentic | 79.87 | 79.58 | 80.17 | 34.77 |
| **GPT-5.5 Agent** | Agentic | 78.13 | 83.82 | 72.43 | **50.49** |
| GPT-5.5 | Direct QA | 75.87 | 79.58 | 72.17 | 3.67 |
| Gemini 3.7 Flash | Direct QA | 75.64 | 71.62 | 79.65 | 9.10 |
| Claude Opus 4.8 | Direct QA | 72.18 | 69.76 | 74.61 | 6.75 |
| Qwen3.5-397B-A17B Agent | Agentic | 71.72 | 50.66 | 92.78 | 43.29 |
| Kimi K2.6 | Direct QA | 70.24 | 81.17 | 59.30 | 8.91 |
| Doubao Seed 2.0 Pro | Direct QA | 70.16 | 77.45 | 62.87 | 8.45 |
| Qwen3.5-397B-A17B | Direct QA | 68.68 | 56.23 | 81.13 | 6.68 |
| MiniMax-M3 | Direct QA | 66.66 | 84.62 | 48.70 | 3.41 |
| GPT-5.2 | Direct QA | 65.84 | 85.94 | 45.74 | 2.95 |
| Qwen3.5-9B Agent（训练前） | Agentic | 65.67 | 70.03 | 61.30 | 3.14 |
| Gemini 3.1 Pro | Direct QA | 62.58 | 28.65 | 96.52 | 5.57 |
| GLM-4.6V | Direct QA | 59.96 | 95.49 | 24.43 | 3.01 |

> 注：表中结果均为百分比（%），↑ 表示数值越高越好。严格证据充分率（Strict Evidence Sufficiency Rate, SESR）是指回答被独立评估模型判定为证据充分、来源可靠，且推理能够支撑最终结论的样本比例。

## 指标定义

平衡准确率（Balanced Accuracy, BAcc）定义为 Real Recall 与 Fake Recall 的算术平均值。两类召回率分别以测试集中全部 377 条 real 样本和 1,150 条 fake 样本为分母。模型请求失败、输出无法解析、缺少 `verdict` 字段，或输出不是有效 `real/fake` 标签时，均作为所属真实类别的未召回样本处理。

严格证据充分率使用同一批 1,527 条样本作为分母，对应 LLM Judge 质量评估中的 `Strong` 类别。该指标强调结论之外的证据覆盖、来源可靠性及推理充分性，不等同于二分类准确率。

GPT-5.5 与 MiniMax-M3 的原始结果各覆盖 1,526 条样本，未覆盖样本仍按错误计入统一分母。GPT-5.5 另有 2 条源答案无有效终态，无法进入证据质量评估；这些样本同样不从分母中剔除。

Qwen3.5-9B Agent（训练前）的二分类结果按冻结 case ID 复用已有的 1,526 条推理。缺失的 1 条 real 按未召回计入 1,527 分母；real 正确 264/377，fake 正确 705/1,150，其 Accuracy 为 63.46%。表中的 SESR 已更新为最终 v3 全材料审核结果 48/1,527（3.14%），替代旧版 44/1,527（2.88%）。二分类记录保存在服务器 `qwen35-base-agent-formal1527-pretrain-from-old1682-20260912/summary.json`；v3 审核保存在 `/volume/ybo/wza/runs/eval/gemini37-qwen35base-agent-batch-judge-v3-low32k-20260913`。旧版 Gemini 3.1 Pro/high 审核不进入本表。

Qwen3.5-9B Agent（首轮 SFT）的冻结结果覆盖 1,526 条可运行样本。原始汇总按 1,526 条可用样本给出的 BAcc 为 79.98%；本表遵守统一定义，把缺失的 1 条 real 计为未召回，因此记录 BAcc 79.87%、Real Recall 79.58%、Fake Recall 80.17%。冻结结果位于服务器 `/volume/ybo/wza/evaluation/qwen35-sft-initial2578-agent-final1526-20260913`。

Qwen3.5-9B Agent（第二轮 SFT）的冻结结果同样覆盖 1,526 条可运行样本。可用样本口径为 real 313/376、fake 884/1,150，BAcc 80.06%；本表把缺失的 1 条 real 计为未召回，因此记录 Real Recall 83.02%、Fake Recall 76.87%、BAcc 79.95%。冻结结果位于服务器 `/volume/ybo/wza/evaluation/qwen35-sft1028-agent-final1526-20260914`；统一 v3 审核得到 `Strong` 558 条，即 SESR 558/1,527（36.54%）。

Gemini 3.1 Pro 的证据质量已使用统一的 Gemini 3.7 Flash、`thinking_level=low` 配置重新审核。最终有 85 条样本被判定为 `Strong`，SESR 为 5.57%；旧的 23.71% 来自 Gemini 3.1 Pro 自审且使用 `thinking_level=high` 的非统一口径，已废弃。

## Agent v3 Batch judge 完整性记录

以下六组均使用 Gemini 3.7 Flash、`thinking_level=low`、`max_output_tokens=32768` 和 v3 全材料输入。每组均有 1,526 条唯一、状态为 `completed` 的审核记录；不可运行的 1 条样本不从 1,527 报告分母中删除。

| Agent | 完成数 | 严格证据充分数 | SESR（1,527 分母） | 服务器记录 |
|:---|---:|---:|---:|:---|
| GPT-5.5 Agent | 1,526/1,526 | 771 | 50.49% | `gpt55-qwen397-agent-batch-judge-v3-low32k-20260913/gpt55-agent` |
| Qwen3.5-397B-A17B Agent | 1,526/1,526 | 661 | 43.29% | `gpt55-qwen397-agent-batch-judge-v3-low32k-20260913/qwen35-397b-a17b-agent` |
| Gemini 3.7 Agent | 1,526/1,526 | 655 | 42.89% | `gemini37-qwen35base-agent-batch-judge-v3-low32k-20260913/gemini37-agent` |
| Qwen3.5-9B Agent（第二轮 SFT） | 1,526/1,526 | 558 | 36.54% | `qwen35-sft1028-agent-batch-judge-v3-low32k-20260914/qwen35-9b-sft1028-agent` |
| Qwen3.5-9B Agent（首轮 SFT） | 1,526/1,526 | 531 | 34.77% | `qwen35-sft-agent-final1526-batch-judge-v3-low32k-20260913/qwen35-9b-sft-agent` |
| Qwen3.5-9B Agent（训练前） | 1,526/1,526 | 48 | 3.14% | `gemini37-qwen35base-agent-batch-judge-v3-low32k-20260913/qwen35-base-agent` |

截至 2026-09-14，以上 Batch judge 均已完成且不应重复提交。当前仅有 Gemini 3.1 Pro Preview Agent 尚未进入最终 judge：其首轮全量运行只成功 389/1,526 条，另外 1,137 条全部因当时服务器 `No space left on device` 结束；这不是模型质量结果，也未提交最终 judge。必须仅重跑这 1,137 条并补齐到 1,526 个唯一成功 case 后，才能提交同口径 v3 Batch judge。原始失败记录位于服务器 `/volume/ybo/wza/runs/eval/gemini31pro-agent-formal1527-20260913/full-r2-c4`。

## 初步结论

- Gemini 3.7 Agent 获得当前最高 BAcc，为 81.57%，且 real/fake 两类召回较为均衡。
- GPT-5.5 Agent 获得当前最高 SESR，为 50.49%；Gemini 3.7 Agent 与 Qwen3.5-397B-A17B Agent 分别为 42.89% 和 43.29%。
- 首轮 SFT 的 Qwen3.5-9B Agent 在严格 1,527 分母下达到 79.87% BAcc 和 34.77% SESR，显著高于训练前的 65.67% 和 3.14%。
- 第二轮 SFT 的 Qwen3.5-9B Agent 在同口径下达到 79.95% BAcc 和 36.54% SESR；相较首轮 SFT，SESR 提升 1.77 个百分点，Real Recall 上升 3.44 个百分点，但 Fake Recall 下降 3.30 个百分点。
- GPT-5.5 与 Gemini 3.7 Flash 的 BAcc 分别为 75.87% 和 75.64%，是表现最好的两组 Direct QA 基线。
- Gemini 3.1 Pro 虽然原始 Accuracy 较高，但 real 召回仅为 28.65%，BAcc 因此降至 62.58%。
- Agent 的优势体现在证据搜集与证据支持的推理质量，但训练前 Qwen3.5-9B Agent 的 SESR 仅 3.14%，说明该优势并非仅由 Agent 流程保证。

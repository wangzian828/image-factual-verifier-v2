# 图像事实核查对比实验结果

## 实验设置

- 评测集：筛选后的统一测试集
- 样本数：1,527
- 评测日期：2026-09-09 至 2026-09-10
- 二分类任务：判断图像及其事件陈述为 `real` 或 `fake`
- 证据质量评估模型：Gemini 3.7 Flash，`thinking_level=low`

## 主要结果

**表 1　不同方法在图像事实核查测试集上的性能比较**

| 方法 | 推理范式 | BAcc ↑ | Real Recall ↑ | Fake Recall ↑ | SESR ↑ |
|:---|:---:|---:|---:|---:|---:|
| **Gemini 3.7 Agent** | Agentic | **81.57** | 84.35 | 78.78 | **50.69** |
| GPT-5.5 | Direct QA | 75.87 | 79.58 | 72.17 | 3.67 |
| Gemini 3.7 Flash | Direct QA | 75.64 | 71.62 | 79.65 | 9.10 |
| Claude Opus 4.8 | Direct QA | 72.18 | 69.76 | 74.61 | 6.75 |
| Kimi K2.6 | Direct QA | 70.24 | 81.17 | 59.30 | 8.91 |
| Doubao Seed 2.0 Pro | Direct QA | 70.16 | 77.45 | 62.87 | 8.45 |
| Qwen3.5-397B-A17B | Direct QA | 68.68 | 56.23 | 81.13 | 6.68 |
| MiniMax-M3 | Direct QA | 66.66 | 84.62 | 48.70 | 3.41 |
| GPT-5.2 | Direct QA | 65.84 | 85.94 | 45.74 | 2.95 |
| Gemini 3.1 Pro | Direct QA | 62.58 | 28.65 | 96.52 | 5.57 |
| GLM-4.6V | Direct QA | 59.96 | 95.49 | 24.43 | 3.01 |

> 注：表中结果均为百分比（%），↑ 表示数值越高越好。严格证据充分率（Strict Evidence Sufficiency Rate, SESR）是指回答被独立评估模型判定为证据充分、来源可靠，且推理能够支撑最终结论的样本比例。

## 指标定义

平衡准确率（Balanced Accuracy, BAcc）定义为 Real Recall 与 Fake Recall 的算术平均值。两类召回率分别以测试集中全部 377 条 real 样本和 1,150 条 fake 样本为分母。模型请求失败、输出无法解析、缺少 `verdict` 字段，或输出不是有效 `real/fake` 标签时，均作为所属真实类别的未召回样本处理。

严格证据充分率使用同一批 1,527 条样本作为分母，对应 LLM Judge 质量评估中的 `Strong` 类别。该指标强调结论之外的证据覆盖、来源可靠性及推理充分性，不等同于二分类准确率。

GPT-5.5 与 MiniMax-M3 的原始结果各覆盖 1,526 条样本，未覆盖样本仍按错误计入统一分母。GPT-5.5 另有 2 条源答案无有效终态，无法进入证据质量评估；这些样本同样不从分母中剔除。

Gemini 3.1 Pro 的证据质量已使用统一的 Gemini 3.7 Flash、`thinking_level=low` 配置重新审核。最终有 85 条样本被判定为 `Strong`，SESR 为 5.57%；旧的 23.71% 来自 Gemini 3.1 Pro 自审且使用 `thinking_level=high` 的非统一口径，已废弃。

## 初步结论

- Gemini 3.7 Agent 获得最高 BAcc，为 81.57%，且 real/fake 两类召回较为均衡。
- GPT-5.5 与 Gemini 3.7 Flash 的 BAcc 分别为 75.87% 和 75.64%，是表现最好的两组 Direct QA 基线。
- Gemini 3.1 Pro 虽然原始 Accuracy 较高，但 real 召回仅为 28.65%，BAcc 因此降至 62.58%。
- Gemini 3.7 Agent 的严格证据充分率达到 50.69%，明显高于所有 Direct QA 基线。
- Agent 的主要优势体现在证据搜集与证据支持的推理质量，而不仅是最终二分类标签。

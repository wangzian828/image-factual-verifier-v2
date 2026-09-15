# 图像事实核查对比实验结果

## 实验设置

- 评测集：筛选后的统一测试集
- 样本数：1,527
- 评测日期：2026-09-09 至 2026-09-16
- 二分类任务：判断图像及其事件陈述为 `real` 或 `fake`
- 证据质量评估模型：Gemini 3.7 Flash，`thinking_level=low`
- Agent 结果的最终公平审核口径：v3 全材料输入，`max_output_tokens=32768`
- 2026-09-15 晚，epoch3 Qwen 与 Gemini3.1Pro 的待审部分因 Batch 限流转普通 API；已受理的 149 条 Batch 保留，不重复。模型和审核内容不变，完成后统一合并，详见 [切换记录](judge-realtime-switch-20260915.md)。

## 主要结果

### Agent 模型

**表 1　Agent 模型性能（按 BAcc 降序）**

| 排名 | 模型 | BAcc ↑ | Real Recall ↑ | Fake Recall ↑ | SESR ↑ |
|---:|:---|---:|---:|---:|---:|
| 1 | **Gemini 3.7 Agent** | **81.57** | 84.35 | 78.78 | 42.89 |
| 2 | Qwen3.5-9B Agent（SFT-4872，从 Base 独立训练） | 79.95 | 83.02 | 76.87 | 36.54 |
| 3 | Qwen3.5-9B Agent（SFT-2578，从 Base 独立训练） | 79.87 | 79.58 | 80.17 | 34.77 |
| 4 | Qwen3.5-9B Agent（SFT-4872，另一次 Base 初始化训练，epoch 3） | 79.08 | 78.25 | 79.91 | 待 judge |
| 5 | **GPT-5.5 Agent** | 78.13 | 83.82 | 72.43 | **50.49** |
| 6 | Qwen3.5-9B Agent（SFT-4872，同一次三 epoch 训练，epoch 2；2 条失败） | 77.80 | 75.33 | 80.26 | 待 judge |
| 7 | Qwen3.5-397B-A17B Agent | 71.72 | 50.66 | 92.78 | 43.29 |
| 8 | Qwen3.5-9B Agent（训练前） | 65.67 | 70.03 | 61.30 | 3.14 |
| 9 | Gemini 3.1 Pro Agent | 62.42 | 29.71 | 95.13 | 待 judge |

### Direct QA 模型

**表 2　Direct QA 模型性能（按 BAcc 降序）**

| 排名 | 模型 | BAcc ↑ | Real Recall ↑ | Fake Recall ↑ | SESR ↑ |
|---:|:---|---:|---:|---:|---:|
| 1 | **GPT-5.5** | **75.87** | 79.58 | 72.17 | 3.67 |
| 2 | Gemini 3.7 Flash | 75.64 | 71.62 | 79.65 | **9.10** |
| 3 | Claude Opus 4.8 | 72.18 | 69.76 | 74.61 | 6.75 |
| 4 | Kimi K2.6 | 70.24 | 81.17 | 59.30 | 8.91 |
| 5 | Doubao Seed 2.0 Pro | 70.16 | 77.45 | 62.87 | 8.45 |
| 6 | Qwen3.5-397B-A17B | 68.68 | 56.23 | 81.13 | 6.68 |
| 7 | MiniMax-M3 | 66.66 | 84.62 | 48.70 | 3.41 |
| 8 | GPT-5.2 | 65.84 | 85.94 | 45.74 | 2.95 |
| 9 | Qwen3.5-9B（训练前） | 65.70 | 79.05 | 52.35 | 1.70 |
| 10 | Gemini 3.1 Pro | 62.58 | 28.65 | 96.52 | 5.57 |
| 11 | GLM-4.6V | 59.96 | 95.49 | 24.43 | 3.01 |

### Qwen3.5-9B 独立 SFT 对照

**表 3　从 Base 模型分别启动的独立 SFT 实验**

| 实验 | 初始化模型 | 训练数据 | BAcc | 较训练前 | SESR | 较训练前 |
|:---|:---|---:|---:|---:|---:|---:|
| 训练前基线 | Qwen3.5-9B Base | 0 | 65.67 | — | 3.14 | — |
| SFT-2578 | Qwen3.5-9B Base | 2,578 条 | 79.87 | +14.20 | 34.77 | +31.63 |
| SFT-4872 | Qwen3.5-9B Base | 4,872 条 | 79.95 | +14.28 | 36.54 | +33.40 |
| SFT-4872，另一次三 epoch 训练的 epoch 2（2 条失败） | Qwen3.5-9B Base | 4,872 条 | 77.80 | +12.13 | 待 judge | — |
| SFT-4872，另一次三 epoch 训练的 epoch 3 | Qwen3.5-9B Base | 4,872 条 | 79.08 | +13.41 | 待 judge | — |

> 注：表中结果均为百分比（%），↑ 表示数值越高越好。严格证据充分率（Strict Evidence Sufficiency Rate, SESR）是指回答被独立评估模型判定为证据充分、来源可靠，且推理能够支撑最终结论的样本比例。

## 指标定义

平衡准确率（Balanced Accuracy, BAcc）定义为 Real Recall 与 Fake Recall 的算术平均值。两类召回率分别以测试集中全部 377 条 real 样本和 1,150 条 fake 样本为分母。模型请求失败、输出无法解析、缺少 `verdict` 字段，或输出不是有效 `real/fake` 标签时，均作为所属真实类别的未召回样本处理。

严格证据充分率使用同一批 1,527 条样本作为分母，对应 LLM Judge 质量评估中的 `Strong` 类别。该指标强调结论之外的证据覆盖、来源可靠性及推理充分性，不等同于二分类准确率。

GPT-5.5 与 MiniMax-M3 的原始结果各覆盖 1,526 条样本，未覆盖样本仍按错误计入统一分母。GPT-5.5 另有 2 条源答案无有效终态，无法进入证据质量评估；这些样本同样不从分母中剔除。

Qwen3.5-9B（训练前）Direct QA 已完成 1,526 条唯一推理及对应的 1,526 条审核。已逐条核对冻结测试集、私有 gold、源预测与审核记录，real 正确 298/377、fake 正确 602/1,150，缺失的 1 条 real 仍计为未召回，因此 BAcc 为 65.70%、Real Recall 为 79.05%、Fake Recall 为 52.35%。沿用已确认的 Gemini 3.7 Flash、`thinking_level=low`、`max_output_tokens=8192` 直接 QA 审核，严格证据充分数为 26/1,527，SESR 为 1.70%；该结果不是 Agent 基线，也未因补表重新调用 judge。源结果位于服务器 `/volume/ybo/wza/runs/eval/qwen35-base-directqa-formal1527-20260912/results.jsonl`，审核记录位于 `/volume/ybo/wza/runs/eval/qwen35-base-directqa-formal1527-20260912-judged/audit-results.jsonl`。

Qwen3.5-9B Agent（训练前）的二分类结果按冻结 case ID 复用已有的 1,526 条推理。缺失的 1 条 real 按未召回计入 1,527 分母；real 正确 264/377，fake 正确 705/1,150，其 Accuracy 为 63.46%。表中的 SESR 已更新为最终 v3 全材料审核结果 48/1,527（3.14%），替代旧版 44/1,527（2.88%）。二分类记录保存在服务器 `qwen35-base-agent-formal1527-pretrain-from-old1682-20260912/summary.json`；v3 审核保存在 `/volume/ybo/wza/runs/eval/gemini37-qwen35base-agent-batch-judge-v3-low32k-20260913`。旧版 Gemini 3.1 Pro/high 审核不进入本表。

Qwen3.5-9B Agent（SFT-2578）的冻结结果覆盖 1,526 条可运行样本。该模型从 Qwen3.5-9B Base 独立启动训练，并非 SFT-4872 的前置阶段。原始汇总按 1,526 条可用样本给出的 BAcc 为 79.98%；本表遵守统一定义，把缺失的 1 条 real 计为未召回，因此记录 BAcc 79.87%、Real Recall 79.58%、Fake Recall 80.17%。冻结结果位于服务器 `/volume/ybo/wza/evaluation/qwen35-sft-initial2578-agent-final1526-20260913`。

Qwen3.5-9B Agent（SFT-4872）的冻结结果同样覆盖 1,526 条可运行样本。该模型也从同一个 Qwen3.5-9B Base 独立启动训练，不是从 SFT-2578 checkpoint 续训。可用样本口径为 real 313/376、fake 884/1,150，BAcc 80.06%；本表把缺失的 1 条 real 计为未召回，因此记录 Real Recall 83.02%、Fake Recall 76.87%、BAcc 79.95%。冻结结果位于服务器 `/volume/ybo/wza/evaluation/qwen35-sft1028-agent-final1526-20260914`；统一 v3 审核得到 `Strong` 558 条，即 SESR 558/1,527（36.54%）。

Gemini 3.1 Pro 的证据质量已使用统一的 Gemini 3.7 Flash、`thinking_level=low` 配置重新审核。最终有 85 条样本被判定为 `Strong`，SESR 为 5.57%；旧的 23.71% 来自 Gemini 3.1 Pro 自审且使用 `thinking_level=high` 的非统一口径，已废弃。

## Agent v3 Batch judge 完整性记录

以下六组均使用 Gemini 3.7 Flash、`thinking_level=low`、`max_output_tokens=32768` 和 v3 全材料输入。每组均有 1,526 条唯一、状态为 `completed` 的审核记录；不可运行的 1 条样本不从 1,527 报告分母中删除。

| Agent | 完成数 | 严格证据充分数 | SESR（1,527 分母） | 服务器记录 |
|:---|---:|---:|---:|:---|
| GPT-5.5 Agent | 1,526/1,526 | 771 | 50.49% | `gpt55-qwen397-agent-batch-judge-v3-low32k-20260913/gpt55-agent` |
| Qwen3.5-397B-A17B Agent | 1,526/1,526 | 661 | 43.29% | `gpt55-qwen397-agent-batch-judge-v3-low32k-20260913/qwen35-397b-a17b-agent` |
| Gemini 3.7 Agent | 1,526/1,526 | 655 | 42.89% | `gemini37-qwen35base-agent-batch-judge-v3-low32k-20260913/gemini37-agent` |
| Qwen3.5-9B Agent（SFT-4872） | 1,526/1,526 | 558 | 36.54% | `qwen35-sft1028-agent-batch-judge-v3-low32k-20260914/qwen35-9b-sft1028-agent` |
| Qwen3.5-9B Agent（SFT-2578） | 1,526/1,526 | 531 | 34.77% | `qwen35-sft-agent-final1526-batch-judge-v3-low32k-20260913/qwen35-9b-sft-agent` |
| Qwen3.5-9B Agent（训练前） | 1,526/1,526 | 48 | 3.14% | `gemini37-qwen35base-agent-batch-judge-v3-low32k-20260913/qwen35-base-agent` |

以上六组 Batch judge 均已完成，不应重复提交。2026-09-15，Gemini 3.1 Pro Agent 的工程失败补跑已完成，冻结 1,526 个唯一成功 case：real 正确 112/377、fake 正确 1094/1150，BAcc 62.42%。旧磁盘满失败及所有补跑仍保留，合并索引位于服务器 `gemini31pro-agent-formal1527-20260913/resume-control-20260914/merged-success-results.jsonl`。

另一次从 Base 初始化的三 epoch SFT 已完成 epoch-3/step-3084 全量 Agent 推理，冻结 1,526 个成功 case：real 正确 295/377、fake 正确 919/1150，BAcc 79.08%。与前述 SFT-4872 单 epoch 实验不是续跑关系。epoch-3 最终选择索引为 `qwen35-sft3084-3epoch-agent-formal1527-20260915/selected-traces.json`。

同一次三 epoch 训练的 epoch-2/step-2056 已于 2026-09-16 用完原协议下的有限补跑：1,524 个唯一成功 case、2 条失败、另 1 条缺图不可运行，不称为 1,526 条全部成功。real 正确 284/377、fake 正确 923/1150，对应 BAcc 77.80%、Real Recall 75.33%、Fake Recall 80.26%；失败和缺图均未从分母剔除。两条最终失败均为 `finish_reason=length` 且无可用答案，此前还发生过读超时。冻结索引及汇总位于服务器 `/volume/ybo/wza/evaluation/qwen35-sft2056-epoch2-agent-budgeted1524-20260916`，索引 SHA256 为 `104bd65fec08f13966156beea4e012363d2ec02ddc67ec7a8f50066efcc210b0`。该轮 `thinking_token_budget=8192` 只是请求值，原 vLLM 未执行此预算；后续隔离 PSD 的预算修复不追溯改变本轮协议或结果。1,524 条有效轨迹的统一 v3 审核候选已准备在 `/volume/ybo/wza/evaluation/sft2056-v3-judge-20260916`，包含 20,275 个原始 observation，gzip 约 21.89 MiB、SHA256 `0bffefb2f4f9b6d9ecb8fcce4d8f0b64eb2cfd3cdb3fa2864cec032dae1a577c`；尚未提交，SESR 仍待审核，不能复用 epoch3 结果。

epoch3 与 Gemini 3.1 Pro 两组同口径 v3 judge 继续采用已固定的混合传输：已受理 28 个 Batch 共 149 条，其余 2,903 条分配给普通 API，不能重新启动旧 Batch 提交阶段。2026-09-16 03:47 巡检普通调用仍在推进，Batch 28 个仍为 RUNNING，尚无最终 SESR；有限重试用尽和大图配额受阻的案例保留，未算作完成。epoch2 不属于这两组既有任务。详情见 [普通 judge 切换记录](judge-realtime-switch-20260915.md)。

## 初步结论

- Gemini 3.7 Agent 获得当前最高 BAcc，为 81.57%，且 real/fake 两类召回较为均衡。
- GPT-5.5 Agent 获得当前最高 SESR，为 50.49%；Gemini 3.7 Agent 与 Qwen3.5-397B-A17B Agent 分别为 42.89% 和 43.29%。
- 从 Base 独立训练的 SFT-2578 在严格 1,527 分母下达到 79.87% BAcc 和 34.77% SESR，显著高于训练前的 65.67% 和 3.14%。
- 同样从 Base 独立训练的 SFT-4872 达到 79.95% BAcc 和 36.54% SESR；与 SFT-2578 横向比较，SESR 高 1.77 个百分点，Real Recall 高 3.44 个百分点，但 Fake Recall 低 3.30 个百分点。两者不是前后续训关系。
- GPT-5.5 与 Gemini 3.7 Flash 的 BAcc 分别为 75.87% 和 75.64%，是表现最好的两组 Direct QA 基线。
- Gemini 3.1 Pro 虽然原始 Accuracy 较高，但 real 召回仅为 28.65%，BAcc 因此降至 62.58%。
- Agent 的优势体现在证据搜集与证据支持的推理质量，但训练前 Qwen3.5-9B Agent 的 SESR 仅 3.14%，说明该优势并非仅由 Agent 流程保证。

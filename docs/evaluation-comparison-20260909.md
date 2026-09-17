# 图像事实核查对比实验结果

> **2026-09-16 缓存身份审计告警：** 三 epoch SFT 的 epoch2 有1,246/1,524条、epoch3有1,205/1,526条选中轨迹命中了不同图片文件 SHA 的共享感知/OCR结果。这是候选筛查数量；不同文件 SHA 本身不能排除同图重编码，不等于逐条确认错配或最终标签全部错误。已另外人工确认真实不同场景共用错误感知结果的反例。下表两轮数值保留为原流程冻结记录，**暂不作为排除工程缺陷后的最终模型对比**。其余Agent尚未完成同类审计，不能据此宣称不受影响；Direct QA不经过此工具缓存。详见[缓存事件及处置](perception-cache-incident-20260916.md)。未擅自重跑历史评测或改动原预测。

## 实验设置

- 2026-09-17：主表固定为 BAcc、Macro-Precision、Macro-F1、Supported Recall、Refuted Recall、SESR 六项。20 组分类指标均已从整数计数恢复；原 BAcc、两类 Recall 和 SESR 数值未变，未重新推理或调用 judge。详见下方“恢复依据与复算”。
- 2026-09-16 09:35：epoch3与Gemini3.1Pro两组judge已收尾，每组1,525条有效审核、1条按用户要求计失败的输入；另1条缺图仍保留在1,527分母中。SESR分别为682/1,527（44.66%）及499/1,527（32.68%）。没有改变原预测或BAcc。
- 按用户要求取消26个停滞Batch，保留62条已成功结果，仅把87条明确取消的请求转普通接口；另24条503尾项给予两次追加预算。补跑111条全部成功，没有重跑有效judge，也没有重置旧attempt。详见[尾项处理](judge-realtime-switch-20260915.md)。
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

| 排名 | 模型 | BAcc ↑ | Macro-Precision ↑ | Macro-F1 ↑ | Supported Recall ↑ | Refuted Recall ↑ | SESR ↑ |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 1 | Gemini 3.7 Agent | 81.57 | 75.23 | 76.70 | 84.35 | 78.78 | 42.89 |
| 2 | Qwen3.5-9B Agent（SFT-4872，从 Base 独立训练） | 79.95 | 73.70 | 74.90 | 83.02 | 76.87 | 36.54 |
| 3 | Qwen3.5-9B Agent（SFT-2578，从 Base 独立训练） | 79.87 | 74.60 | 76.07 | 79.58 | 80.17 | 34.77 |
| 4 | Qwen3.5-9B Agent（SFT-4872，另一次 Base 初始化训练，epoch 3；缓存缺陷记录） | 79.08 | 73.99 | 75.41 | 78.25 | 79.91 | 44.66 |
| 5 | GPT-5.5 Agent | 78.13 | 71.60 | 72.06 | 83.82 | 72.43 | 50.49 |
| 6 | Qwen3.5-9B Agent（SFT-4872，同一次三 epoch 训练，epoch 2；2 条失败、缓存缺陷记录） | 77.80 | 73.36 | 74.67 | 75.33 | 80.26 | 暂停新提交 |
| 7 | Qwen3.5-397B-A17B Agent | 71.72 | 77.47 | 73.76 | 50.66 | 92.78 | 43.29 |
| 8 | Qwen3.5-9B Agent（训练前） | 65.67 | 61.76 | 60.15 | 70.03 | 61.30 | 3.14 |
| 9 | Gemini 3.1 Pro Agent | 62.42 | 73.61 | 64.17 | 29.71 | 95.13 | 32.68 |

### Direct QA 模型

**表 2　Direct QA 模型性能（按 BAcc 降序）**

| 排名 | 模型 | BAcc ↑ | Macro-Precision ↑ | Macro-F1 ↑ | Supported Recall ↑ | Refuted Recall ↑ | SESR ↑ |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 1 | GPT-5.5 | 75.87 | 70.09 | 70.51 | 79.58 | 72.17 | 3.67 |
| 2 | Gemini 3.7 Flash | 75.64 | 71.56 | 72.80 | 71.62 | 79.65 | 9.10 |
| 3 | Claude Opus 4.8 | 72.18 | 67.87 | 68.67 | 69.76 | 74.61 | 6.75 |
| 4 | Kimi K2.6 | 70.24 | 65.11 | 62.44 | 81.17 | 59.30 | 8.91 |
| 5 | Doubao Seed 2.0 Pro | 70.16 | 65.38 | 63.77 | 77.45 | 62.87 | 8.45 |
| 6 | Qwen3.5-397B-A17B | 68.68 | 67.23 | 67.82 | 56.23 | 81.13 | 6.68 |
| 7 | MiniMax-M3 | 66.66 | 62.99 | 56.56 | 84.62 | 48.70 | 3.41 |
| 8 | GPT-5.2 | 65.84 | 63.47 | 55.41 | 85.94 | 45.74 | 2.95 |
| 9 | Qwen3.5-9B（训练前） | 65.70 | 61.88 | 57.26 | 79.05 | 52.35 | 1.70 |
| 10 | Gemini 3.1 Pro | 62.58 | 77.23 | 64.54 | 28.65 | 96.52 | 5.57 |
| 11 | GLM-4.6V | 59.96 | 61.98 | 41.86 | 95.49 | 24.43 | 3.01 |

### Qwen3.5-9B 独立 SFT 对照

**表 3　从 Base 模型分别启动的独立 SFT 实验**

| 实验 | 初始化模型 | 训练数据 | BAcc | 较训练前 | SESR | 较训练前 |
|:---|:---|---:|---:|---:|---:|---:|
| 训练前基线 | Qwen3.5-9B Base | 0 | 65.67 | — | 3.14 | — |
| SFT-2578 | Qwen3.5-9B Base | 2,578 条 | 79.87 | +14.20 | 34.77 | +31.63 |
| SFT-4872 | Qwen3.5-9B Base | 4,872 条 | 79.95 | +14.28 | 36.54 | +33.40 |
| SFT-4872，另一次三 epoch 训练的 epoch 2（缓存缺陷记录，2 条失败） | Qwen3.5-9B Base | 4,872 条 | 77.80 | +12.13 | 暂停新提交 | — |
| SFT-4872，另一次三 epoch 训练的 epoch 3（缓存缺陷记录） | Qwen3.5-9B Base | 4,872 条 | 79.08 | +13.41 | 44.66 | +41.52 |

> 注：表中结果均为百分比（%），↑ 表示数值越高越好。严格证据充分率（Strict Evidence Sufficiency Rate, SESR）是指回答被独立评估模型判定为证据充分、来源可靠，且推理能够支撑最终结论的样本比例。

## 指标定义

Supported 对应原 `real`，Refuted 对应原 `fake`，只是报告名称变更，不重新标注数据。平衡准确率（Balanced Accuracy, BAcc）定义为 Supported Recall 与 Refuted Recall 的算术平均值。两类召回率分别以测试集中全部 377 条 real 样本和 1,150 条 fake 样本为分母。模型请求失败、输出无法解析、缺少 `verdict` 字段，或输出不是有效 `real/fake` 标签时，均作为所属真实类别的未召回样本处理。

每类 Precision 为 `TP / (TP + FP)`；Macro-Precision 是两类 Precision 的等权平均，不按类别数量加权。若某类没有任何有效预测，Precision 定义为 0。

每类 F1 为 `2TP / (2TP + FP + FN)`；Macro-F1 是两类 F1 的等权平均，**不是**先对 Precision 和 Recall 求宏平均后再取调和平均。缺图、失败和无效输出均计入真实类别的 FN，但没有实际预测类别，不虚构为另一类的 FP；它们也不作为第三个类别参加宏平均。因此 Precision 没有统一的 1,527 分母，但 Recall/F1 仍覆盖冻结测试集全部样本。

严格证据充分率使用同一批 1,527 条样本作为分母，对应 LLM Judge 质量评估中的 `Strong` 类别。该指标强调结论之外的证据覆盖、来源可靠性及推理充分性，不等同于二分类准确率。

GPT-5.5 与 MiniMax-M3 的原始结果各覆盖 1,526 条样本，未覆盖样本仍按错误计入统一分母。GPT-5.5 另有 2 条源答案无有效终态，无法进入证据质量评估；这些样本同样不从分母中剔除。

Qwen3.5-9B（训练前）Direct QA 已完成 1,526 条唯一推理及对应的 1,526 条审核。已逐条核对冻结测试集、私有 gold、源预测与审核记录，real 正确 298/377、fake 正确 602/1,150，缺失的 1 条 real 仍计为未召回，因此 BAcc 为 65.70%、Real Recall 为 79.05%、Fake Recall 为 52.35%。沿用已确认的 Gemini 3.7 Flash、`thinking_level=low`、`max_output_tokens=8192` 直接 QA 审核，严格证据充分数为 26/1,527，SESR 为 1.70%；该结果不是 Agent 基线，也未因补表重新调用 judge。源结果位于服务器 `/volume/ybo/wza/runs/eval/qwen35-base-directqa-formal1527-20260912/results.jsonl`，审核记录位于 `/volume/ybo/wza/runs/eval/qwen35-base-directqa-formal1527-20260912-judged/audit-results.jsonl`。

Qwen3.5-9B Agent（训练前）的二分类结果按冻结 case ID 复用已有的 1,526 条推理。缺失的 1 条 real 按未召回计入 1,527 分母；real 正确 264/377，fake 正确 705/1,150，其 Accuracy 为 63.46%。表中的 SESR 已更新为最终 v3 全材料审核结果 48/1,527（3.14%），替代旧版 44/1,527（2.88%）。二分类记录保存在服务器 `qwen35-base-agent-formal1527-pretrain-from-old1682-20260912/summary.json`；v3 审核保存在 `/volume/ybo/wza/runs/eval/gemini37-qwen35base-agent-batch-judge-v3-low32k-20260913`。旧版 Gemini 3.1 Pro/high 审核不进入本表。

Qwen3.5-9B Agent（SFT-2578）的冻结结果覆盖 1,526 条可运行样本。该模型从 Qwen3.5-9B Base 独立启动训练，并非 SFT-4872 的前置阶段。原始汇总按 1,526 条可用样本给出的 BAcc 为 79.98%；本表遵守统一定义，把缺失的 1 条 real 计为未召回，因此记录 BAcc 79.87%、Real Recall 79.58%、Fake Recall 80.17%。冻结结果位于服务器 `/volume/ybo/wza/evaluation/qwen35-sft-initial2578-agent-final1526-20260913`。

Qwen3.5-9B Agent（SFT-4872）的冻结结果同样覆盖 1,526 条可运行样本。该模型也从同一个 Qwen3.5-9B Base 独立启动训练，不是从 SFT-2578 checkpoint 续训。可用样本口径为 real 313/376、fake 884/1,150，BAcc 80.06%；本表把缺失的 1 条 real 计为未召回，因此记录 Real Recall 83.02%、Fake Recall 76.87%、BAcc 79.95%。冻结结果位于服务器 `/volume/ybo/wza/evaluation/qwen35-sft1028-agent-final1526-20260914`；统一 v3 审核得到 `Strong` 558 条，即 SESR 558/1,527（36.54%）。

Gemini 3.1 Pro 的证据质量已使用统一的 Gemini 3.7 Flash、`thinking_level=low` 配置重新审核。最终有 85 条样本被判定为 `Strong`，SESR 为 5.57%；旧的 23.71% 来自 Gemini 3.1 Pro 自审且使用 `thinking_level=high` 的非统一口径，已废弃。

### 恢复依据与复算（2026-09-17）

20 组结果的整数混淆计数、缺失/无效数量和来源哈希保存在 [指标计数与来源](evaluation-metric-counts-20260917.json)，不包含逐样本隐私数据。主表新增的两列均由这些整数重新计算，并校验全部 20 组原 BAcc 和两类 Recall 四舍五入后完全一致。

- 9 组来自本次核对的 H20 冻结记录：8 组逐条预测/候选标签按 case ID 与冻结 gold 对齐，epoch2 使用已冻结的完整混淆矩阵。
- 另 11 组（旧 10 组 Direct QA 和 Gemini 3.7 Agent）来自原任务 `01a06fd4-69e4-7b92-bfdd-34efac6a8aed` 保存的工具输出：第 34755 行的各类正确/错误/无效整数汇总，以及第 34763 行的 GPT-5.5、MiniMax-M3 整数混淆矩阵。本次未重新登录旧服务器复核这些逐样本原文件，也没有从两位小数百分比反推计数。
- Gemini 3.7 Agent 的原分类记录覆盖 1,527 条；后续只迁移了 1,526 条审核材料，不应因此把原先已知的分类结果改成缺失。分类统计和 SESR 的原口径均保留。
- SESR 沿用既有最终审核结果；epoch2 原本待审且暂停新提交，仍保留该状态。页首缓存缺陷限定不因补指标而解除。

复算命令：`python scripts/rebuild_comparison_metrics.py`。该脚本只读聚合计数，不启动模型、训练或付费 judge；修改主表时需同步计数记录，回归测试会检查两张主表与复算结果一致。

## Agent v3 Batch judge 完整性记录

以下八组均使用 Gemini 3.7 Flash、`thinking_level=low`、`max_output_tokens=32768` 和 v3 全材料输入。最早六组各有1,526条有效审核；后完成的epoch3和Gemini3.1Pro各有1,525条有效审核与1条终态失败。失败或不可运行的样本均不从1,527报告分母中删除。

| Agent | 完成数 | 严格证据充分数 | SESR（1,527 分母） | 服务器记录 |
|:---|---:|---:|---:|:---|
| GPT-5.5 Agent | 1,526/1,526 | 771 | 50.49% | `gpt55-qwen397-agent-batch-judge-v3-low32k-20260913/gpt55-agent` |
| Qwen3.5-9B Agent（同次三epoch训练，epoch3；缓存限定） | 1,525有效＋1失败 | 682 | 44.66% | `sft3084-gemini31pro-hybrid-judge-v3-low32k-20260915/qwen35-9b-sft3084-agent` |
| Qwen3.5-397B-A17B Agent | 1,526/1,526 | 661 | 43.29% | `gpt55-qwen397-agent-batch-judge-v3-low32k-20260913/qwen35-397b-a17b-agent` |
| Gemini 3.7 Agent | 1,526/1,526 | 655 | 42.89% | `gemini37-qwen35base-agent-batch-judge-v3-low32k-20260913/gemini37-agent` |
| Qwen3.5-9B Agent（SFT-4872） | 1,526/1,526 | 558 | 36.54% | `qwen35-sft1028-agent-batch-judge-v3-low32k-20260914/qwen35-9b-sft1028-agent` |
| Qwen3.5-9B Agent（SFT-2578） | 1,526/1,526 | 531 | 34.77% | `qwen35-sft-agent-final1526-batch-judge-v3-low32k-20260913/qwen35-9b-sft-agent` |
| Gemini 3.1 Pro Agent | 1,525有效＋1失败 | 499 | 32.68% | `sft3084-gemini31pro-hybrid-judge-v3-low32k-20260915/gemini31pro-agent` |
| Qwen3.5-9B Agent（训练前） | 1,526/1,526 | 48 | 3.14% | `gemini37-qwen35base-agent-batch-judge-v3-low32k-20260913/qwen35-base-agent` |

以上八组judge均已收尾，不应重复提交；新增两组使用Batch与普通API互斥合并。2026-09-15，Gemini 3.1 Pro Agent 的工程失败补跑已完成，冻结 1,526 个唯一成功 case：real 正确 112/377、fake 正确1094/1150，BAcc62.42%。旧磁盘满失败及所有补跑仍保留，合并索引位于服务器 `gemini31pro-agent-formal1527-20260913/resume-control-20260914/merged-success-results.jsonl`。以下时间序列保留早期快照，以本页09:35收尾状态为准。

另一次从 Base 初始化的三 epoch SFT 已完成 epoch-3/step-3084 全量 Agent 推理，冻结 1,526 个成功 case：real 正确 295/377、fake 正确 919/1150，BAcc 79.08%。与前述 SFT-4872 单 epoch 实验不是续跑关系。epoch-3 最终选择索引为 `qwen35-sft3084-3epoch-agent-formal1527-20260915/selected-traces.json`。

同一次三 epoch 训练的 epoch-2/step-2056 已于 2026-09-16 用完原协议下的有限补跑：1,524 个唯一成功 case、2 条失败、另 1 条缺图不可运行，不称为 1,526 条全部成功。real 正确 284/377、fake 正确 923/1150，对应 BAcc 77.80%、Real Recall 75.33%、Fake Recall 80.26%；失败和缺图均未从分母剔除。两条最终失败均为 `finish_reason=length` 且无可用答案，此前还发生过读超时。冻结索引及汇总位于服务器 `/volume/ybo/wza/evaluation/qwen35-sft2056-epoch2-agent-budgeted1524-20260916`，索引 SHA256 为 `104bd65fec08f13966156beea4e012363d2ec02ddc67ec7a8f50066efcc210b0`。该轮 `thinking_token_budget=8192` 只是请求值，原 vLLM 未执行此预算；后续隔离 PSD 的预算修复不追溯改变本轮协议或结果。1,524 条有效轨迹的统一 v3 审核候选已准备在 `/volume/ybo/wza/evaluation/sft2056-v3-judge-20260916`，包含 20,275 个原始 observation，gzip 约 21.89 MiB、SHA256 `0bffefb2f4f9b6d9ecb8fcce4d8f0b64eb2cfd3cdb3fa2864cec032dae1a577c`；尚未提交，SESR 仍待审核，不能复用 epoch3 结果。

epoch3 与 Gemini 3.1 Pro 两组同口径 v3 judge 继续采用已固定的混合传输：已受理 28 个 Batch 共 149 条，其余 2,903 条分配给普通 API，不能重新启动旧 Batch 提交阶段。2026-09-16 05:04 巡检普通调用已有1,241条有效结果（epoch3 622、Pro619），25条明确拒绝未记作成功；Batch 28个仍为RUNNING、0个已回收，尚无最终SESR。epoch2不属于这两组既有任务。详情见 [普通 judge 切换记录](judge-realtime-switch-20260915.md)。

05:13，epoch2独立提交入口完成精确case-ID关联：1,523条可按原图内联分为282个Batch，另1条GIF编码后约23.5MB超过保守内联阈值，保留待File API配额处理，不缩图、不丢图。05:13及06:41两次实际Batch创建均被HTTP429拒绝，仍0受理。独立收据位于上述epoch2准备目录的`create-intents`和`submission-progress.json`。发现跨图片感知缓存后已加入`data-quality-hold.json`并在启动及逐片提交前强制检查，**原按一小时冷却续交的规则已暂停，不能仅因配额恢复就新付费提交**。不删除收据、不恢复旧epoch3/Pro Batch提交器，也不自动重跑历史全量。[Gemini Batch内联限制](https://ai.google.dev/gemini-api/docs/batch-api)

07:02，仍运行的普通judge有效结果为epoch3 1,178条、Pro 1,177条，两个worker继续运行；07:06原Batch collector再次观测28个RUNNING、0已回收、无collector错误。它们审核的是原冻结输出，结果需保留本页顶部的缓存缺陷限定，尚无最终SESR。

## 初步结论

- Gemini 3.7 Agent 获得当前最高 BAcc，为 81.57%，且 real/fake 两类召回较为均衡。
- GPT-5.5 Agent 获得当前最高 SESR，为 50.49%；Gemini 3.7 Agent 与 Qwen3.5-397B-A17B Agent 分别为 42.89% 和 43.29%。
- 从 Base 独立训练的 SFT-2578 在严格 1,527 分母下达到 79.87% BAcc 和 34.77% SESR，显著高于训练前的 65.67% 和 3.14%。
- 同样从 Base 独立训练的 SFT-4872 达到 79.95% BAcc 和 36.54% SESR；与 SFT-2578 横向比较，SESR 高 1.77 个百分点，Real Recall 高 3.44 个百分点，但 Fake Recall 低 3.30 个百分点。两者不是前后续训关系。
- GPT-5.5 与 Gemini 3.7 Flash 的 BAcc 分别为 75.87% 和 75.64%，是表现最好的两组 Direct QA 基线。
- Gemini 3.1 Pro 虽然原始 Accuracy 较高，但 real 召回仅为 28.65%，BAcc 因此降至 62.58%。
- Agent 的优势体现在证据搜集与证据支持的推理质量，但训练前 Qwen3.5-9B Agent 的 SESR 仅 3.14%，说明该优势并非仅由 Agent 流程保证。

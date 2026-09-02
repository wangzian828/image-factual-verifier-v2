# 新版 Agent-100 真实重跑与逐条审阅

日期：2026 年 9 月 2 日

## 运行配置

- 模型：Gemini 3.7 Flash
- Agent：当前 `unified-react-v1`
- 样本：复用历史 Agent-100 的同一批 100 条测试 case
- 构成：5 个构造子路线各 20 条
- Agent 并发：10
- 工程恢复：最多 4 个 attempt，只补跑没有 terminal success 的 case
- 测试集 release：
  `/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini37-agent-test100-newagent-release-20260902-r55ef0b3/runtime-release/`
- rollout：
  `/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini37-agent-test100-newagent-rollout-20260902-r55ef0b3/`

测试 release 只给 Agent 提供 `case_id`、图片路径和 SHA-256。
private gold 和来源访问策略没有进入 Agent 输入。

## Rollout 结果

- 100 个 case 中 99 条生成了成功轨迹；
- 1 条 case 在 4 次 attempt 中都被 Gemini 3.7 的 HTTP 400
  `content_blocked` 拒绝；
- 该 case 是：
  `route-aware-hrc-final-3000-20260816-input:2920:web_r015-web_crawled_refuted-web-backlog-00000`
- 该 case 没有产生模型请求、工具调用或可审计轨迹，单独记为 provider
  `content_blocked`，不计入语义三分类；
- 工程恢复队列正常工作，没有发现连接泄漏式增长。

## 统一 private-gold judge

Judge 使用 Gemini 3.7 Flash，并发 12，输出上限 8192。
审计输入为 99 条 merged terminal trace；第 100 条 provider error 仍保留在
审计结果中，状态为 error。

审计目录：

`/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini37-agent-test100-newagent-judge-gemini37-20260902-r55ef0b3-r3/`

主三分类：

| 分类 | 数量 | 占全部 100 条 | 占 99 条可审计轨迹 |
|---|---:|---:|---:|
| 判断正确且理由/依据充分 | 34 | 34.00% | 34.34% |
| 判断正确但理由/依据不足 | 40 | 40.00% | 40.40% |
| 判断错误 | 25 | 25.00% | 25.25% |
| provider `content_blocked` | 1 | 1.00% | 不适用 |

因此：

- 99 条成功轨迹的 Accuracy：`74 / 99 = 74.75%`；
- 100 个 case 的完成率：`99.00%`；
- 若把未完成的 provider case 作为整体 case 覆盖率中的非正确项，正确数为
  `74 / 100 = 74.00%`，这不是语义 Accuracy；
- gold 为 real 的 40 条中，正确 31 条，召回率 77.50%；
- gold 为 fake 的 59 条中，正确 43 条，召回率 72.88%；
- BACC：`(77.50% + 72.88%) / 2 = 75.19%`。

其他 judge 诊断：

- `reason_quality`：决定性且有依据 34、部分有依据 21、artifact-based 21、
  unsupported 19、contradictory 4；
- `fact_alignment`：same fact 54、compatible subfact 20、
  overgeneralized subfact 17、different fact 8；
- quality bucket：strong 32、usable 23、rejected 44。

主三分类以 `verdict_matches_gold` 和 `reason_quality` 为准；quality bucket
还包含更严格的事实对齐和报告忠实度约束，因此两组数字不要求相等。

## 逐条审阅

每个 case 都已经生成一条审阅记录：

- JSONL：
  `/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini37-agent-test100-newagent-rollout-20260902-r55ef0b3/trace-quality-review.jsonl`
- Markdown：
  `/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini37-agent-test100-newagent-rollout-20260902-r55ef0b3/trace-quality-review.md`

审阅记录包含最终 trace、动作数、工具序列、Evidence 数、Discovery 数、
搜索空结果、外部访问失败、严格审计失败、private-gold 分类、事实对齐、
理由质量和 judge 解释。问题标签允许重叠，不是互斥分类。

本批自动诊断的主要问题：

- 69 条 thought/report 出现 AI 痕迹或视觉异常类理由；
- 22 条最终 Evidence ledger 为空；
- 19 条动作数不超过 3；
- 7 条包含外部访问失败；
- 6 条出现严格审计问题；
- 44 条被 judge 标记为理由质量问题；
- 25 条被 judge 标记为事实对齐问题。

这些标签用于人工复核，不表示每一条都必然是同一种根因。当前最集中、
需要后续统一处理的问题仍是：

1. 把 AI 生成痕迹、乱码、光影/反射或解剖观感当作事实反证；
2. 搜索为空或页面不可访问后仍快速结束；
3. 只验证图片中出现了什么，没有闭合图片表达的现实事实；
4. Evidence ledger 为空时，报告仍写成高置信度结论；
5. 事实核查 query 泄漏和 source policy rejection；
6. 把一个局部或不同事实扩大成完整目标。

本轮只记录问题，没有在看到这些结果后继续修改 Agent 结构或 prompt。

## 工程与文档收尾

- 代码提交：`55ef0b3 Harden Gemini scene perception recovery`
- 最终文档提交：`0f6efce docs: record new agent100 audit and trace review`
- gpu-13 已 fast-forward 到 `0f6efce`；
- 服务器全量测试：`468 passed`；
- 32 个旧 Jupyter kernel CLOSE-WAIT 已清理；当前只剩少量无所属进程的
  代理 socket，未继续增长；
- 训练集大规模教师 rollout、SFT 正式训练和 RL 正式训练均未启动；
- 本批 100 条仅用于 Agent 诊断和 private-gold 审计，不进入训练。

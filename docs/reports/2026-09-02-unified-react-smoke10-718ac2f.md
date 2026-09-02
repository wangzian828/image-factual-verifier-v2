# unified ReAct 新版本 10 条 smoke 与 SFT 链路记录

日期：2026 年 9 月 2 日

## 运行配置

- Agent：`unified-react-v1`
- trace 基线：`718ac2f`
- 合并器修复：`2e64147`
- 模型：Gemini 3.7 Flash
- rollout 并发：10
- 测试集：5 个构造子路线各 2 条
- source policy：当前 59 个事实核查域名

测试 release：

`/gsdata/home/wza/image-factual-verifier-v2-data/generated/direct-qa-baselines/gemini37-agent-smoke10-release-20260902-718ac2f/runtime-release/`

最终 merged trace：

`/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/unified-react-v1-gemini37-smoke10-20260902-p10-718ac2f/rollouts/test/merged/`

## Rollout 结果

- 10/10 case 最终生成 terminal trace；
- 首轮 8 条完成，2 条进入工程恢复 attempt-02 后完成；
- 最终 engineering error：0；
- 10 条均有最终 report；
- accepted ReAct action：73 次；
- Gemini 请求：121 次；
- action 数范围：3–17；
- `text_image_search` 主动调用：4/10 条；
- 候选图进入后续 Interaction，仍保持 Discovery，不自动升级为 Evidence；
- 使用真实 source policy 的 strict trace audit：10/10；
- CLOSE-WAIT：0。

独立运行 audit CLI 时必须传入本次 run 的 active source policy；否则它会把普通
`fact check` 词组误报成 query leak。本次最终审计使用 merged manifest 中的 59 域名
policy。

## 逐条检查

| case | verdict | actions | Evidence | Discovery | 观察 |
|---|---:|---:|---:|---:|---|
| 0443 | fake | 17 | 7 | 42 | 文搜图已使用；2 次视觉子调用 429，失败被保留 |
| 0558 | fake | 5 | 4 | 3 | 反向搜图与比较形成证据 |
| 2295 | real | 5 | 4 | 13 | 网页访问和一致性检查正常 |
| 2348 | real | 4 | 1 | 13 | 参考图外部不可访问，未生成比较证据 |
| 2400 | real | 5 | 3 | 21 | 搜索与访问链完整 |
| 2507 | fake | 3 | 3 | 11 | 搜索→访问→结束，未出现空轨迹 |
| 2587 | fake | 10 | 3 | 24 | 视觉子调用 2 次 429，未伪装成功 |
| 2689 | real | 7 | 1 | 33 | 文搜图已使用；参考图外部不可访问 |
| 0534 | fake | 10 | 8 | 27 | 1 次非法 bbox 被记录为工具错误 |
| 1852 | fake | 7 | 3 | 23 | 文搜图已使用，搜索和访问链完整 |

429、外部不可访问和非法 bbox 均没有被包装成有效 Evidence。本轮没有发现
工具结果丢失、空 `status=success` 冒充搜索结果或 source policy 泄漏。

## SFT eligibility

### Gemini 3.7 Flash

目录：

`/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/unified-react-v1-gemini37-smoke10-20260902-p10-718ac2f/sft-eligibility-gemini37/`

- 初次 judge 运行完成 7/10，3 条因 Gemini HTTP 429 未完成；
- 随后最小 Interactions 协议探针在并发 1 下恢复成功；
- 这 3 条可在同一目录按缓存续跑，之前的 7 条 artifact 不需要重算；
- 初次 7/10 结果不作为正式 3.7 分桶，需等续跑完成。

### Gemini 3.1 Pro Preview 对照

目录：

`/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/unified-react-v1-gemini37-smoke10-20260902-p10-718ac2f/sft-eligibility-gemini31pro/`

- 10/10 完成；
- 7/10 通过，3/10 拒绝；
- 该结果只验证新版 SFT packet、gold 和 judge 链路，不替代 3.7 指标。

judge 能看到有序 action、公开参数、完整文本观察、搜索候选、网页证据、
runtime 状态、失败记录、预算和 stop reason；report 只作为摘要，不能创造 Evidence。

## 3.7 配额与 100 条 preparation

并发 1 的最小 Gemini Interactions 探针仍返回 HTTP 429
`too_many_requests`。因此 100 条训练 rollout 尚未启动。

已完成的 100 条 preparation：

`/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/unified-react-v1-gemini37-train100-fourround-20260902-2e64147/`

- 输入：统一训练集 `train-manifest.jsonl`；
- runtime rows：100；
- private-gold rows：100；
- gold 缺失：0；
- 计划：初始 rollout + 最多 3 轮串行 quality reroll；
- rollout / SFT judge 并发：10；
- 100 条不进入正式训练；
- 全量 8,490 条教师 rollout 未启动。

3.7 恢复后从同一 preparation 目录启动；每轮先 rollout、再 SFT judge，
只把上一轮未通过者放入下一轮，四轮仍未通过者记为 hard case。

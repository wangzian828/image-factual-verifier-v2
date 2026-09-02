# Unified ReAct high thinking 10 条对照实验

日期：2026 年 9 月 2 日

## 实验变量

- Agent：`unified-react-v1`
- 模型：Gemini 3.7 Flash
- ReAct：`GEMINI_UNIFIED_REACT_THINKING_LEVEL=high`
- 其它阶段：保持 `low`
- case：与此前 low smoke 完全相同的 10 条
- 首轮并发：10
- 工程错误补跑并发：3
- Judgment：不改变

## 运行产物

首轮目录：

`/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/unified-react-v1-gemini37-react-high-smoke-20260902-p10-54c2ee4/`

3 条工程错误补跑目录：

`/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/unified-react-v1-gemini37-react-high-smoke-20260902-rerun3-c3-54c2ee4/`

最终 10 条合并清单：

`/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/unified-react-v1-gemini37-react-high-smoke-20260902-final10-c3-54c2ee4/`

private-gold 审计：

`/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/unified-react-v1-gemini37-react-high-smoke-20260902-final10-c3-54c2ee4/private-gold-audit-gemini37/`

## 完成情况

首轮 10 条中有 3 条在最终请求阶段遇到 Gemini HTTP 429，分别为
`main-02728`、`main-02730`、`main-02756`。降低并发到 3 后重跑，3 条全部成功。
最终合并结果为 10/10 成功，0 工程错误；原始首轮目录和补跑目录均保留。

最终选定轨迹的平均运行指标：

| 指标 | high ReAct |
|---|---:|
| case 完成 | 10/10 |
| 工程错误 | 0 |
| 平均耗时 | 343.67 秒 |
| 平均工具调用 | 9.4 |
| 平均 Gemini 请求 | 15.2 |
| strict trace audit | 10/10 |

## Private-gold 结果

| 分类 | 数量 | 比例 |
|---|---:|---:|
| 判断正确且理由充分 | 3 | 30% |
| 判断正确但理由不足 | 1 | 10% |
| 判断错误 | 6 | 60% |

判定正确的 4 条为：`main-02730`、`main-02731`、`main-02732`、
`main-02756`。其中前三条为 strong，`main-02756` 标签正确但理由被判为
artifact-based，不能进入高质量理由桶。

审计附加结果：

- quality bucket：strong 3、rejected 7；
- fact alignment：same_fact 3、overgeneralized_subfact 5、different_fact 2；
- reason quality：decisive_and_grounded 3、artifact_based 3、
  unsupported 3、contradictory 1。

## 与此前 low smoke 对比

| 指标 | low ReAct | high ReAct |
|---|---:|---:|
| 标签正确 | 4/10 | 4/10 |
| 平均耗时 | 124.80 秒 | 343.67 秒 |
| 平均工具调用 | 6.2 | 9.4 |
| 平均 Gemini 请求 | 9.8 | 15.2 |
| 最终工程错误 | 0 | 0（补跑后） |

结论：这 10 条固定 case 只能说明本批样例中的 high 表现，不能据此推广到整体
数据集。high 增加了调查耗时、工具调用和请求量；本批没有观察到标签正确率提升，
但也不能据此否定 high 对更大样本或更复杂 case 的价值。按当前运行决定，后续
`unified_react` 默认固定使用 high；如需做 low 对照，必须显式设置
`GEMINI_UNIFIED_REACT_THINKING_LEVEL=low` 并创建新的 run。

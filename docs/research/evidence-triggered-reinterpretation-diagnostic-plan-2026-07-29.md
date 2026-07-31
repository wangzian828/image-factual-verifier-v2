# Evidence-Triggered Visual Reinterpretation：行为诊断计划

**状态：** 已规划，暂不实施  
**记录日期：** 2026-07-29  
**适用基线：** 正式 v4 `discrepancy-first-v4`  

## 1. 研究动机

希望验证一种调查行为：随着外部搜索获得新知识，Agent 可能发现原先没有建模的细粒度视觉区别，因而重新观察原图，并让新的视觉观察改变后续调查或最终判断。

典型情形不是“马车与巴士”这类粗粒度纠错，而是“普通蝴蝶、帝王蝶与相似物种”之间的细粒度区分。新知识的价值在于告诉 Agent 原图中的什么细节具有鉴别意义。

目标行为链为：

```text
搜索获得新知识
→ 识别出此前未注意的、可在原图中检查的视觉区别
→ 针对该区别复看原图
→ 获得新的视觉观察
→ 改变后续调查方向或最终 Decision
```

这里暂称为 **evidence-triggered visual reinterpretation**。名称不是最终论文术语。

## 2. 当前判断

正式 v4 已经具备一条受限的 Evidence-to-Vision 路径：

```text
Web Evidence
→ Discrepancy Decision
→ VisualReinspectionRequest
→ focused_visual_inspection
→ pixel Evidence
→ Discrepancy Decision
```

但尚不清楚该路径在真实样本上是否能自然形成上述完整行为，也不清楚失败主要发生在搜索、复看触发、视觉问题生成、局部观察，还是复看后的状态消费。

因此，不先假设系统需要新的 belief architecture，也不先添加 `current_visual_interpretation`。先对不修改的正式 v4 做小规模行为诊断。

## 3. 冻结边界

诊断阶段保持以下内容不变：

- 不修改 Agent prompt、schema、reducer 或控制流；
- 不新增 belief、referent、interpretation revision 或 target revision 数据结构；
- 不调整现有一次 v4 visual reinspection 预算；
- 不把人工知道的鉴别特征输入 Agent；
- 不用 evaluator-private gold 引导搜索或运行时判断；
- 不以最终准确率代替过程诊断。

该计划是研究记录，不替代 `docs/architecture.md`、`docs/gemini-interaction-sequence.md` 或当前 runtime 契约。

## 4. 第一轮诊断样本

先选 5–10 个高诊断价值样本，不直接扩展到 20–30 个。每个样本应满足：

1. 初看可以形成合理但较粗或可能错误的视觉解释；
2. 开放网络中存在可检索的相关知识；
3. 该知识能揭示一个此前未显式建模的细粒度视觉区别；
4. 该区别原则上能在原图中被检查，而非完全依赖不可见 provenance；
5. 原图分辨率至少有可能支持该检查；
6. 我们能够事后人工判断合理的搜索知识、鉴别细节和复看区域。

优先类别包括物种、产品型号、制服或徽记、地标局部、机械部件和细粒度工业对象。样本应覆盖成功可能性和视觉不可判定情形，避免只挑必然成功的展示案例。

## 5. 每个样本的诊断记录

逐个检查以下阶段：

| 阶段 | 核心问题 | 记录内容 |
|---|---|---|
| 初始理解 | Agent 起初怎样描述相关对象或关系？ | 初始 Claim、视觉 anchors、候选解释 |
| 知识发现 | 是否搜到正确且可用的新知识？ | query、来源、exact Evidence、鉴别知识 |
| Evidence-to-Vision | 是否意识到该知识对应原图中可检查的区别？ | Decision 理由、是否请求复看 |
| 复看任务 | 问题和区域是否具有区分性？ | question、expected property、anchor/crop |
| 视觉观察 | 是否正确看出关键细节或诚实返回模糊？ | observed / not_observed / ambiguous、limitations |
| 状态消费 | 新观察是否进入下一次 Decision？ | Claim、hypothesis、discrepancy 的变化 |
| 因果影响 | 如果没有该次复看，后续动作或 verdict 是否会不同？ | 下一步动作、停止点、最终判断 |

最后一项是核心：只产生一段好看的 `before/after` 描述，但不影响后续行为，不算机制成功。

## 6. 失败定位与后续动作

诊断后只修复最早断裂的一环：

| 观察结果 | 初步归因 | 后续方向 |
|---|---|---|
| 没有搜到相关知识 | 搜索策略或来源访问问题 | 先修搜索，不增加 belief state |
| 搜到区别，但没有触发复看 | Evidence-to-Vision 触发问题 | 调整 Discrepancy Decision 的复看策略 |
| 请求复看，但问题或区域无区分度 | bridge/schema/region 问题 | 改进 VisualReinspectionRequest 或 crop 选择 |
| 问题正确，但视觉模型无法观察 | 分辨率、crop 或视觉能力问题 | 改进视图；允许并保留 ambiguous |
| 正确观察，但后续仍沿用旧理解 | 状态缺少可消费的解释更新 | 再评估最小 `current_visual_interpretation` |
| 闭环已发生并改变行为 | 现有能力存在 | 提高触发可靠性，并设计机制消融 |
| 搜索一开始就偏离原核查问题 | 调查策略漂移 | 先处理 target preservation / route control |

只有第五种结果能直接支持新增显式 interpretation state；在此之前不预设该改动必要。

## 7. 第一轮输出

第一轮不追求 benchmark 结论，只产出：

- 5–10 条完整 canonical trace；
- 一张逐阶段诊断表；
- 每个样本最早失败环节的人工标注；
- 各失败类型的计数；
- 是否需要修改 Agent 结构的 go / no-go 判断；
- 如需修改，只提出最小改动和对应消融，不直接扩展为复杂 belief system。

## 8. 暂不做的事情

- 不把 `before / after / cause` 日志包装成方法贡献；
- 不强制所有 Web Evidence 都经过视觉复看；
- 不把 Visual Observer 人为隔离为看不到候选或外部信息的“盲法官”；
- 不建立自由改写的多层 belief graph；
- 不允许新解释静默替换原始核查目标；
- 不在行为证据出现前承诺这是论文的核心方法贡献。

## 9. 论文层面的判定标准

只有当完整闭环相较于“搜索后直接判断”或“复看但不消费解释变化”的基线，能够稳定改变正确的后续行为，并控制调查漂移与错误诱导时，才将其提升为方法贡献。

潜在消融将在诊断通过后再确定，候选包括：

1. 无 targeted visual reinspection；
2. 有 reinspection，但后续 Decision 不消费其结果；
3. 完整 Evidence-to-Vision 闭环。

潜在过程指标包括有效复看触发率、鉴别问题正确率、视觉观察正确率、decision-impact rate、错误线索诱发率和最终任务表现。当前均为候选，不作为既定评测协议。

## 10. 下一次执行起点

恢复该计划时，第一步是从现有数据中只读筛选候选样本，并建立诊断表；在完成首轮原样 v4 运行和人工轨迹审计前，不修改 Agent。

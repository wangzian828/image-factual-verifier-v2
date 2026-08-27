# Agent 结构

IFV 是“策略模型 + 工具 + runtime state”的闭环，不是多个固定顺序的 Agent 阶段。

```text
                 ┌────────────────────────────┐
图片、case 字段 ─>│ unified-react-v1 policy     │
                 │ thought -> 一个工具调用     │
                 └─────────────┬──────────────┘
                               │
                         runtime adapter
                               │
       ┌───────────────────────┼───────────────────────┐
       │                       │                       │
  视觉工具                 检索工具                控制工具
 scene/OCR/compare      search/visit/read       stop_route
       │                       │                       │
       └──────────────> reducer/state delta <──────────┘
                               │
                  Reflection / Decision / Judgment
                               │
                         canonical trace
```

模型每轮只能调用一个 native tool。Replan 不是独立请求：换 query、换网页、换视觉方向，都
直接表现为下一轮 ReAct 的新动作。Reflection 只做低频全局检查；Discrepancy Decision 只
处理已有证据的语义；Judgment 只处理 runtime 编译的最终 basis。

字段边界：

- `target_facts`：图片传达、需要核查的正向现实事实；
- `search_hypotheses` / `tasks`：runtime 管理的路线和任务；
- `discoveries`：检索线索；
- `evidence`：成功工具结果中可追溯的事实材料；
- `failures`：访问失败或工具失败记录；
- `verdict_basis`：最终结论实际引用的对象集合。

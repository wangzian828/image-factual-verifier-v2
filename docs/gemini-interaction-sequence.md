# Gemini 交互顺序

## 单条 episode

```text
创建空 workspace
  -> Gemini ReAct：选择一个视觉工具
  -> function result
  -> Reducer 写入观察和 delta
  -> Gemini ReAct：选择另一个视觉工具
  -> function result
  -> Gemini ReAct：首次调查动作 + investigation_intent
  -> ...重复 ReAct action...
  -> 低频 Reflection / Discrepancy Decision
  -> runtime 编译 basis
  -> Judgment
```

每个 ReAct action 只允许一个 native function。`previous_interaction_id` 只用于同一个
function call/function result 短链，不跨 action 复用。工具失败写入当前 action；外部不可访问
不等同于工程失败，malformed contract 才进入工程错误路径。

## 训练记录

每个 policy turn 保存 provider 实际返回的 thought（如果有）、函数名、参数、工具结果和紧凑
state delta。隐藏思考链不进入数据；只有 API 实际返回的可读 thought 才能进入 reasoning SFT。

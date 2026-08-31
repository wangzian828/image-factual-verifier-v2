# Gemini 交互顺序

当前生产路径是 `unified-react-v1`。本文件只描述当前主流程；旧的
Claim/route/task 链路见带日期的历史计划和实验记录。

## 单条 episode

```text
创建空 workspace
  -> Gemini ReAct：选择任意一个当前可用工具
  -> function result
  -> Reducer 写入观察和 delta
  -> Gemini ReAct：根据原图和紧凑记忆选择下一动作
  -> ...重复 ReAct action...
  -> finish_investigation 或达到预算
  -> runtime 编译 basis
  -> Judgment
```

每个 ReAct action 只允许一个 native function。视觉工具是普通工具，顺序不固定，后续也可以
再次调用。`previous_interaction_id` 不跨 action 复用；每个新 action 都用新的紧凑请求。
原图临时附加到请求，不写入累计文本历史。

工具失败写入当前 action 和 runtime failure 列表；外部不可访问不等同于工程失败，
malformed contract 才进入工程错误路径。

## 训练记录

每个 policy turn 保存 provider 实际返回的 thought（如果有）、函数名、参数、工具结果和紧凑
state delta。隐藏思考链不进入数据；只有 API 实际返回的可读 thought 才能进入 reasoning SFT。

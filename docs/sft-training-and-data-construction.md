# SFT 训练数据构造

## 1. 一条训练样本

一条样本对应一条完整 episode，不按每个阶段拆成互不相干的样本：

```text
system
user：图片 case + 初始观察
assistant：<think>...</think><tool_call>...</tool_call>
tool：真实结果 + 紧凑 state delta
assistant：<think>...</think><tool_call>...</tool_call>
...
assistant：Decision / Reflection / Judgment 结构化输出
```

ReAct 的行为目标是 thought 和工具调用；tool observation 只作为上下文，不作为模型要模仿的
输出。Qwen 使用原生 `<think>` 与 `<tool_call><function=...>` 模板，不自造另一套格式。

搜索动作要区分：

- `text_search`：文字到网页候选；
- `text_image_search`：文字到图片/图片页面候选；
- `reverse_image_search`：当前图片到图像对应候选。

搜索候选在训练上下文中保留为 Discovery，不能被导出器或 SFT judge 自动当成
Evidence；只有后续 `visit`、比较或其他成功观察形成的内容才进入证据链。

## 2. 上下文、loss 和长度

- 完整历史保留在 episode；每轮输入只放当前紧凑 workspace、最近观察、开放 gap、活跃路线和
  state delta，不重复嵌入完整累计 workspace。
- loss 只监督 assistant 输出：可见 thought、tool call 以及 Decision/Reflection/Judgment 的
 结构化策略输出。
- user、tool observation、state delta 和图片元数据只提供上下文，不计算监督 loss。
- 真实 provider 没有返回可读 thought 的 ReAct turn，不伪造 `<think>`；整条轨迹进入
  `action_only`，可供动作模仿或 RL 初始化，不能混入 reasoning SFT。
- 用真实 Qwen processor 编码后以 128K 作为准入线；超出者保留到 holdout，不截断核心轨迹。
- SFT eligibility judge 会看到按真实顺序排列的 ReAct 工具动作、工具观察、
  搜索候选和 state delta；图片二进制、base64、raw HTML 和 provider 内部快照不
  进入 judge packet。候选仍与 Evidence 分开。

## 3. 质量分桶

SFT judge 在工程审计之后运行。通过轨迹继续按质量分桶，例如高质量、可用但较弱、holdout；
所有原始轨迹保留。另标记：

- `reasoning_sft`：每个 ReAct action 都有 provider thought；
- `action_only`：动作可用但 thought 不完整；
- `rl_candidate`：可进入后续奖励/策略优化流程。

## 4. 图片 API 分工

- `direct_multimodal`：每个 Gemini Interaction 的根请求携带一份受控原图，
  后续请求复用 provider session 中的原图；
- `separate_vlm`：视觉工具/VLM 处理图片，主策略收到结构化观察。

在 direct 模式中，原图是每个请求的独立多模态输入，不追加到累计文本历史，也不写入
SFT 消息或 policy snapshot 的 base64。视觉 API 统一使用最长边 1024、JPEG quality 95；
环境变量只能在不超过 1024 的范围内调整。结构化视觉观察只保留有限实体属性、可见关系、
场景细节和不确定性。两种模式的工具 schema、Reducer、trace 和训练消息格式保持一致。

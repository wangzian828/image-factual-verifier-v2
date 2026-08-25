# SFT 训练与数据构造

## 结论

- 完整 trace 是归档、审计和 RL 的基础数据。
- 当前 policy SFT 导出为“一条 accepted case 对应一行完整 episode”。
- 这不是 search agent 唯一格式；也可以从完整 trace 派生“历史上下文 → 下一步动作”的样本。
- 协议纠错、失败和拒绝轨迹保留，但不直接作为主 SFT 目标；需要时进入 recovery / reroll 桶。

## 1. 数据边界

- rollout、SFT judge、SFT 和 RL 只读取 `train-manifest.jsonl`。
- `test-manifest.jsonl` 只用于独立评测和 QA。
- Agent 只看公开输入；标签、构造信息和 private gold 在 trace 完成后才用于审计。
- SFT 与 RL 可以引用同一批原始轨迹，但导出格式和筛选标准独立。

## 2. 自动流程

```text
rollout
  → 工程重试
  → 确定性审计 + SFT LLM judge
  → 质量分桶
  → 拒绝 / final-only 质量 reroll
  → 再审计
  → accepted release
  → policy / perception 数据转换
  → 生成 training_plan.json
```

训练默认不自动启动；只有显式 `--run-training` 才启动 GPU 训练。

## 3. 轨迹分桶

| 桶 | 含义 | 用途 |
| --- | --- | --- |
| `high` | 通过审计，证据链和行为基本干净 | 主训练候选 |
| `usable` | 通过审计，但有轻微 warning | 次级训练候选 |
| `rejected` | 已结束，但未通过质量门槛 | 保留并可 reroll |
| `engineering_error` | 未正常结束或有未恢复工程/协议错误 | 自动重试或诊断 |

通过轨迹再记录路线：

- `early_correct_judgment`：较早的 `image_only_discrepancy_judgment` 已给出正确标签。
- `final_only_judgment`：直到强制最终判断才首次给出正确标签。

所有轨迹都进入归档；同一 case 的多次 rollout 最终只选一条进入主 SFT 行，避免重复加权。

## 4. SFT 审计

确定性审计检查：

- trace 是否完整结束；
- 工具协议、图片和 hash 是否正确；
- Evidence 是否来自成功调用；
- 是否有不可恢复的工程或安全错误；
- 是否可导出为训练样本。

LLM judge 检查：

- verdict 是否正确；
- 是否核查了图片真正表达的事实；
- 是否有 decisive Evidence；
- 是否存在错绑证据或过度推断。

工程错误和质量拒绝分开处理：工程错误重跑，质量拒绝进入质量 reroll。

## 5. policy SFT 数据如何构造

### 5.1 完整轨迹与训练行

当前主 policy SFT 使用完整 episode：每个最终选中的 case 对应一行数据。完整轨迹仍然是
归档、审计和 RL 的原始来源；同一 case 的工程重试和质量 reroll 不会覆盖原轨迹。

当前 Qwen 原生格式版本为 `ifv-trajectory-sft-dataset-v2`；旧的自定义
`role=tool_call/tool_response` 数据不再作为默认训练输入。

search agent 也常见另一种格式：从完整轨迹按决策点派生多条
`历史上下文 → 下一步动作` 样本。两者都属于 Agent SFT；本项目当前默认使用前者，今后
如受上下文长度限制，可以从同一份完整轨迹派生后者，不需要删除原始轨迹。

### 5.2 一行样本的拼接

导出直接使用 Qwen3.5 的原生 chat-template 兼容结构，不自定义新的工具协议。训练行仍是
`messages + tools`，但 ReAct 的 assistant 内容直接使用 Qwen 模板产生的文本；格式是
`<think>...</think>` 和 `<tool_call><function=...><parameter=...>`。

一行按真实发生顺序拼成：

```text
system
  固定训练规则和角色说明
user
  初始 Perception/OCR 结果、case 标识和公开图片信息
user
  当前 stage packet：阶段、任务、workspace、可用工具或输出 schema
assistant
  非 ReAct：`<think>Gemini thought 摘要</think>` + 结构化 JSON
assistant
  ReAct：`<think>Gemini thought 摘要</think>` + Qwen 原生 `<tool_call>` 模板文本
tool
  工具真实返回的观察结果
user
  程序归并后的新 workspace 和下一阶段 packet
assistant
  下一次模型输出
...直到 Judgment 和结束
```

实际导出规则：

- 从 trace 提取初始 `perceive_scene` 和 `ocr_with_position` 结果；
- 每个 policy turn 的 `policy_input` 压缩成该轮 stage packet；
- ReAct 的 `policy_action` 转成 Qwen 模板的 `<tool_call><function=...>` 文本；
  参数使用 `<parameter=...>`，不使用 `role=tool_call`；
- 工具结果使用原生 `role=tool`；不使用 `role=tool_response`；
- Gemini thought 摘要直接写入 assistant `content` 的 `<think>...</think>`；
  这样 ms-swift 会对 thought 和 tool call 一起计算 loss；完整隐藏 CoT 不存在，也不导出；
- 其他阶段的 `policy_action` 是该阶段结构化 JSON；
- 工具返回值作为下一轮上下文，不当作教师答案；
- `tools` 字段记录该行使用的工具 schema；
- 不把原始 trace JSON、evaluator-private 字段或程序内部日志整体塞进训练上下文。

因此模型看到的是运行时真正依赖的状态交接，而不是事后才知道的答案或完整内部日志。

### 5.3 哪些内容计算 loss

| 内容 | loss | 作用 |
| --- | --- | --- |
| `system` | 否 | 固定规则 |
| `user` / stage packet | 否 | 当前任务和状态 |
| `tool` | 否 | 环境观察 |
| assistant `<think>...</think>` | 是 | 学习 Gemini 返回的 thought 摘要 |
| assistant 结构化输出 | 是 | 学习 Planning、Decision、Reflection、Replan、Judgment |
| assistant `<tool_call>` | 是 | 学习工具选择和参数生成 |

也就是说，loss 计算 assistant 中的 `<think>`、结构化输出和 `<tool_call>`；
`system`、`user`、`tool` 只作为上下文。当前转换使用
`loss_scale=ignore_empty_think`（当前 Qwen3.5 训练配置）：有内容的 thought 摘要参与
loss，空 thought 不产生额外目标。完整隐藏 CoT 不可得，也不会伪造或补写。

### 5.4 协议纠错如何处理

`planning_revision`、`format_error`、`output_rejected` 等步骤全部保留在原始 trace 和审计
记录中，用于排障、统计和 recovery 训练实验；默认不作为主 policy SFT 的正常目标。

可以恢复且最终成功的轨迹仍可进入 accepted 桶，但纠错动作不应被误当成理想策略。未恢复
的协议错误归入 `engineering_error`，自动重试，不进入主 SFT。

## 6. policy 与 perception 分开

| 数据 | 带图 | 监督目标 |
| --- | --- | --- |
| `ms-swift-policy` | 否 | 状态管理、工具调用、证据决策、终止判断 |
| `ms-swift-perception` | 是 | 场景、实体、文字、位置和不确定性 |

policy 使用 Perception/OCR 文本和后续工具观察，不重复上传原图；转换时使用
`chat_template_kwargs.enable_thinking=true`，让 Qwen 模板保持 thinking 模式。
perception 样本单独带
`images`，形式是“图片 + 感知指令 → `PerceptionReport`”。它不训练网页搜索、工具路线
或最终真假判断。

拆分原因是在线 Agent 已将策略调用和视觉调用分开：policy 学状态管理与调查决策，perception
学看图和定位，避免长轨迹每一步重复编码同一张图。

## 7. 训练设置

当前常用 profile：

```text
tuner_type=full
freeze_llm=false
freeze_vit=false
freeze_aligner=false
learning_rate=1e-5（按 profile）
micro_batch_size=1（按显存调整）
loss_scale=ignore_empty_think
max_length=32768（Qwen3.5 常用）
```

分布式使用已验证的 FSDP2 或 DeepSpeed ZeRO-3 profile。`max_length` 是训练长度，不等于
在线服务的上下文上限；超长轨迹需单独处理，不能静默截断。

## 8. 代码入口

- rollout：`scripts/trajectory/run_teacher_rollout_autopilot.py`
- 审计：`python -m src.eval.score_sft_eligibility`
- accepted release：`scripts/trajectory/stage_accepted_teacher_release.py`
- SFT 包：`scripts/trajectory/build_sft_training_package.py`
- policy 导出：`training/ifv_training/policy.py`
- perception 导出：`training/ifv_training/perception.py`
- 训练启动：`training/scripts/train/run_sft.sh`

训练输入是 `ms-swift-policy/`，以及可选的 `ms-swift-perception/`。审计、gold、catalog 和
原始轨迹目录只用于溯源，不直接作为模型输入。

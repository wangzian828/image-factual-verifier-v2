# Qwen3.5-9B 部署与训练设施统一执行计划

日期：2026-07-21
状态：已确认，待执行
Runtime：`visual-fact-discrepancy-agent-v4`
Training：`image-factual-verifier-training`
学生模型：`Qwen/Qwen3.5-9B`

> 本文是 Qwen 部署、SFT 和 Agent RL 的唯一执行主计划。2026-07-15 与
> 2026-07-16 的 Qwen3-VL 计划保留为调查记录；与本文冲突时以本文为准。

## 1. 已确认决策

1. 正式学生模型改为 `Qwen/Qwen3.5-9B`，不再以 Qwen3-VL-8B 为默认目标。
2. `Qwen3.5-4B` 作为快速启动模型：先单卡打通依赖、processor、serving、工具协议、
   v4 adapter 和短训练，再把同一门禁原样迁到 9B。4B 不替代正式学生，也不运行完整
   20 例。`Qwen3.5-35B-A3B` 只作为后续基础模型/LoRA 能力对照，不阻塞 9B 主线。
3. gpu-13 只使用物理 GPU `4,5,6,7`，每次按真实空闲数量使用，最多四张；
   不占用 `0,1,2,3`，不抢占或终止其他用户进程。
4. 暂停 Gemini 完整 20 例。先部署 Qwen 基础模型，完成协议与少量 canary 后，
   由 Qwen 依次运行完整 20 例；同一 20 例随后用于 Qwen base、SFT、RL 的冻结对照。
5. 20 例是开发评测集，不进入 SFT 或 RL 训练上下文；private gold 只在 rollout
   完成后由评估器读取。
6. SFT 使用成熟的 `ms-swift + DeepSpeed ZeRO-3`，目标是 Qwen3.5-9B 真正的
   全参数多模态训练；不以 LoRA 结果冒充全参数门禁。
7. Agent RL 保留 v4 runtime 的环境所有权，优先采用 `rLLM gateway + veRL`
   和 vLLM/SGLang rollout，不在训练框架中复制一套 v4 状态机。
8. Gemini 是冻结的离线教师、过程评分器和回归基线；最终 Qwen 部署不依赖 Gemini。
9. Prompt 继续保持简短，只描述任务、可用工具和输出 schema；调查语义不堆成
   大量“如果……那么……”规则。

## 2. 最终边界

```text
数据管线 release
  -> v4 deterministic runtime
       -> teacher-gemini（教师轨迹/冻结评分/基线）
       -> student-qwen35-local（部署、SFT、RL、最终运行）

v4 runtime 拥有：
  工具、Evidence、Finding、route、archive/recall、Coverage、停止、审计

Qwen policy 拥有：
  感知、Planning、ReAct action、Decision、Judgment
```

Gemini Interactions 和 Qwen OpenAI-compatible chat 是不同 wire protocol。两者只在
canonical stage input/output、tool schema 和 trace contract 上对齐；不得把 Gemini
`previous_interaction_id` 迁移到 Qwen，也不得在 Qwen 失败后静默回退 Gemini。

## 3. GPU 与任务策略

每次执行前重新读取 `nvidia-smi` 的显存、利用率、PID、进程用户和运行时长。只有
`4,5,6,7` 中确认空闲的卡可以加入 `CUDA_VISIBLE_DEVICES`，并在 manifest 同时记录
物理 ID 与进程内逻辑 ID。

| 可用卡数 | 允许任务 |
|---:|---|
| 1 | 模型/processor 探针、单卡 BF16 serving、短协议 smoke |
| 2 | serving、并发/工具协议、短 forward/backward 和显存测量 |
| 3 | serving、数据/适配验证；仅在实测可行时做 ZeRO-3 训练 smoke |
| 4 | 全参数 SFT smoke、正式 SFT、Agent RL smoke/训练 |

正式全参数 SFT 默认等待四张卡同时空闲。1～3 张卡空闲时继续做不会制造错误训练结论的
部署、协议、数据或正确性工作，不为赶进度降级为未声明的 LoRA。

所有任务必须：

- `OMP_NUM_THREADS=1`；
- 有显式 experiment/run ID、PID、GPU manifest、日志目录和停止命令；
- 拒绝非空输出目录，恢复训练必须显式指定 checkpoint；
- 将模型、数据、checkpoint、rollout 和日志写入 `/gsdata`，不写进 Git checkout；
- 不修改 `ifv-agent`，不在 gpu-13 直接编辑源代码。

## 4. 环境与仓库

### 4.1 仓库边界

```text
D:\image-factual-verifier-v2-worktrees\visual-fact-discrepancy-agent-v4
  provider-neutral runtime、Qwen adapter、trace/dataset exporter、评估

D:\image-factual-verifier-training
  processor 转换、SFT、RL gateway/backend、checkpoint/serving manifest
```

服务器对应路径继续位于 `/gs/home/wza/projects`。两个仓库只通过版本化 JSONL、manifest
和 checkpoint profile 交互，不互相 import Python package。

训练仓库当前存在尚未提交的 Qwen3-VL 全参改动。执行前先逐项审计和提交有用改动，删除
失效入口时保留研究记录；不得覆盖或丢弃已有工作。

### 4.2 独立环境

```text
ifv-agent
  已验收 Gemini runtime；只读依赖基线

ifv-qwen35-serve
  Python 3.12、Qwen3.5-compatible vLLM、OpenAI-compatible endpoint

ifv-qwen35-sft
  Python 3.12、ms-swift、Transformers、DeepSpeed、Qwen VL/FLA 依赖

ifv-qwen35-rl
  rLLM 固定 commit、veRL 固定 release/commit、vLLM/SGLang rollout
```

先在 gpu-13 核对 driver/CUDA/PyTorch wheel，再冻结精确版本。候选起点为当前正式发行的
`ms-swift 4.4.1`、`Transformers 5.14.x`、`vLLM 0.25.x`、`veRL 0.8.x`；文档版本
不是安装许可，只有真实 processor、serving、训练和恢复门禁通过后才能写入 lock manifest。

Qwen3.5 需要的 `flash-linear-attention`、`causal-conv1d` 和 Flash Attention 必须分别做
import、forward、backward 和 checkpoint 恢复测试。任何编译或 ABI 问题只影响 Qwen
环境，不得升级或污染 `ifv-agent`。

## 5. Serving 计划

### 5.1 初始配置

- 快速启动：`Qwen/Qwen3.5-4B`，单卡完成环境和端到端协议 smoke；
- 正式模型：`Qwen/Qwen3.5-9B`，固定下载 revision 和文件 SHA-256；
- dtype：BF16，不以 GPTQ/FP8 量化结果作为训练前基线；
- engine：vLLM；SGLang 只在 Agent RL throughput 有明确收益时加入；
- endpoint：loopback `127.0.0.1`，与 Jupyter 8333 和现有服务隔离；
- tool parser：Qwen3-compatible parser，通过实测后写入 serving profile；
- 初始 `max_model_len=32768`，通过 41k 级真实状态包后提高到 65536；
- 不因模型原生支持 262k 就向 runtime 开放 262k。

4B 和 9B BF16 都先在一张空闲卡上服务。4B 通过后必须用 9B 原样重跑 processor、图像、
structured output、工具和上下文门禁；不能以 4B 通过推断 9B 已通过。如果 9B 的 64k、多图
或并发造成显存不足，再在 `4,5,6,7` 内提高 tensor parallel，不能先占满四张卡掩盖配置问题。

### 5.2 Thinking A/B

Qwen3.5 支持 hybrid thinking。部署时用同一 base checkpoint 对四条既有 canary 做一次
`enable_thinking=false | true` 隔离比较，记录：

- verdict、Evidence chain 和 strict audit；
- Planning/查询是否更开放且相关；
- 工具调用、协议拒绝、总 token、最大单请求 token 和延迟；
- thinking 内容能否被 provider adapter 正确隔离，不污染 canonical action。

在比较完成前不把 thinking 开关写死为训练事实。SFT 不监督 Gemini 隐藏思维；只对
canonical 可见输出计算 loss。若 RL 开启 Qwen 自身 thinking，必须明确记录并正确计算其
token、logprob 和 mask。

### 5.3 协议门禁

以下门禁先用 4B 快速定位环境/协议问题，再由 9B 原样重跑并作为正式验收：

1. 文本请求和确定性 JSON/schema 输出；
2. 单张受控图像输入、OCR 和 positioned observation；
3. 正好一次 native tool call；
4. tool result continuation；
5. 八次短链工具交互，每次都由显式 StageHandoffPacket 重建状态；
6. Planning、Decision、Judgment 跨阶段不能继承旧 chat/session；
7. 并发 health probe、超时、取消和干净 shutdown；
8. 服务失败时显式 `engineering_error`，不得切换 Gemini。

## 6. 图像与上下文

- 原图按 SHA-256 永久保存，模型默认接收受控分辨率版本；
- 初始 perception/planning 图像 token 先限制在约 1024，训练 smoke 先用 512；
- ReAct 文本回合不重复发送原图；只有 `inspect_image` 或视觉 Decision 显式读取原图/crop；
- 每次图像发送记录原始尺寸、发送尺寸、crop、编码字节、估算 token 和用途；
- 单次请求正常目标 16k～32k，服务上限先 32k 后 64k，128k 是硬审计边界而非训练目标；
- 超长网页、重复工具结果和旧模型措辞留在 archive，通过 ID 精确回读，不拼入完整聊天记录。

训练样本是一项阶段决策，不是一整条几十轮对话：

```text
immutable brief + protected workspace + selected archive spans + tools
  -> 一个 Planning / ReAct / Decision / Judgment canonical output
```

工具返回、reducer 生成状态和外部网页文本是 observation，默认 loss mask 为 0。

## 7. Qwen runtime 接入

新增显式 profile：

```text
teacher-gemini
student-qwen35-local
```

Qwen adapter 负责：

- Qwen chat template 和图像占位符；
- tool schema、tool-call parser 和 tool result wire format；
- reasoning/thinking 内容隔离；
- JSON/schema correction 短链；
- token、图片和 provider usage 记录。

v4 orchestrator/reducer 不感知这些 provider 细节。Gemini/Qwen 必须消费同一公开 case、
同一显式工作区、同一工具和同一 canonical output schema。迁移期间允许混合 profile 做问题
定位，但正式 Qwen 20 例不得调用 Gemini perception、policy 或 Judgment。

## 8. 20 例执行顺序

20 例保持暂停，直到 Qwen base 完成下列前置：

1. processor、图像、structured output 和工具协议门禁通过；
2. Qwen adapter 的 provider-neutral 测试通过；
3. Queen、Andreea、Pillars、Monarch 四条 Qwen base canary 无工程错误并可 strict audit；
4. 最大单请求不超过 128k，且无跨阶段隐藏历史；
5. run manifest 明确记录 Qwen checkpoint、revision、thinking 配置和 serving profile。

随后按冻结配置运行完整 Qwen base 20 例。不得看一例改一次 Prompt；先完成整批，再按
错误类型分析模型能力、retrieval/tool、runtime control 和协议问题。保存：

- `real | fake` 结果和逐例错误；
- Claim、查询、Evidence、Discrepancy、basis、停止轨迹；
- action/tool/model call、token、图像 token、时间和显存；
- 工程错误、协议纠正、route 重复和 post-determination action；
- strict audit 与 training eligibility。

同一冻结 20 例在 SFT 和 RL checkpoint 后原样重跑，形成 `Qwen base -> Qwen SFT ->
Qwen SFT+RL` 对照。Gemini 四条已验收轨迹保留为教师/回归参照，不先运行 Gemini 20 例。

## 9. SFT 设施

### 9.1 数据契约

runtime 导出 provider-neutral release：

```text
manifest.json
episodes.jsonl
perception.jsonl
planning.jsonl
react.jsonl
decision.jsonl
judgment.jsonl
excluded_episodes.jsonl
assets/sha256/*
SHA256SUMS
```

每行包含 case/source-family、stage、teacher identity、runtime commit、model-visible input、
canonical target、引用 ID、loss policy 和质量门禁。以下内容不得进入 model-visible 字段：

- private gold、评估器 reference evidence 和 scorer 反馈；
- Gemini provider wire/interaction ID；
- 未经审计的隐藏思维；
- 不可回读的压缩事实。

train/validation/test 按 case 与 source family 同时隔离。20 例全部属于冻结开发评测，不得
被预处理器加入训练 split。

### 9.2 Processor 对齐

训练仓库使用准确 Qwen3.5 processor/chat template 生成派生数据，并验证：

- image placeholder 数量与资产一致；
- tool schema 和 tool-call target 可往返；
- assistant span/loss mask 精确；
- tool result、网页 observation 和 reducer 状态不产生 loss；
- truncation 不删除 protected context、引用或 target；
- processor/model revision 变化会生成新的 derived dataset version。

### 9.3 全参数门禁

使用公开或合成的多模态 smoke fixture，不使用 20 例 gold。只有四卡同时可用时运行：

```text
Qwen3.5-4B：1 forward/backward + 3 optimizer steps（快速设施检查）
Qwen3.5-9B：
1 forward/backward step
3 optimizer steps
20 optimizer steps
save
resume 3 steps
reload through serving
```

固定要求：

```text
tuner_type=full
freeze_llm=false
freeze_vit=false
freeze_aligner=false
BF16 + ZeRO-3/offload + gradient checkpointing
```

验收必须证明 LLM、vision encoder 和 aligner 都有非空有限梯度，optimizer/checkpoint 覆盖
三者，恢复后的 global step、optimizer 和 scheduler 连续，重新服务结果可解析。OOM、NaN、
零梯度或不可恢复都属于设施失败，不能降级后标记通过。

### 9.4 正式 SFT

只有接受轨迹规模足以进行 source-family 隔离、且各阶段不被极少样本主导时启动。初始采样
比例作为配置保存，不写死在代码；根据 held-out 的感知、Planning、工具选择、Decision 和
停止错误调整。

checkpoint 选择依据是 held-out stage 指标和真实 runtime 行为，不是训练 loss。候选 checkpoint
先跑四条 Qwen canary，再跑冻结 20 例；任何 Prompt/runtime 改动必须单独成为新实验，不能与
权重收益混在一起。

## 10. Agent RL 设施

### 10.1 架构

```text
Qwen policy rollout endpoint
       ^
       | token/logprob gateway
     rLLM
       |
现有 v4 runtime -> tools -> deterministic reducer -> next StageHandoffPacket
       |
terminal canonical trace -> private reward service
       |
     veRL update -> 新 Qwen checkpoint -> 重新采样
```

一条 case 是一个 Episode；每次 Planning、ReAct、Decision 或 Judgment 模型调用是一个 Step。
v4 runtime 继续运行原有工具、归档、状态更新和停止，不翻译成第二套 Gym 状态机。

### 10.2 训练范围

RL 必须从已通过真实 Qwen canary 的 SFT checkpoint 启动，不能从 base 直接做真实工具 RL。
第一版优先训练语言 policy，冻结 vision/aligner；因为多数在线搜索 Step 是文本状态，图像只
在 perception 和显式 reinspection 出现。若视觉错误是主要瓶颈，再单独设计有图像覆盖与
可验证 reward 的多模态 RL，不用稀疏视觉梯度冒充端到端改善。

四卡是否支持 9B full-policy RL 由真实 actor/rollout/optimizer 显存门禁决定。默认先验证
参数高效 policy RL；只有 full-policy 的 save/resume、吞吐和稳定性实测通过，才将其升级为
正式训练配置。所有训练方法必须写进 checkpoint manifest，不能把 LoRA 标成 full。

### 10.3 Reward 与故障处理

主要 reward：

- 最终事实结果；
- 决定性 Evidence 和可追溯 basis；
- Claim/Discrepancy 对齐与冲突处理；
- 合理停止和无 post-determination action；
- 协议有效性与成本。

Gemini 可以冻结地给调查方向、Evidence 使用和过程质量做语义评分，但不能成为唯一 reward。
网络超时、provider 失败、网页不可访问和工具基础设施错误必须 fatal-aware mask，不能作为
Qwen 的负奖励。reward 永不进入模型上下文，所有分量独立记录，先做 replay 人工审计再训练。

### 10.4 RL 阶段

1. **R0 reward replay**：冻结 Gemini/Qwen 轨迹，确认好轨迹稳定高于错误、浪费或协议失败轨迹；
2. **R1 gateway smoke**：mock 工具，验证每个独立 stage call 的 token/logprob/mask；
3. **R2 cached tools**：使用冻结搜索/网页响应，排除网络方差；
4. **R3 controlled live tools**：只在训练 split 开放受控真实工具，限流并缓存；
5. **R4 acceptance**：四条 canary 和冻结 20 例比较 SFT 与 SFT+RL。

每次 policy checkpoint 更新后重新 rollout。Gemini 的固定示范可用于 SFT、reward replay 或
离线比较，不能冒充 on-policy RL 轨迹。

## 11. 实施阶段

### Phase A：冻结基线与清理训练仓库

- 记录两个仓库 commit/dirty state；
- 审计并整理现有 Qwen3-VL 未提交改动；
- 冻结现有四条 Gemini canary 和 v4 canonical contract；
- 添加 GPU 4～7 allowlist、进程所有者检查和 manifest schema。

### Phase B：Qwen3.5 环境和模型资产

- 建立三个独立 Qwen3.5 环境；
- 先下载 Qwen3.5-4B 并记录 revision/SHA-256，完成单卡快速启动；
- 再下载 Qwen3.5-9B 并记录 revision/SHA-256；
- 先在 4B、再在 9B 完成 processor、FLA、forward/backward 和依赖锁；
- 不启动正式训练。

### Phase C：Serving 与协议

- 4B 单卡 BF16 vLLM 快速 smoke；
- 9B 单卡 BF16 vLLM 原样复验；
- 图像、JSON、tool call、continuation、八步短链和 shutdown；
- thinking A/B；
- 生成并验证 serving profile。

### Phase D：v4 Qwen adapter

- 接入 `student-qwen35-local`；
- provider-neutral 单元/集成测试；
- 4B 只运行 mock 与至多一条真实端到端 smoke，不运行完整 20 例；
- 四条 Qwen base canary；
- 修复通用协议/信息传递问题，不加样例规则。

### Phase E：Qwen base 20 例

- 冻结 Prompt、runtime、checkpoint 和 serving profile；
- 一次性运行 20 例；
- 完成逐例 strict audit 和错误分类；
- 不将结果用于训练输入。

### Phase F：SFT 设施

- 数据导出、Qwen processor、loss mask 和 split 门禁；
- 四卡全参数 1/3/20-step、save/resume/reload；
- 满足数据规模门禁后运行正式 SFT；
- 四条 canary 和 20 例复测。

### Phase G：Agent RL 设施

- rLLM/veRL gateway protocol smoke；
- reward replay、mock、cached、live 顺序推进；
- 每次更新重新采样；
- SFT+RL canary 和 20 例复测。

## 12. 完成与暂停条件

部署完成：Qwen3.5-9B 可以稳定处理受控图像、显式状态包和 native tools，四条真实 canary
可审计，无 Gemini fallback，Qwen base 完整 20 例已保存。

SFT 设施完成：四卡全参数多模态训练的梯度、保存、恢复和重新服务全部通过；数据契约和
loss mask 可复核。设施完成不等于模型效果已经改善。

RL 设施完成：现有 v4 runtime 可在不复制状态机的条件下产生 on-policy Qwen rollout，
token/logprob/mask、reward、故障屏蔽、更新和重新采样都可审计。

遇到以下情况立即停止相应阶段并保留证据：

- `4,5,6,7` 中计划使用的卡被其他进程占用；
- 环境升级触及 `ifv-agent`；
- private gold 或 scorer 输出进入模型上下文；
- checkpoint 不可恢复、组件零梯度、NaN/OOM 或 provider 静默 fallback；
- 为单例添加人物、交通工具、URL、关键词或答案专用规则。

# Qwen3-VL-8B-Thinking 部署与训练设施统一执行计划

日期：2026-07-21  
状态：执行中，随 gpu-13 实测动态维护  
Runtime：`visual-fact-discrepancy-agent-v4`  
Training：`image-factual-verifier-training`  
唯一正式学生：`/gsdata/home/wza/models/Qwen3-VL-8B-Thinking`

> 本文是 Qwen 部署、SFT、Agent RL 和 20 例评测的唯一执行计划。模型、数据契约和验收
> 标准已经冻结；具体框架必须经过 gpu-13 实测才能冻结，不因计划中的候选名称而硬撑。

## 1. 已确认决策

1. 全部主线统一为服务器已有的 `Qwen3-VL-8B-Thinking`，不再下载、部署或训练 Qwen3.5。
2. gpu-13 只使用物理 GPU `4,5,6,7` 中实时空闲的卡，最多四张；不占用 0～3，
   不终止其他用户进程。
3. 先通过模型、processor、serving、图像、JSON 和原生工具协议门禁，再跑 Queen、
   Andreea、Pillars、Monarch 四例 canary；四例通过并冻结配置后才运行 Qwen base 20 例。
4. 20 例是冻结开发评测集，不进入 SFT 或 RL 数据，也不向运行时暴露 private gold。
5. SFT 必须是 LLM、视觉编码器和 aligner 均不冻结的全参数多模态训练；LoRA 只能作为
   显式诊断，不能冒充全参数门禁。
6. Agent RL 的 rollout 随 Qwen checkpoint 更新而重新采样。v4 runtime 持有工具、状态、
   Evidence、停止和审计语义；训练框架不得复制第二套 Agent 状态机。
7. Gemini 只作为冻结的离线教师、过程评分器和回归基线，不生成 on-policy RL rollout，
   最终 Qwen 运行不依赖 Gemini。
8. Prompt 只描述任务、可用工具和输出 schema；不堆样例补丁或大量条件规则。
9. 服务器 checkpoint 的 chat template 固定以 `<think>` 开始，`enable_thinking=false`
   不会关闭思考，因此按原生 Thinking 模式运行，并用 serving reasoning parser 将推理与
   canonical action 隔离。隐藏推理不作为 Gemini SFT 监督目标。

## 2. 不变边界

```text
data-pipeline image_only release
  -> v4 runtime（环境与证据所有者）
       -> teacher-gemini（离线示范 / 冻结评分 / 基线）
       -> student-qwen3-vl-local（base / SFT / RL / 最终运行）

训练数据：provider-neutral stage rows + 显式 workspace/archive + tool schema
训练目标：单阶段可见 policy output；工具结果和确定性状态默认 loss mask = 0
评测顺序：base -> SFT -> RL，三者使用同一冻结 20 例和同一审计器
```

Runtime 和 Training 仓库只通过版本化 JSONL、manifest 和 checkpoint profile 交互，
不互相 import Python package。Gemini Interactions 与 Qwen OpenAI-compatible endpoint 是
不同 wire protocol，只在 canonical stage input/output、tool schema 和 trace contract 上对齐。

## 3. gpu-13 已知事实

- 完整权重：`/gsdata/home/wza/models/Qwen3-VL-8B-Thinking`，约 17 GB，4 个完整
  safetensors 分片，架构为 `Qwen3VLForConditionalGeneration`。
- 现有环境：`/gs/home/wza/anaconda3/envs/qwen3vl`。
- 已实测基础栈：Python 3.10.18、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、
  Transformers 4.57.6、qwen-vl-utils 0.0.14、LMDeploy 0.13.0、Ray 2.55.1。
- LMDeploy 0.13.0 已成功加载该权重并完成约 530 次 Chat Completions 请求；历史终止原因是
  节点系统内存触发 Ray 95% 阈值，不是模型协议失败。
- vLLM 0.8.5 已因 `Qwen3VLConfig.vocab_size` 兼容问题失败；现有 vLLM 0.10.2 只能作为
  新实测候选，不是默认 serving 主线。
- gpu-13 直连仍不可用，但正式代理 `http://100.10.1.210:47899` 已于 2026-07-21
  复测恢复（PyPI HTTP 200）。旧 shell 中残留的 `47894` 必须清除；权重已在本地，不重复下载。

每次 GPU 任务前重新记录 `nvidia-smi` 的物理卡、显存、利用率、PID、用户和运行时长。
所有任务设置 `OMP_NUM_THREADS=1`，保存 run ID、Git commit、GPU 映射、环境 manifest、
stdout/stderr、停止命令和输出路径。模型、数据、checkpoint、rollout 和日志写入 `/gsdata`，
不写入 Git checkout。

## 4. 动态框架门禁

框架选择遵循“先实测，后冻结”。切换框架时只替换执行层，不改变数据契约和验收标准。

| 层 | 首选候选 | 通过条件 | 不通过时的动作 |
|---|---|---|---|
| Serving | LMDeploy 0.13.0 | 图像、JSON、reasoning 隔离、tool call、tool continuation、8 轮短链、并发与干净退出均通过 | 已通过加载、图像、JSON 和 512-token native tool call；继续补 tool continuation/8 轮/生命周期实测，失败再切候选 |
| SFT | ms-swift 4.4.1 + DeepSpeed ZeRO-3 | 精确 processor/template、全参数梯度、1/3-step、保存、恢复和重新加载均通过 | 优先调整兼容版本或 ms-swift 后端；仍失败才选支持 Qwen3-VL 的成熟 Transformers/DeepSpeed 方案 |
| Agent RL | rLLM gateway + veRL | 能由 v4 runtime 驱动 on-policy rollout，保留 tool provenance、logprob、mask、恢复和审计 | rLLM 主线依赖独立的新 PyTorch/Transformers/vLLM 栈；与 serving/SFT 隔离，先验证 rollout worker 的 token ID/logprob 对齐，不通过则比较成熟替代框架 |

框架失败必须留下最小复现、环境 manifest、日志和失败分类。只有模型本身、数据契约或验收
目标失败才需要修改上层计划；框架 API/ABI 不兼容只触发成熟替代框架选择。

## 5. Runtime 接入

活动 profile：

```text
teacher-gemini
student-qwen3-vl-local
```

默认 served model 为 `ifv-qwen3-vl-8b-thinking`。`qwen_local` provider 和
`QWEN_LOCAL_*` 环境变量保持 provider-neutral 命名，不保留 Qwen3.5 的模型默认值或特例。

Qwen adapter 必须支持：

- 受控单图输入和图像尺寸/token 记账；
- native `tools` / `tool_choice`；
- assistant `tool_calls` 到 `role=tool` JSON continuation；
- JSON schema response format 与同请求格式纠正短链；
- reasoning/thinking 与 canonical action 隔离；
- provider-neutral `policy_input` / `policy_action` 导出；
- 训练快照不复制 image data URL，只保留可校验资产引用。

Planning、Decision、Reflection、Judgment 是独立请求；一次 function call、result、output 只形成
一条短链。跨阶段信息由显式 `StageHandoffPacket`、workspace 和按 ID 回读 archive 传递，
不能继承旧会话，也不能因短链化丢失决定性 Evidence 或图像理解更新。

## 6. Serving 验收

初始配置：BF16、loopback、单张空闲 GPU、`max_model_len=32768`、Thinking reasoning parser。
只有真实状态包需要且显存、
系统内存与延迟均可接受时才提高到 65536；128k 是审计边界，不是日常输入目标。

依次验证：

1. `/health` 与纯文本 Chat Completions；
2. 单张受控图像，记录原始尺寸、发送尺寸、编码字节和估算 image token；
3. 确定性 JSON/schema 输出；
4. 正好一次 native tool call；
5. `role=tool` continuation；
6. 八次短链，每次显式重建阶段状态；
7. Planning/Decision/Judgment 不继承旧 session；
8. 超时、取消、并发、服务失败与干净 shutdown；
9. 普通 Agent serving 与开放 token IDs/logprobs 的 RL rollout profile 分离；
10. 记录 RSS、GPU memory、executor 状态，避免再次触发系统内存阈值。

任何 provider/protocol/runtime 失败都输出 `engineering_error`，不得静默切换 Gemini 或猜测
`real|fake`。

2026-07-21 第一轮实测：LMDeploy PyTorch/`uni` 在 GPU 4 成功加载，约占 33.9 GB；图像和
JSON Schema 通过。128 output token 会在 Thinking 中截断 native tool call，512 token 能返回
正确 `tool_calls`，因此门禁预算不得误设为 128。开启 `raw_logprobs` 后能返回生成 token IDs，
但标准 logprobs 仍为空；LMDeploy 当前只冻结为普通 Agent serving，不冒充 RL rollout worker。

## 7. 数据与上下文

训练集只能来自通过 schema、provenance、图像资产、阶段归属、loss mask、去重和泄漏审计的
accepted teacher release。20 例 case ID、图像哈希及其派生轨迹必须加入拒绝清单。

一条 SFT 行对应一次真实阶段决策，而非数十轮完整聊天：

```text
immutable brief
+ protected workspace
+ selected archive spans
+ current image/crop（仅该阶段需要时）
+ tools
-> one canonical Planning / ReAct / Decision / Judgment output
```

保留决定性证据、未解决冲突、当前调查方向、路线状态、图像理解修订和精确 archive 引用；
压缩重复网页、旧模型措辞、无效工具输出和已被新状态替代的历史。任何正常请求目标不超过
128k；超限样本拒绝进入训练并生成可审计报告，不做静默截断。

## 8. 全参数多模态 SFT

活动模型 profile 仅使用 `qwen3-vl-8b-thinking.env`。训练 profile 分为 1、3、20 optimizer
step；三者均要求：

```text
tuner_type=full
freeze_llm=false
freeze_vit=false
freeze_aligner=false
DeepSpeed ZeRO-3
BF16
```

执行门禁：

1. processor/chat template 能处理图像、tool schema 和阶段输出，token/loss mask 精确；
2. 非 20 例合成小数据完成一次 forward/backward，三部分参数均出现有限、非零梯度；
3. 1-step 保存完整 checkpoint、optimizer、scheduler、RNG 和 manifest；
4. 从 1-step 显式恢复并继续到 3-step，验证 global step、样本顺序和参数变化；
5. 独立进程重新加载 checkpoint，重新运行文本、图像、JSON 和 tool smoke；
6. 20-step 只用于设施稳定性，不代表模型效果；通过后才接受真实 teacher dataset。

不能因为显存不足静默冻结视觉塔、改 LoRA、量化或丢图。应先调整 activation checkpoint、
ZeRO-3 offload、序列/图像预算或等待四卡空闲，并记录吞吐与内存代价。

## 9. 四例 canary 与 20 例

四例顺序固定为 Queen、Andreea、Pillars、Monarch。每例保存完整 canonical trace、原生工具
记录、Evidence、图像理解修订、停止原因、token、图片发送记录和 strict audit。

四例门禁不要求 base 全部判对才允许研究继续，但要求：运行无工程错误；工具协议完整；
调查方向不是被 prompt 硬绑；搜到的新材料能更新图像理解；裁决只消费合格 Evidence；失败可
归因为模型、检索、抽取、上下文或控制流。完成诊断并冻结模型、engine、thinking、context、
image 和 prompt 配置后，才运行完整 20 例。

20 例运行中不得调 prompt、预算、engine 或逐例修补。base 完成后固化评测产物；SFT 和 RL
checkpoint 使用同一配置重跑，比较 verdict、证据质量、轨迹方向、成本、延迟和工程错误率。

## 10. Agent RL

共享契约：SFT 与 RL 使用相同的 stage input/output、workspace/archive/action/gain/verdict、
tool provenance 和工程错误语义。Qwen checkpoint 是 rollout policy；每次更新后重新采样。

奖励分层：

- 硬门禁：schema、工具调用合法性、来源访问、Evidence provenance、预算与 private-gold 隔离；
- 过程分：信息增益、证据利用、图像理解修订、路线切换、无效重复和成本；
- 结果分：rollout 结束后读取 private gold 计算 binary verdict 与校准；
- 冻结 Gemini 可给语义过程评分，但不得覆盖硬门禁或直接操纵环境状态。

先运行 mock tool gym 验证多轮、取消、异常、reward 回传、logprob/mask、checkpoint 与恢复；
再运行少量真实工具 rollout。任何训练框架都必须通过 v4 gateway 调用环境。

## 11. 实施顺序与完成定义

### Phase A：本地统一

- 删除活动 Qwen3.5 profile、依赖、launcher、测试和文档引用；
- Runtime 改为 `student-qwen3-vl-local`；
- Training 改为 Qwen3-VL model、LMDeploy serving、全参数 1/3/20-step SFT 和 RL profiles；
- 两仓库通过 pytest、compileall、diff check 和 Bash 语法门禁后提交。

### Phase B：服务器协议门禁

- Git bundle 安全 fast-forward 两仓库；
- 实时选择 GPU 4～7 中空闲卡；
- 对现成环境做 environment/model manifest；
- 启动并验证 LMDeploy，失败时按第 4 节动态选择成熟替代框架；
- 通过文本、图像、JSON、tools、continuation、8 轮短链和生命周期门禁。

### Phase C：训练门禁

- processor/template probe；
- 全参数 forward/backward、1-step、恢复到 3-step、重新加载；
- 20-step 设施检查；
- mock Agent RL gateway/rollout/reward/resume 门禁；
- 将真实通过的框架和精确版本写入 lock manifest 与本文。

### Phase D：真实评测

- 四例 Qwen base canary；
- 修复通用工程或 prompt 问题后冻结配置；
- Qwen base 运行完整 20 例并汇总；
- SFT、RL checkpoint 沿同一冻结协议复评。

训练基建完成需同时满足：Qwen3.5 活动入口清零；Qwen3-VL runtime adapter 和 serving 协议
真实通过；全参数 SFT 可保存、恢复和重新加载；Agent RL gateway 可运行可审计的 mock rollout；
四例 canary 产物完整；框架实测决策、环境 manifest、命令、日志和失败记录均可复现。

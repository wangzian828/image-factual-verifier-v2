# PSD 实机验收续记（2026-09-16）

本记录优先于较早的 v12/v14 状态段。用户要求当前任务持续修复；定时任务保持暂停。
**正式 400×8 源采集尚未启动，正式 PSD optimizer step 为 0。**
以下更新、checkpoint 和压测都是独立工程验收，不是正式训练结果。
原三轮 SFT 模型和完整 checkpoint 保持只读，冻结 Agent `src` 未改。

## 已确认并修复的训练问题

### 1. 语法屏蔽后的概率不能当成原始 teacher 分布

vLLM 0.18.1 先执行 grammar bitmask，再进入 Sampler 的 `raw_logprobs`。
真实请求有些位置的 top20 为 `[0, -9999, ...]`；仅检查有限值不能识别这种
已经屏蔽、近似 one-hot 的分布。它与原方法的 forced-token 原模型评分不同。
上游参考：[River top-k 收集器](https://github.com/essamsleiman/psd/blob/778be78bdac582b51a975ff819046583aad383e0/psd/targets/river_topk.py)。
此处只参考评分位置/分布语义，配比仍以 Qwen3.5 发表配置为准，不采用 River 的 1:1 配比。

修复边界：

- 独立 RawTeacherWorker 保存语法屏蔽前 logits；原 sampler 仍按原工具约束、温度、预算选 token。
  只替换返回的 top-k/采样 token logprob，不改变采样结果或屏蔽规则。
- 无 grammar 的请求使用原实现；有 grammar 时的原始 logits 只消费一次，异常清理，禁止跨 batch 复用。
- 新 metadata `pre_grammar_unprocessed_v1` 从受验证的服务，经 PSD-only capture 适配器、候选、目标传到 datum。
  网关只有在所有实际路由的 worker 命令、PID、代码哈希匹配时才标注；混用旧副本不得标注。
- 未带来源的旧 capture 保留 token/轨迹，但 top20 变成 pending；强制按原 teacher prompt、原 completion、原图片补算。
  不删除训练样本，不重采调查，不把旧值换个标签继续用。新 datum/preflight 拒绝无来源的旧“complete”包。

实测：四张 H20、每卡 10 并发、3 波共 **120/120 原始真实请求成功**，非有限值 0，`-9999` 哨兵值 0。
请求不执行工具，不当作训练数据。此候选还显式关闭 async scheduling，仍保留 128K、图片、think8192、out32768。
它不是“NaN 永久消失”的证明；更早 eager 通过后仍曾复现。完整真实 Agent 仍需继续验证。

### 2. 原生 FSDP LoRA 保存成功，但恢复包装错误

四卡原生入口在旧 199-target 工程银行连续执行两步并成功保存完整 DCP，每个约
1,248,153,248 bytes（约 1.16 GiB），包含模型、optimizer、scheduler、四 rank RNG。
该银行现已知道使用了未绑定语义的旧 top20，**只能证明机械训练/存储路径，不是合格 PSD 训练目标**。

真实恢复失败：ms-swift 4.4.2 在缺少 PEFT `adapter_config.json` 的 FSDP DCP 目录调用
`Swift.from_pretrained`，得到零可训练参数的 SwiftModel，然后 Accelerate 的 `assign=True` 加载失败。
未执行错误优化步骤。仅为 `load_state_dict` 加一个参数会掩盖 LoRA 根本没建起来的问题。

新增 PSD-only bridge：对已通过 resume 绑定的 FSDP2 LoRA，先按原配方重建 PEFT 结构，随后让原生 Trainer
加载 DCP 参数和 optimizer/scheduler/RNG。恢复参数在结构构建结束后原样还原；拒绝零 trainable 参数或非 LoRA
trainable 参数，不影响其他 SFT/普通 PEFT 路径。**四卡实际恢复对照现已通过（见下）。**

## 已验证的训练数值边界

真实 9B、完整图片和 repair/preserve 两个目标的单卡 update/save/resume：

- 默认 FlashAttention 反向：恢复参数最大差 `6.888226e-5`，严格门槛未通过。
- `FLASH_ATTENTION_DETERMINISTIC=1`：恢复 loss、梯度范数和 adapter 参数逐位相同，参数最大差 0。
  没有放宽容差。该设置已加入 DP4/SP4 PSD 配方和 resume 身份。
- 单卡通过不能替代四卡 FSDP 原生恢复验收。

## 当前产物和验收边界

- 已审核真实工程银行：1 repair + 198 preservation targets，最长输入 49,546 token，完整图片和 token 无删减。
- 旧银行：`runs/psd-sft3084-captured-canary4x8-20260916/psd-grounded-review-canary-v15`。
- 新独立银行：同采集根下 `psd-raw-teacher-canary-v18`，199/199 已按原 token/图片重新评分，0 样本拒绝，数据 gate 通过。
- 模型服务：`inference/psd-sft3084-20260916/raw-teacher-noasync-v1`；逐卡当前 PID 必须读取服务根 `replica-*.json`。
- 40 并发证据：服务根 `nan-raw-teacher-concurrency40-v1/summary.json`。
- 全量修复候选代码：`training-artifacts/psd-capture-callsite-20260916-v21/code`。
- `psd-fsdp-resume-bridge-20260916-v19` 仅用于同旧银行的机械恢复对照，不能启动正式采样/训练。
- v18 的 434 项训练测试通过；安装的真实 vLLM 环境下，50 项网关/协议/概率检查通过。
  v20 新增 4 项 FSDP bridge 测试。真实恢复及新银行 GPU 更新不得用这些单测替代。

### 后续实机结果（19:53 CST 更新）

- 新 raw 银行单卡真实更新通过：loss `169.9499359 -> 162.9988403`，视觉分支梯度非零；
  第二步原地恢复与连续执行的 loss、梯度范数、所有 adapter 参数逐位相同，最大参数差 0。
  证据：新银行 `gpu-update-gate-raw-v1/probe/result.json`。这是完整两条真实目标，不是全局 batch32 验收。
- 旧四卡 DCP checkpoint 的 PEFT 导出通过：716 张量、102,530,048 adapter 元素；原 checkpoint 未改。
- v21 修复 Python 导入绑定问题：`stage_runner` 提前 `from ... import extract_policy_token_capture`，
  只替换 llm_backend 定义处不能保证覆盖 collector/repair 的实际调用处。现同步替换该 PSD 进程的已导入别名；
  冻结 `src` 不改。真实 import 顺序检查通过，442 项训练测试通过。
- **完整 Agent runtime gate 失败，不能放行正式源采集。** 两条固定工程轨迹的前四个请求正常，
  第五个请求都含 5 张历史图片，返回 `Out of range float values are not JSON compliant: nan`。
  原始 runtime/错误保留于 `runs/psd-raw-runtime-gate-20260916-v1`，不加入训练目标。
- 从已落盘 started-event/artifact 重建第五请求，未更改原轨迹；诊断副本位于服务根
  `raw-multimage-failure-v1`。它不是原 wire 字节副本，不能声称 nonce/序列化逐字节相同。
  对两个重建请求各测 3 次首 token（仅此诊断缩为 1 输出 token 并移除未来 think-budget closure），
  6/6 有限，不能当作完整请求或稳定性的通过证据。正式预算仍为 think8192/out32768。
- 四卡 native resume 对照已通过：旧工程银行 `dp4-resume-gate-v2/result.json`，
  从 checkpoint-1 原生加载模型、optimizer、scheduler、RNG 后执行第二步，导出的全部 adapter
  与连续运行 checkpoint-2 逐位相同，最大参数差 0。仅验证机械恢复，绝不冒充正式训练。
- 原四卡服务已由恢复控制器重新启动，保活恢复。新增单卡诊断 `nan-boundary-v1`：
  GPU2 使用独立 JIT 缓存、prefill 层级有限值记录和每步原始 logits 检查；不修改数值、不重试掩盖错误。
  两条重建请求仍保留完整 5 图、32768 输出和 8192 思考预算。诊断会影响时序，不能把通过当成根因修复。

新的定位线索包括动态 kernel/状态复用。官方 [vLLM #52413](https://github.com/vllm-project/vllm/issues/52413)
报告过共享 Triton 编译缓存与首步 NaN，但模型/版本/硬件不同；尚未证明我们具有相同根因，不据此声称修复。

### 参考原正常评测服务（20:20 CST）

用户指出同一权重在测试集推理中基本正常。重新核对原 `sft3084-3epoch-20260915`：
模型导出目录相同，评测最终在既定补跑后 1,526/1,526 成功；四份服务日志未检出
`Out of range float values`、NaN/nan 或 token0 FSM 拒绝模式。这不是首遍零错误证明。
原启动日志确认 vLLM0.18.1、BF16、128K、图32、seq8、batched32768、mem.94、
qwen3_coder、compile+CUDA Graph、async enabled。PSD 后加的 custom worker、预算插件、
single-tool parser 和 seq16 不是原正常评测配方。

- `nan-boundary-v1` 六次重建完整请求全部正常返回工具调用、无非有限 logprob；这组改变了
  时序和 JIT 缓存，仍不能宣称 NaN 已修复。原 GPU2 服务已恢复，其他三卡未动。
- 新 `eval-reference-ab-v1` 只租用 GPU2，恢复原评测的执行设置，三组分别普通输出、
  仅原生 token IDs、原生 IDs+stock top20。每组四并发，共十二请求；两条固定输入不变。
  为可比性仍用失败 PSD 请求的 T0.7/seed/全部图片/工具/32768 输出，故不是重现旧测试题。
  请求的 thinking_token_budget8192 仅复现原 API 字段，旧服务并不落实该预算；不得作为正式配方。
- 诊断的 stock top20 可能已被 grammar 屏蔽，只用于观察故障，绝不作 raw teacher 目标。
  无工具执行、不产生训练样本。结束后自动恢复原 worker/guard。
- `eval-reference-ab-v1` 已完成：普通输出、仅 token IDs、IDs+stock top20 各 4/4，合计
  12/12 正常单工具输出，返回概率未见非有限值。冷启动和热请求不可混作速度对比。
  这不是完整 Agent 验收，也没有证明 NaN 根因已修复。
- `eval-style-agent-v1` 已启动两条完整 Agent 隔离验收：采用原评测 compile/CUDA Graph、
  async、seq8 和 stock worker；仍保留实际生效的 think8192、单工具约束、禁用 MM processor
  cache。使用原来失败的两条固定 case、T0.7、全部图片和 out32768；不把 stock top20 当 teacher。
  验收结束恢复原 GPU2 服务，其他卡保持现有服务/计算保活；正式源采集尚未开始。
- 该完整 Agent 验收现已完成：两条均 `termination=success`，严格审计无拒绝/警告；
  分别 20/18 个完整 token capture，非有限概率 0，最长思考 622/483，最长 prompt 49,587/26,124。
  未标记为 raw teacher；后续必须独立评分。该结果只代表这两条，不代表并发稳定性通过。
- 随后对全部 38 个 capture 做原生 AutoProcessor 图片重建检查：从对应 archived request
  按顺序恢复图片，逐段 image-token run 与 image_grid_thw/merge_size 完全匹配，像素均有限。
  两条轨迹单次请求最多分别带 20/18 张图片；此次 CPU 检查未产生新的大 tensor 缓存或使用 GPU。
- 可选的正式简化路径是保留精确 token、将概率留待同一个冻结模型 forced-token 补算，
  不删 top20、图片或目标。原仓库 `psd/targets/river_topk.py` 本身分离生成和评分；这里只
  参考精确评分方式，不引入该 River 实验的模型/1:1权重等不同设置。
  是否采用仍需三组对照及完整 Agent token 链路验收，尚未切换正式协议。

### 框架原生实现复核（用户要求，2026-09-16）

以服务器实际 vLLM 0.18.1、Transformers 5.12.1、ms-swift 4.4.2 源码为准，
不把新版本文档中的能力假定为已安装版本具备，也不为核对接口更换环境。

1. **9B 原实现已经分离采样和 teacher 评分。** 上游固定提交 `778be78` 的
   [`psd/training/turn_kl.py`](https://github.com/essamsleiman/psd/blob/778be78bdac582b51a975ff819046583aad383e0/psd/training/turn_kl.py#L950)
   中 `teacher_topk_for_completion_ids` 用 Tinker 原生 `sample_async`，输入原
   `teacher_prompt_ids + completion_ids`，请求 `include_prompt_logprobs=True`、
   `topk_prompt_logprobs=20`。额外生成的 1 token 不作训练目标；只取既定 completion 各位置的分布。
   因此 deferred scoring 不是削减 PSD 监督的折中，也不要求把采样 logits 内核改写。
   Tinker 是上游托管后端；本任务不擅自上传数据或迁移训练到该服务。
2. **vLLM 原生接口足以承担采样和精确 token 返回。** ChatCompletionRequest 已定义
   `return_token_ids`、`return_tokens_as_token_ids` 和 `prompt_logprobs`；
   `GPUModelRunner._get_prompt_logprobs_dict` 单独从 prompt hidden states 算 logits/logprobs，
   不走生成 token 的 grammar bitmask。反之，生成 token 的 `raw_logprobs` 仍在外部 grammar
   bitmask 之后，不能因为名字有 raw 就认定是原始 teacher 分布。
3. **不能直接用本版本 token-only HTTP 接口给多图评分。**
   [`MultiModalFeatures`](https://docs.vllm.ai/en/v0.18.1/api/vllm/entrypoints/serve/disagg/protocol/#vllm.entrypoints.serve.disagg.protocol.MultiModalFeatures)
   明确是 metadata-only；实际 `ServingTokens.serve_tokens` 只预处理 `request.token_ids`，
   未消费图片特征。普通 CompletionRequest 也没有多模态输入字段。离线 `TokensPrompt`
   支持 `multi_modal_data`，但会执行 MM processor 占位符更新；已经展开的原 token IDs 不能未经
   一致性验证就再次送入。暂不新增私有 endpoint、不去掉图片、不 decode/re-encode 绕过此限制。
4. **已验证的多图替代路径是 Transformers 原生 forward。** 当前 `FrozenTeacher` 只组装
   `input_ids`、经校验的原 `pixel_values/image_grid_thw`、`mm_token_type_ids` 和
   `logits_to_keep`，调用原生 Qwen3.5 模型的 `forward(use_cache=False)`。
   精确 completion/token/图片无删减，原始 top20 后按上游归一化。199/199 完整目标评分和真实
   GPU update/resume 已通过；不是每次评分重跑 Agent，也不采一个更容易的答案代替原 completion。
5. **ms-swift GKD 不能直接当作相同 PSD loss。** 已安装 `gkd_loss.py` 的 top-k 分支先
   gather 学生的 K 个 logits，再在 K 内 log-softmax；`GKDTrainer._compute_jsd_loss` 按有效
   token 数平均。我们的上游目标是 teacher top20 归一化、学生全词表 softmax、token loss 求和，
   不能仅设置 `topk=20` 就声称等价。继续复用 Trainer/FSDP/optimizer/checkpoint 的原生实现，
   仅保留与原 PSD 目标一致的 loss/数据适配，不把配方替换成另一种蒸馏算法。

当前选择：优先验证 **stock vLLM worker 采样 + 原生 Transformers 独立评分**。
RawTeacherWorker 留作历史诊断候选，不因已写过就作为正式路径的必要组件。
思考预算和单工具约束仍通过 vLLM 官方扩展接口加载；它们不是 teacher logits 钩子，不能为了
称作“纯原生”而退回实际无效的 8192 字段或改变 Agent 行为。现阶段不宣称所有自定义适配已移除。

### 四卡 stock worker 压测发现的剩余故障（20:51 CST）

- `deferred-teacher-four-gpu-v1` 四卡部署完成，网关 19025，stock worker、compile/graphs/async、
  seq8，实际 think8192、out32768、128K、32图均保留；每卡独立 JIT cache。
  没有改 vLLM 安装包，也没有升级环境。旧模型/export/checkpoint 未写入。
- `deferred-concurrency40-v1` 预定三波 120 请求，但**首波 40 只通过 39 个**；后两波未执行。
  GPU3 slot39 连续产生 token0 FSM/grammar rejection；这条在完整返回前被终止，
  **只能确认异常输出，不能把这一条的 NaN 当作已直接测得**。
- 原 client PID 经 command/PGID 身份验证后停止；框架随后释放该请求，GPU3 running/waiting 回到 0。
  39 个正常结果和 1 个明确的基础设施失败分别保留，没有把未完成的诊断当成功，也没有有利重采。
  证据：`infrastructure-failure.json`、各 slot result、服务 GPU3 日志。正式采集和训练均未开始。
- 因此不能归因于 RawTeacherWorker，也不能说“恢复原推理设置已修复”。单卡完整 Agent 成功与
  四卡并发故障并存。保留生成/评分分离方向，但当前 serving 候选**不具备正式放行资格**。
- 框架源码已核实：`VLLM_COMPUTE_NANS_IN_LOGITS` 是原生观察功能，不自动终止坏请求；
  scheduler 对 grammar 拒绝只打印 warning。`/abort_requests` 本版本仅在 tokens-only 服务注册，
  当前 Chat 服务没有该路由；本次通过结束该诊断 client 触发框架取消，没有终止共享模型服务。
- 上游 [#51562](https://github.com/vllm-project/vllm/issues/51562) 和
  [#53059](https://github.com/vllm-project/vllm/pull/53059) 分别讨论 GDN 无初态短 prefill
  被归为 decode、以及 shape-only uniform-decode 图调度。已安装代码存在相似的 query-length/
  shape 判断，但本次没有记录到触发请求的调度元数据，**不能认定就是这两个问题，尚未移植补丁**。
  shared JIT cache 不再是本次四副本之间的共享变量；每副本独立 cache 仍不能排除其他 JIT/kernel 问题。

所有上述服务器路径均以 `/volume/ybo/wza/` 为根，不在服务器提交 Git。
早期 FlashInfer 诊断曾触及其默认 `/root/.cache`，已告知用户并停止该分支；未删除或改动该根外目录。
后续显式指定所有相关缓存在允许根内。不能把共享磁盘 `df` 当作个人配额。

## 尚需现场完成

### 已增加的整轨迹基础设施恢复（用户授权，2026-09-16）

- 新 PSD 代码快照：`training-artifacts/psd-infrastructure-retry-20260916-v23/code`。
  冻结 `src/` 逐文件保持不变；不重启四卡服务，不修改受保护 SFT 权重。
- 原图、采样槽/seed、模型和配置不变。首次加最多两次完整重跑，5/10 秒退避；
  只处理模型边界明确标记的数值或传输故障。普通答错、格式/契约错误、长度耗尽不重采。
  修复阶段保留同一 hint slate，基础设施重跑不消耗下一轮提示/语义反馈预算。
- 每次尝试独立归档，非有限概率响应在进入工具执行前隔离。采样槽耗尽仍保留 error 行，
  不发布坏的 canonical trace，不减少 400×8 分母，并阻止未补齐数据进入目标构建。
  Gateway POST retries 仍为 0，独立 frozen teacher 评分和原有 strict/semantic 门槛不变。
- 重试状态带身份和 SHA 校验、单槽互斥锁；重启不能重置次数。未知错误、进程中断和
  不确定的在途尝试暂停检查，不假定是基础设施故障。不会把一般模型失败重试到成功。
- 服务器 Python 3.12 完整训练测试 472 项通过；本地 Python 3.9 的 66 项定向测试通过，
  其中额外覆盖真实 slate 搜索的“同一提示恢复、不增加提案次数”。本地完整套件受现有
  Python 3.10+ 类型表达式限制，已在实际服务器环境运行，而不是声称本地也全通过。
- 真实验收位置：`runs/psd-infrastructure-retry-gate-20260916-v1`。两条完整 Agent，
  在第一条第 2 次真实模型请求返回后**人工注入一次选中 token 的 NaN**；明确作为故障注入，
  不计自然失败率，整组永不进入 PSD 训练。验证整条隔离、新上下文恢复及最终严格审计。
  这验证重试链路，不代表底层 NaN/grammar 问题消失，也不自动证明 40 并发的长期可靠性。
- **真实验收已通过**：2 个采样槽、3 次尝试（1 次明确的人工故障注入，随后自动恢复），
  两条均 `termination=success`、strict audit 通过；最终分别保留 17/15 个完整 token capture。
  失败样本只在隔离目录，canonical traces 只有两条最终结果。
  `state.json.phase=full_agent_retry_gate_passed`；正式 400×8 源采集和 PSD optimizer 尚未启动。

1. 199 个旧目标的补算、数据验证和单卡真实 GPU 更新已通过。
2. 修复后的四卡 native DCP 恢复已通过同 horizon 第二步逐位权重比较。
3. 普通采样 + 独立评分的完整 Agent capture 已通过两条，但四卡 40 并发仍失败 1/40；
   现按用户授权增加整轨迹自动恢复；并发运行仍需记录恢复率、耗尽槽和故障相关性，
   不能将一次故障注入通过解释为底层数值问题已修复。
4. 门槛通过后启动已授权 400×8、T0.7、并发40的正式源采集、全部目标构建、5 epoch PSD。
   大规模时关闭重复 raw wire 诊断归档，保留必须的 native token/top20/工具轨迹，避免额外空间及预留预算瓶颈。

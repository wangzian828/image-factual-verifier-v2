# PSD 后续执行授权与交接

## 2026-09-16 11:42 真实runtime和32条源checker均通过技术验收

- `runtime-acceptance-v5.json`：固定4图×8=32槽，448个原生决策捕获完整，59次感知/OCR均cache_hit=false；546个网关请求中536响应和10个保留的感知辅助HTTP中断，422个required调用原始数组均为单调用，最大输出2926token，无截断。runtime gate通过，full PSD gate仍未通过。
- 源checker首次32条完成，29个有效判定、3个非逐字引用被拒绝：一例将原文70年错引为10年，另两例把概述作为逐字引用。没有放松literal校验或丢弃这些轨迹。
- `psd_source_review.py`新增**每个无效引用结果最多一次纠错**，两次原始响应和各自packet/prompt/schema/response哈希均保留；只有首轮因非逐字引用无效时允许纠错，已有效的pass/fail/unresolved都不得重采。接受首个完整有效判定，不偏向pass；第二次仍无效则保持pending，恢复也不产生第三次请求。纠错只在私有checker内，不向Agent/proposer传递任何gold或私有解释。
- 实际三条纠错全部通过：原两个fail仍为fail，原一个pass仍为pass。最终源判定 **12 pass、20 fail、0 unresolved、0 pending**。这是训练源任务验收而非测试指标；20条失败是修复来源，按原分组规则每任务选最长失败源，不代表重跑20×6。
- 当前控制器：`/volume/ybo/wza/training-artifacts/psd-epoch3-repair-20260916-v6/resume_psd_epoch3_canary.py`（PID1037237仅作线索，以process/state/log核验）。它复用同一个`RUN/psd-round-v5`输出，保留29个有效review和全部32个原响应，新的CODE快照只替换源checker纠错模块。v5控制器已完成并暂停，勿重启或新采源轨迹。
- 当前继续后处理和多位置修复；仍使用`--defer-topk`，没有正式GPU优化。后续GPU top20、真实更新/最长target、native完整保存恢复和400×8全量目标/五epoch训练仍须完成。
- 本地100项定向测试通过；最新v6远端既有完整training/tests **389项通过**。原三epoch导出和完整checkpoint未改，新增训练恢复探针仍待真实GPU执行。

## 2026-09-16 11:27 固定32槽已完成

32/32轨迹完成且无顶层错误，`strict-audit-v4.json`通过。审核脚本起初只把`perceive_scene`识别为辅助VLM请求，实际还有冻结的裁剪、关系、异常检测、参考图比较工具；辅助调用同样使用该本地端点，但不是需要policy logprobs的Agent决策。因此v4审计停止，尚未调源checker或启动repair，没有重采任何槽。

继续入口已改为`/volume/ybo/wza/training-artifacts/psd-epoch3-repair-20260916-v5`（process/state/run.log），输出新RUN/`psd-round-v5`。v3/v4和所有证据保留。新审计依据冻结`OpenAIVisionClient`实际的双消息、带图片、temperature0、禁思考、JSON schema请求结构区分辅助响应；原生Agent每个决策仍须完整top20 capture。只有**确认为感知工具的已知HTTP中断**可进入逐条单独核对，其他辅助错误和所有native/capture失败仍暂停。

比较重试请求只排除网关生成、格式已验证的`cache_salt=ifv-psd-isolated-<32hex>`，其他全部生成字段（模型、图片、prompt、schema、温度等）保持匹配。对应成功请求仅是旁证，不把它冒称为同一次逻辑重试。原始wire包含完整salt和哈希，不改写。此前“相同完整请求”应理解为“除隔离nonce外相同生成请求”。新增wire审核测试15项通过；正式目标构建和GPU训练仍未完成。

## 2026-09-16 11:20 实际执行更新（原始32槽诊断不混入新bank）

- 用户已明确选定 **400张×8**，授权完成全部目标构建和首轮五个优化epoch；原epoch3参数及全状态checkpoint保留，PSD独立LoRA输出。
- 初始epoch3 canary发现启动环境缺少`IFV_CAPTURE_POLICY_TOKENS`/`IFV_POLICY_TOPK`，不能充当精确token训练bank；已按PID/完整命令核对后停止，原产物及`capture-preflight-hold.json`保留。没有用网关观测记录冒充原生policy capture。
- 新采集位置：`/volume/ybo/wza/runs/psd-sft3084-captured-canary4x8-20260916`；代码快照`/volume/ybo/wza/training-artifacts/psd-epoch3-capture-20260916-v2/code`，collector receipt在新RUN/process.json。相同预选4图×8槽/T0.7，不按结果换样。
- 新网关19018，`SERVICE/captured-v2/gateway.json`；GPU1仍为原三epoch模型。native capture已在真实轨迹中确认完整prompt/completion token IDs、逐token logprob及top20，启动前缺失开关现在直接拒绝运行。
- 独立继续控制器：`/volume/ybo/wza/training-artifacts/psd-epoch3-repair-20260916-v4/continue_psd_epoch3_canary.py`；`state.json`/`process.json`/`run.log`为实况。只替换尚在等待的v3控制器，不停止collector。固定32槽结束后自动strict audit、源checker和最多6次完整多位置repair，输出新RUN/`psd-round-v4`。`--defer-topk`在GPU交接前停止HF打分，不盲目在满显存上加载teacher。
- 修复提案真正执行12次预算；无效/不合规的完成提案有缓存和计数，给proposer的反馈仅通用原因，不含gold或私有checker解释。有效提案在生成前持久化，断点不重复提案/重采已完成修复。
- 11:15抽查已完成25/32完整轨迹，均无顶层错误。wire中10次中断均是感知辅助请求，存在相同完整请求的成功响应；这不能证明每次中断都是同一逻辑重试，也不算native policy失败。`psd_wire_audit.py`严格区分：未知请求、native失败、截断、capture错误仍拒绝；辅助中断保留逐条错误凭证及同请求成功凭证，canonical tool outcome另审计，不删任何失败。
- wire总预算仍2GiB，单响应上限提高到64MiB以容纳top20，保守预留在请求完成后换算实际压缩字节。11:16实测新RUN约1.10GB、wire约0.776GB。共享GPFS `df`约45TiB空闲**不是个人配额**；quota工具未提供，不能据此宣称无存储风险。全量采样前须按实际产物预算，诊断wire不能无界扩增或影响原生capture。
- 已新增独立`resume_probe`配置：原batch32/LR4e-5等不变，固定两步horizon、每步存全状态并保留两份；从checkpoint-1到新输出恢复第二步，不能改变horizon后宣称精确续跑。该模式不是production，也尚未运行GPU验收。正式production仍五epoch。
- 本地72项定向测试通过；远端v3代码既有`training/tests`389项通过。二者是代码验证，不能代替尚未完成的真实checker、repair、GPU更新/保存恢复。

**仍待完成**：固定32槽验收→真实source checker/多位置repair→同epoch3 frozen teacher top20及MM精确对齐→GPU典型/最长target和native保存/恢复→400×8完整构建→首轮五epoch正式优化。当前不能表述为“PSD已完整验收”或“正式训练已启动”。

## 2026-09-16 10:45 新授权及实际起点（优先于下方历史）

用户明确要求完整修复后开始训练，并确认使用原计划 **400 张 × 8 次**，完成全部目标构建和正式训练；不扩大到4,000张。自动执行终点已从“仅3,200槽采样”扩展到首轮PSD五个优化epoch完成。不是无限多轮PSD，也不改变原有限修复预算和真实验收要求。

- 起点改为三epoch SFT的 **epoch3 / step3084**。此前epoch2诊断只保留作证据，不混入epoch3 bank。
- 冻结导出：`/volume/ybo/wza/exports/h20-sft-merged4872-3epoch-step3084-20260915/model`。`export.json` SHA256：`1c342e73e6fc82bfa573e4435c67c2c38f26207030307313e7a1396a09a1850c`。
- 原完整checkpoint：`/volume/ybo/wza/checkpoints/h20-sft-merged4929-agent-v2-3epoch-fullstate-20260914/v0-20260914-200132/checkpoint-3084`；原三轮权重、optimizer/RNG和旧实验权重均不删除，不原地merge adapter。PSD LoRA/checkpoint单独输出。
- 新部署：`/volume/ybo/wza/training-artifacts/psd-epoch3-20260916-v1`；新的code快照保留原冻结src，只带入已修group-size校验。
- 服务：`/volume/ybo/wza/inference/psd-sft3084-20260916`。`protected-sft.json`记载全量模型SHA与checkpoint元数据实际通过校验；GPU1/19003加载epoch3，专用网关19017/public alias `ifv-qwen3.5-9b-sft-3084`。
- 4图×8正式分组验收：`/volume/ybo/wza/runs/psd-sft3084-canary4x8-20260916`，10:43已开始采集32槽；不是400×8全量，也不是优化器训练。
- 工具与感知缓存都关；单工具生成约束、think8192/out32768、128K、多图32、T0.7保持。原GPU0/2/3仍是旧服务闲时计算守护，不混入新policy。
- 新GPU守护在新SERVICE/guard.json，按各卡实际模型ID调用，避免GPU1换权重后沿用旧alias。旧guard已正常停止，不双开。
- wire archive仍为2GiB硬界限，改成“已完成实际压缩字节＋在途保守预留”，不将每条响应未用完的8MiB永久累计；不清理旧证据或放开无限存储。

完成顺序仍是：真实32槽及checker、多位置修复、去hint精确token/MM目标、同epoch3冻结teacher top20、真实GPU有限loss/梯度和native保存恢复验收，然后400×8→全量目标→LoRA r32/LR4e-5/batch32/5epoch训练。LoRA/优化参数再次核对[上游发表配置](https://github.com/essamsleiman/psd/blob/778be78bdac582b51a975ff819046583aad383e0/experiments/bfcl/configs/qwen35_9b_published.json)。

`run_psd_round.py prepare --defer-topk`新增显式阶段边界：先修复/组装目标，不在推理占满GPU时盲目加载HF teacher。随后不带此参数继续同一绑定产物，已有生成和审核不重采。

每小时监控已按照新授权和epoch3路径更新；更新方式使用[官方OpenAI文档](https://learn.chatgpt.com/docs/automations?surface=app)核对。judge收尾和朋友实验的已有边界不变。

## 2026-09-15 历史授权

2026-09-15 晚，用户明确确认“刚才你关于psd的计划可以的”。这是对后续顺序的批准，不是声称 PSD 已经开始采样或训练。

## 执行顺序

1. 当前 epoch2/step2056 测试集 Agent 推理先完成，包括原协议下的有限失败补跑。现有实验不中断，不热改采样、超时、Agent 提示或工具，不把后续修复协议的结果混入当前评测。
2. GPU 交接后，先在隔离 PSD 环境修复/验收实际生效的 thinking 8192 预算，以及模型客户端、网关和执行器的超时/取消链路。不能仅凭请求包含字段就认定生效；必须测到预算边界，确认能正常结束思考并继续工具调用/最终回答，而非 `length` 硬截断。检查超时后遗留请求及 POST 重试，避免重复生成。优先保持用户要求的 vLLM 版本，不把本次批准解释为可任意升级版本、砍图片/思考/动作预算。
3. 绑定同一 SFT 策略快照（明确 epoch、模型路径、权重 SHA、serving 设置），开展新多位置实现的真实训练案例 canary。验证源采样、Gemini checker 定位、方法性提示、实际重跑、多位置局部目标、去提示 student 输入与精确图像/token 一致性。不得用旧单位置结果或 mock 替代。Teacher 与 student 均绑定本轮初始策略；外部模型不提供被蒸馏的 top-k 分布。
4. 同一 canary 的 top-20 数据通过后，验证真实 GPU 更新、loss/梯度有限、典型及最长 target、native 完整保存/加载/下一步续跑和存储边界。不是“能算一个 loss”就认为正式训练已验证。
5. 上述门槛通过后，自动推进准备好的 400 张训练输入 × 8 次采样、temperature=0.7，共 3,200 个固定采样槽。每轮只采一次，不在每个优化 epoch 重采；不重复已完成采样去挑成功结果。这是本次获批夜间衔接的明确终点，不能将“开始采样”称为“正式 PSD 训练已启动”。

后续正式目标构建与训练方案保持：多位置 repair 每题最多六次完整重跑；合格 repair 与 preservation 每 target 权重均为 1，不强制两类各占一半；LoRA r32、LR 4e-5、有效 batch32、5 个优化 epoch、top20、seed0、128K、多图不变。没有合格 repair 不把纯 preservation 叫完整 PSD。具体 steps、个人存储与预算须在目标银行形成后核实；本次夜间批准不单独扩展为自动放行正式五 epoch 训练。

这次授权覆盖以上五步有门槛的推进；不授权绕过失败验收、无限重试、无限轮数的 PSD 或扩大到整个 4,000 图池。真实 canary 失败先诊断并保留产物，修复通过才进入大规模采样。训练前不得把 gold 或 checker 私有解释给 Agent/提示 proposer。测试观察清单只用于评测，不混入训练。

## GPU 与其他实验

外部 Gemini judge 不占 H20，因此不用等待 judge 全结束才开展 PSD。当前 epoch2 推理及重试必须先收尾；保留所有旧权重、optimizer、RNG、原始轨迹与结果。切换服务/训练前，核验请求排空，先停止当前 guard 和其目录下的矩阵 worker，再交接 GPU，避免占位与真实作业竞争。空档继续按真实 GPU 利用率守护，不把显存占用当利用率。

两组混合 Batch/普通 judge、朋友的串行实验继续各自流程，不为 PSD 改模型或重跑成功案例。关于当前评测 think8192 未生效的已知缺陷，继续如实记录；新 PSD 验收通过不追溯改变旧实验协议。

## 已核实的位置与状态

- 输入：`/volume/ybo/wza/runs/psd-pilot400-preparation-20260915-v1`。
- 隔离实现：`/volume/ybo/wza/training-artifacts/psd-adjustments-20260915/code`。
- 原准备状态 `inputs_ready_not_started`，400 个唯一图，real/fake 各 200，保留 32 条训练 canary，图片硬链接复用。原 `prepared.json` 及所有哈希绑定文件保持不变，本文件单独记录新授权。
- 原 GPU 评测：`/volume/ybo/wza/runs/eval/qwen35-sft2056-epoch2-agent-formal1527-20260915`。2026-09-16 有限补跑已结束（1524 成功、2 失败、1 缺图），结果冻结后已排空并交接到 `/volume/ybo/wza/inference/psd-sft2056-safety-20260916`；不是全量全部成功，也不继续无限补跑。
- 主实现细节与差异：[PSD 调整记录](psd-adjustments-and-agent-latency-20260915.md)。
- 2026-09-16 新增隔离预算插件与取消安全网关候选，位于 `/volume/ybo/wza/training-artifacts/psd-serving-safety-20260916`；详见 [PSD serving 安全候选](psd-serving-safety-20260916.md)。没有修改现有评测或原 PSD 准备快照。GPU 交接后优先核验该候选及显式超时配置，仍须真实生成/取消/多图验收，不可因为 CPU 测试通过就放行采样。

原小时监控将 PSD 从“暂停/未授权”更新为“当前推理收尾后，有门槛的已授权执行”，并已同步新服务/守护路径和原评测终态。使用 OpenAI Docs 核对既有定时任务更新方式，不新增重复监控。运行中未出现新异常时保持安静；真实验收失败、无法继续、完整完成或需要新增权限时报告。所有服务器写入只在 `/volume/ybo/wza` 内，Git 仅从本地提交。

2026-09-16 06:04：cache-off两个完整Agent协议案例和tokenizer转发对照已通过，当前小样推进至`/volume/ybo/wza/runs/psd-slate-canary4x8-20260916`，固定4图×8槽、T0.7、单卡并发8；它与400图正式源采集分开。源checker/多位置repair/top20/优化器及恢复测试仍须完成，不能据此将第4、5步跳过。具体现场记录见[serving安全记录](psd-serving-safety-20260916.md)。

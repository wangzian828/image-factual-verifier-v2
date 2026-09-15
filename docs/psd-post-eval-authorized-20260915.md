# PSD 后续执行授权与交接

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
- 当前 GPU 任务：`/volume/ybo/wza/runs/eval/qwen35-sft2056-epoch2-agent-formal1527-20260915`，核查时仍是 attempt0、并发40、四卡真实利用率100%。尚未到 GPU 交接条件。
- 主实现细节与差异：[PSD 调整记录](psd-adjustments-and-agent-latency-20260915.md)。

原小时监控将 PSD 从“暂停/未授权”更新为“当前推理收尾后，有门槛的已授权执行”。使用 OpenAI Docs 核对既有定时任务更新方式，不新增重复监控。运行中未出现新异常时保持安静；真实验收失败、无法继续、完整完成或需要新增权限时报告。所有服务器写入只在 `/volume/ybo/wza` 内，Git 仅从本地提交。

# PSD 调整与 Agent 耗时调查（2026-09-15）

## 结论与边界

本次补齐分组采样入口、多位置迭代修复、局部提示移除、修正历史上的 exact-token 目标和缓存恢复。400 张训练输入已落盘；尚未采集 3,200 条轨迹，也未启动新 PSD 梯度训练。当前三轮 SFT 模型和 Gemini 的正式推理不改配置、不重启、不抢卡。旧权重、原始轨迹和 4,000 条输入池均保留。

工程调查发现，近期成功轨迹平均约 **8.44 分钟**，约 **81%** 时间在主模型请求，工具约 **15%**。同期服务短窗口中，decode 占模型请求均值约 **79%**。因此优先实测推理调度/批量，随后评估前缀缓存与推测解码；目前没有部署候选加速配置，没有已测得的加速倍数。

## PSD 本次落实

以 [上游固定提交 778be78](https://github.com/essamsleiman/psd/tree/778be78bdac582b51a975ff819046583aad383e0) 为比较基准：

| 环节 | 当前实现与准备 | 验证边界 |
|---|---|---|
| 初始采样 | 400 图 × 8，temperature=0.7；训练专用 collector，固定策略快照，每轮收集一次，不是每个优化 epoch 重采 | 已准备，未执行；400 是试验规模，原实验 360 |
| 覆盖验证 | 校验每图 8 个 episode/seed/index/group 槽位，以及显式采样配置来源 | 不能把 400 条误当 400×8；不按成功结果挑采样槽 |
| 修复选源 | 默认每题取最长失败轨迹；其余原始轨迹不删除，preservation 不受此筛选影响 | IFV 按记录步骤数排序，和 BFCL 的任务步骤不是同一粒度 |
| 多位置修复 | 同一题最多 6 次完整重跑；保留已验证提示，只修改当前失败位置，按新轨迹继续定位 | IFV 映射为原生动作 0–23 与最终 judgment 24，不发明多轮用户任务 |
| 修复预算 | 删除默认 8-action 后缀限制；保留原生动作预算、阶段输出额度、4 次协议纠正，策略温度 0.7 | 1,800 秒目前是轮间软上限，不杀在途请求；与上游硬超时不完全相同 |
| 提示范围 | 提示只影响指定动作；动作完成后移出后续历史，工具返回和已产生动作完整保留 | 多位置不再把提示一直带到最终答案 |
| 局部目标 | 每个实际用到的提示对应一个 target；student 使用该次完整重跑中已修正、去提示的历史 | 不是套回原失败轨迹前缀，也不是监督整条带提示轨迹 |
| 验收 | 完整任务通过、局部动作有原文证据、结构/来源/图像/精确 token 校验通过，才接入已有 top-20 训练链 | 不强加温度 0、多次重复成功或逐提示因果对照门槛 |
| Preservation | 已通过源 checker 的完整成功轨迹即可；保持每 target 权重 1 | 不人为把两类总量各配一半；若无有效修复，不把纯 preservation 冒充完整 PSD 试验 |
| 调度与恢复 | 题间有界异步，题内依赖 checker 的重跑串行；生成、审查、目标分阶段缓存；终态输出先落盘再提交状态 | 不在采样中途更新策略，不重复已完成生成去“抽一个通过结果” |

现有发表配置对齐项保持：LoRA r32、LR 4e-5、有效 batch 32、5 个优化 epoch、seed 0、局部 top-20 归一化目标、按既有 target loss 定义归约。多模态和 128K 仍是本任务要求，不照搬 BFCL 的短上下文。没有通过缩短 think、少图、少工具动作、降低修复准入来提速。[发表配置](https://github.com/essamsleiman/psd/blob/778be78bdac582b51a975ff819046583aad383e0/experiments/bfcl/configs/qwen35_9b_published.json)

新 `run_psd_round.py prepare` CLI 默认要求 8 条/图、`longest_failed`、`slate`。旧反馈入口保留用于历史记录回放，不将其历史结果改称新多位置结果。

### 精确输入重建修复

一次只读真实训练请求诊断发现三个问题：归档字段叫 `tool_schema`；原生工具 schema 必须转换为 OpenAI function envelope；归档的 JSON key 排序和 JSON 字符串 pretty-print 会改变模板 token。生产捕获现在保留 **在内存中的原始文本和工具顺序**，只从归档恢复图片字节，再逐 token 对照真实 teacher capture。去掉局部提示后才构造 student token；图片字节、顺序必须不变。

诊断记录来自 `psd-real-training-canary-20260912-r2/on-policy-r1` 的一条历史请求，14 个工具、2 个图片块。定位排版差异后，4,486 / 4,486 个输入 token 完全匹配。历史诊断中还原已知 JSON 的方法只用于确定原因；生产代码不以启发式压缩任意归档文本来绕过精确校验。此检查没有执行新的模型生成，**不等于新多位置真实轨迹验收**。

### 仍须真实验收的部分

- Gemini 代替 BFCL 确定性 checker、代替 Opus 写提示是任务适配。当前 proposer 只看公开轨迹、图片及 checker 的位置反馈；私有 gold/审查解释不泄漏给 proposer。定位是否准确、提示是否足够有效，仍需新策略真实案例验证，不能继承 BFCL 的效果结论。
- 新策略上完成一次真实多位置修复、全部局部 token/MM targets 的生成和目标构建；不能用两个模拟 action 的测试替代。
- 新目标的典型/最坏长度 GPU 吞吐、optimizer update、完整 native 保存/加载/下一步续跑；旧无保存一步 smoke 不能替代。
- 核实用户实际存储 quota/支出预算；共享文件系统空闲空间不是个人配额。

CPU 回归：本地无 torch 子集 376 passed；服务器禁用 CUDA 的完整 `training/tests` 389 passed。包含多位置历史隔离、live/归档 JSON 与工具顺序、严格 token mismatch 拒绝、两次修复后物化中断恢复、review 证据验证和样本覆盖检查。没有由这些测试推出能力提升。

## 400 图准备位置

服务器隔离代码：`/volume/ybo/wza/training-artifacts/psd-adjustments-20260915/code`，没有替换运行中的主实验源码。

新输入目录：`/volume/ybo/wza/runs/psd-pilot400-preparation-20260915-v1`：400 个唯一图，real/fake 各 200，保留既定 32 条训练 canary；公共输入与私有 gold/split 分开。图片硬链接复用，新增图片字节为 0。旧 4,000 图池不改。这里的 400 图训练 pilot **不是**原来的 400 条测试观察清单；后者只评测。

可重复验证的准备命令（无外部 API、无 GPU 生成）：

```bash
PYTHONPATH=.:training python scripts/prepare_psd_pilot.py \
  --previous-plan /volume/ybo/wza/runs/psd-experiment-preparation-4000-20260915-v1 \
  --storage-root /volume/ybo/wza \
  --output /volume/ybo/wza/runs/psd-pilot400-preparation-20260915-v1
```

`collect_psd_rollouts.py` 是独立训练采样入口，要求显式训练 split、公共 benchmark、8 次采样以及原生 `run_cases` 参数。实际启动仍须绑定最终 SFT 快照/serving attestation、外部环境和新输出目录。此文不提供伪造快照的“可执行”全量命令。

## 真实耗时拆分

2026-09-15 约 14:21（中国时间），检查三轮 SFT 模型正式评测目录最近写出的 150 个 trace 文件，其中 143 个成功、7 个错误/未完整。以下耗时只统计这 143 个成功 attempt，**排除了仍在途和失败长尾，不是全测试集速度或失败率**；亦未按最终 case 合并用于质量指标。

| 指标 | 测量值 |
|---|---:|
| 单条平均 / 中位数 | 506.27 / 484.17 秒 |
| 单条 P90 / P95 | 759.72 / 855.87 秒 |
| 主模型请求耗时均值 | 410.04 秒（80.99%） |
| 工具外层耗时均值 | 78.24 秒（15.45%） |
| 其他/未归因 | 17.99 秒（3.55%） |
| 主模型请求数 / 原生动作数均值 | 14.89 / 13.58 |
| 累计输入 / 输出 token 均值 | 300,902 / 6,273 |
| 最终 judgment 模型耗时均值 | 66.95 秒 |

累计输入是十几次请求重复携带历史的总和，**不是单次超过 128K**。思考统计中的字符数不是 token 数。主模型耗时含排队、prefill、decode、网络；工具中可能包含嵌套模型调用，不能再次叠加子调用时间。

工具累计耗时最突出的是网页访问（347 次，单次均值 12.27 秒）；纯文字搜索 576 次，均值 2.03 秒；反搜 155 次，均值 7.08 秒。视觉分析/参考图对比等单次更慢但调用较少。模型后端本来已有复用 HTTP client，不能把“加连接池”当成尚未使用的核心收益。

14:23–14:27，四副本 Prometheus 取差值，窗口 247.926 秒、完成 312 个服务请求：

| 服务请求均值 | 秒 |
|---|---:|
| 排队 | 2.334 |
| Prefill | 2.424 |
| Decode | 21.772 |
| 端到端 | 27.660 |

decode/端到端均值约 78.7%。窗口内输出吞吐约 563.88 tokens/s（四副本总计），preemption 为 0；末端各副本 running=8、waiting=0/1/0/0，未看到旧的严重单副本排队失衡。服务请求可能包括工具内部模型调用，与上述 trace 样本不是严格一一对应的 cohort；不直接相乘推算全批时间。

当前配置为 vLLM 0.18.1、4 个 TP1 副本、每副本 seq8、batched tokens 32768、128K、显存比例 .94、CUDA graph 已启用。前缀缓存关闭（查询/命中增量均为 0）。采样时四卡 GPU 利用率均 100%，已不是“只占显存不计算”。

## 工程优化优先级与验收

### 1. 先测试批量与调度组合

候选为 `max_num_batched_tokens=8192/16384/32768`，配合 `max_num_seqs=8/12/16`，再测 case 并发 16/24/32/40。先少量筛选，不能全排列消耗正式任务。维持同一模型、128K、多图处理、prompt、think/输出额度和工具语义。

小 prefill chunk 可以降低生成时的干扰，但吞吐未必更高；大 batch 可能增吞吐但增加单条等待。官方文档明确区分 ITL、TTFT 和吞吐，不支持“并发越大一条越快”的推论。[vLLM 0.18.1 调优说明](https://github.com/vllm-project/vllm/blob/v0.18.1/docs/configuration/optimization.md)

分别记录单条/P95 耗时、稳定 tokens/s、有效完成 cases/h、超时/重试、preemption、峰值显存。默认保留目前均衡的 least-inflight 网关；不要为压低单请求延迟牺牲整轮 PSD 完成时间。

### 2. 前缀与多模态缓存

同一轨迹多次携带历史有复用空间；400×8 首轮还重复同一初始输入。优先验证当前版本 Qwen3.5 混合结构的缓存支持，再尝试带负载溢出的软亲和路由，避免重新造成某副本堆积。缓存不减少输出生成计算，不能把输入复用比例当整体加速倍数。[APC 原理](https://github.com/vllm-project/vllm/blob/v0.18.1/docs/features/automatic_prefix_caching.md)

`mamba_cache_mode=align` 在官方 Qwen3.5 recipe 中仍标为实验性；该 recipe 的验证环境主要是大 MoE/8×H200 等，不是当前 9B/H20。须实测多图、精确 token/top-k、长上下文和稳定性，不直接热改现有服务。[官方 Qwen3.5 recipe](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md)

### 3. 推测解码 / MTP：有针对性，但目前不能直接打开

最终 SFT 导出模型的 index 有 760 个权重条目，`mtp`/`nextn` 权重条目为 0；配置里的 `mtp_num_hidden_layers=1` 不等于权重存在。这只说明辅助 MTP 不在导出中，**不说明主模型训练权重丢失**。Transformers 对相应 MTP key 有忽略加载规则，支持“常规 SFT 导出不保留辅助 head”的解释，但不是本次训练路径的完整因果证明。[Transformers 实现](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

因此先查兼容 draft/MTP 权重和加载方式，保持主 SFT 参数不变；再测接受率、单条耗时、并发吞吐及 exact logprob 路径。不能随机初始化 head 后宣称开启成功，更不能改回 Base 主模型。官方说明 MTP 可降低低并发 token 延迟，但高并发可能损害吞吐；尚无此硬件/权重上 2× 加速的证据。[推测解码限制](https://github.com/vllm-project/vllm/blob/v0.18.1/docs/features/speculative_decoding/README.md)

### 4. 外部工具与 PSD 流水化

继续利用已有连接复用、按完成顺序的题间并发和持久缓存。可以测试工具内部**相互独立**的下载/解析工作重叠，但不能并行执行依赖上一步观察的 Agent 决策、复用会过期的搜索结果或隐藏失败响应。需要改 Agent 工具实现的候选仅列为后续方案，不在当前冻结正式评测中实施。

PSD 的源采集、审查和独立题目修复可做有界流水化；同一题提示依赖实际重跑反馈，不能当成六个独立请求一起发。保持整轮 teacher 快照冻结与目标来源一致。

## 预期不能夸大

仅作 Amdahl 示例：假设主模型请求整体提速 2×、其他不变，当前样本单条约从 8.44 分钟降至 5.0 分钟（1.68×），不是整条 2×；即便工具时间归零，整体上限也只有约 1.18×。这不是实测收益。现在可以确认优化方向，尚不能承诺把完整调查压到几十秒。

下一阶段在主评测交接后先做隔离小规模性能 A/B 和真实 PSD canary，确认收益与质量后才决定正式配置。所有服务器操作仍限 `/volume/ybo/wza`，Git 提交仅从本地进行。

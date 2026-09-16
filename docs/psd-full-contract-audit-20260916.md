# PSD 全链路契约审计（2026-09-16）

**当前更新：v12 已结束，v13 位置硬锁方案已废止；使用 v14 观察位置修复协议。**
参见 [位置纠正记录](psd-observed-position-correction-20260916.md)。以下 v12 启动信息是历史记录，不是当前运行状态。

本次范围是 PSD 的采集、修复、审核、目标、训练及续跑链路；不是重做主实验评测。
三轮 SFT 原参数、完整 optimizer/RNG checkpoint、旧轨迹和旧审核全部保留。
**正式 PSD 优化仍为 0 step。单测、CPU 梯度对照、短服务请求均不等于真实 GPU 全流程验收。**

## 已确认的问题与处理

| 问题 | 实际证据 / 影响 | 本次处理 |
|---|---|---|
| 已渲染 system 被再次当作基础 prompt | 真实异常快照出现两份工具可用性/结构约束；修复 Agent 偏离原 Agent | 从冻结 Orchestrator 获取原始 ReAct/Judgment prompt，只渲染一次；不改 `src` |
| 目标捕获拿旧 system 比对新工具可用性 | 两份真实失败请求可以精确重建为 16,817 / 18,516 teacher tokens | 先绑定 live snapshot 与 archive，再采用当次 system；teacher token 必须逐个相等，未放宽匹配 |
| PSD 诊断 stage 名称让合法证据 ID 消失 | 一个真实修复有 14 个观察 ID，冻结 reader 只接受 `unified_react`，读成 0 | 仅向 reader 传规范化副本，原 stage/ID/错误不改；原先被拒绝的 ID 均在观察集合 |
| checker 投影遗漏 transport ID | 固定 32 条原轨迹共 422 个工具 ID，旧投影全部未保留 | 只增加 `metadata.function_call_id` 白名单字段，不导出 token dump、凭据或私有 metadata |
| 修复审核看不到实际提示 | 原 packet 只有提示位置，没有完整提示文本；正则不能证明语义上不泄漏答案 | reviewer 接收实际 hint，并检查方法性和动作；v2 审核绑定 hint hash，无匹配 v2 审核不准入 |
| 最终报告前被拒绝的 Judgment 草稿丢失 | 完整轨迹构建只留最终成功 output | 保留实际 format_error/rejection 决策、原错误和用量，不伪造干净轨迹 |
| 新 slate 路径缺少单写者锁 | 新分支绕开旧 feedback 路径的锁；重复监控/续跑可竞争 | 定位、写输入、调用 provider 之前获取进程所有锁；验证第二写者拒绝、退出释放 |
| source 审核断点缺少材料版本 | prompt/schema hash 不变时，旧材料的审核可能被新脚本复用 | `transport_ids_v2` 纳入批量输入身份和每份结果校验；旧结果可按旧协议读取，但不能当新版使用 |
| vLLM 多模态缓存实际缺项 | 13:40:11 backend0 traceback：`Expected a cached item`；两个新诊断请求 HTTP 500 | 排空后仅重启所属四副本，设置 `--mm-processor-cache-gb 0`，保留其他生成参数；随后重复多图验收 |

前两份精确 token 验证与 14 个 ID 验证使用真实失败现场，不是只有 mock；旧失败产物不因此变成可训练目标。
缓存问题是引擎请求预处理失败，不是“模型回答错”，也不是无证据推断缓存污染。
[vLLM 官方说明](https://docs.vllm.ai/en/latest/configuration/optimization/)明确该设置关闭 processor/IPC cache；本机 0.18.1 源码也核实了该参数。尚未声称确定了上游缓存失同步的更深触发条件。

## 验证与当前实况

- 固定源仍是 4 张预选训练图 × 8 = 32 条，没有按审核结果替换图片或重采；原验收有 448 个去重原生决策、59 次感知/OCR，均保留 native capture。
- 405 项 `training/tests` 在服务器 Python 3.12 通过。安装的真实 vLLM 0.18.1 环境另 **92 项服务/网关/parser 测试全部通过**；已显式设置 epoch3 tokenizer 路径补测原先跳过的 6 项预算边界测试，不需要加载 GPU 权重。
- 真实 HF Trainer + Swift bridge 梯度累积与独立 sum-loss 参考一致（包含不完整尾批）；4 进程 CPU DDP 的全局梯度一致。
- tiny random Qwen3.5、真实 processor、两张图、视觉梯度、adapter 保存重载验证通过；**这是 CPU 架构检查，不是 9B 长目标 GPU/FSDP 保存续跑验收**。
- v10 三任务真实搜索结束：1 个 `no_recoverable_site`，2 个 HTTP 500；0 个完整 repair continuation / 0 个准入目标。不得标为“模型修复成功”或重复审核来挑有利结果。
- 独立 `source-reviews-transport-v2` 已完成相同 32 条原轨迹的材料版本修正 checker：**15 pass / 17 fail / 0 unresolved / 0 pending**。旧 12 pass / 20 fail 只代表 legacy 投影，不能作为新版结果。该变化不是重采源轨迹或挑选有利 judge。主实验已完成的 judge 不重跑。
- 四副本缓存关闭完成，逐卡 receipt 和 epoch3 模型核对通过；**16/16 重复双图请求成功**且返回 native token/top20。保留此检查的界限：不是完整 Agent/PSD GPU 验收。新部署和缓存恢复各约 17MiB，整个固定源/诊断根约 1.5GiB；不是复制四份权重。
- 新独立源银行已构建：17 repair candidates、15 preservation candidates，engineering/token-capture/pending/rejection 均为 0。原源文件逐一哈希确认未变；只硬链接不再写入的原始 traces，派生评分/audits/rewards 全部写独立目录。源 gate 仍如实绑定原 19018，修复切到同权重 19019，不篡改源 manifest。
- **v12 有限完整修复已启动**，并发配置40、同题最多6次/12提案、最长失败源/任务；只是固定32源的小样验收，不是400×8采样或优化。新的 source 审核、锁和全部代码修复已在当前代码快照生效。
- 四卡 guard 使用真实推理脉冲，只在引擎 idle 时触发。某瞬间 0% 不等于 guard 不工作；显存占用本身也不能证明保活。服务滚动切换期间按 PID/完整命令核验，不停止无关实验。

## 证据和控制入口

服务器根目录固定 `/volume/ybo/wza`，其外不写、不清理；服务器不运行 Git。

- 原采集根：`/volume/ybo/wza/runs/psd-sft3084-captured-canary4x8-20260916`。
- 审计部署：`/volume/ybo/wza/training-artifacts/psd-contract-audit-20260916-v10`。`state.json` 是三例诊断状态；`*-failure.json` 指向受限权限的诊断 gzip，不把原始轨迹/秘密提交仓库。
- 修复结果：采集根的 `psd-contract-audit-v10/progress.json`。v8 已结束，v9 从未启动；不要恢复旧损坏逻辑。
- 新 source checker：部署下 `source-review-v2-process.json`、`source-review-v2-state.json`；独立代码 `source-review-v2-code`；结果在采集根 `source-reviews-transport-v2`。
- **当前继续入口**：部署下 `run_psd_corrected_contract_gate.py`，输出采集根 `psd-corrected-contract-v12` 的 `process.json` / `state.json` / `run.log` / `search/progress.json`。PID1060473仅启动线索；不要重新启动旧v10。运行中不热改 `source-review-v2-code`。每小时监控已同步此入口并保留原400×8/5epoch授权及门槛。
- 缓存修复：部署下 `mm-recovery-process.json`；服务根 `/volume/ybo/wza/inference/psd-sft3084-20260916/mm-cache-recovery-v11` 的 state/各副本 receipt/日志/`multimodal-smoke.json`。
- **服务当前 PID 以服务根 `replica-{0,1,2,3}.json` 为准**，不能拿 earlier v7 的历史 PID 杀进程。公共网关 19019，后端 19002/19003/19004/19005。
- 不变的 SFT 导出：`/volume/ybo/wza/exports/h20-sft-merged4872-3epoch-step3084-20260915/model`；export SHA256 `1c342e73e6fc82bfa573e4435c67c2c38f26207030307313e7a1396a09a1850c`。
- 原 fullstate：`/volume/ybo/wza/checkpoints/h20-sft-merged4929-agent-v2-3epoch-fullstate-20260914/v0-20260914-200132/checkpoint-3084`，包括 optimizer/RNG，禁止覆盖/删除。

## 放行前仍需完成

1. 已完成：四副本缓存设置、重复多图、原生 token/top20 的实际返回验证。继续核查长真实请求，不能由短请求推断长期无故障。
2. 已完成：新版 32 条 source checker 全部有效，独立派生产物新路由构造通过，原源银行及旧 `post_rollout_rewards.jsonl` 哈希未变。后续若使用同一 `prepare` 入口，仍须隔离其会写的派生文件，不能硬链接后原地覆写。
3. 新版完整多位置修复及提示语义审核实际通过，核对 teacher/student 只去 hint、图片顺序/字节不变、token 精确相等、私有参考不流向 proposer/Agent。旧审核或失败 target 不准混入。
4. 同 epoch3 frozen teacher 的 top20 评分；典型及最长真实 target 的 9B GPU 有限 loss/梯度/更新；固定两步 horizon 的 native fullstate 保存、恢复和第二步对照。CPU probe 不能替代。
5. 上述通过后继续已授权 400×8 源采样、全目标构建、5 个优化 epoch；采样/跨案例修复并发 40，同题修复串行。GPU teacher/训练交接前排空所属服务和 guard。存储先查个人实际用量，共享 `df` 空闲不是个人配额。

## 与原方法保持一致的边界

对照 [上游发表配置 778be78](https://github.com/essamsleiman/psd/blob/778be78bdac582b51a975ff819046583aad383e0/experiments/bfcl/configs/qwen35_9b_published.json)：top20、每目标权重 1、LoRA r32、LR 4e-5、有效 batch32、5 epoch。成功轨迹不要求 temperature=0 或反复证明稳定成功；不强制 preservation/repair 总量各半。

当前 8 次源采样、T0.7；单题完整 repair 最多 6 次、proposal 最多 12 次。128K、多图32、think8192/out32768及完整工具步骤不削减。事实核查用 Gemini 定位/验收是任务适配，不能冒充 BFCL 的确定性环境 checker。通过数量为零时不能将纯 preservation 当完整 PSD，也不能降低审核标准凑数。

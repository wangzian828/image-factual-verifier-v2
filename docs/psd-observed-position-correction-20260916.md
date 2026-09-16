# PSD 修复位置硬锁纠正（2026-09-16）

本记录取代 v13 的“把 schema 也限制到 failed_position”部署方案。v13 不应上线。
用户核对原方法后明确要求纠正。原三轮 SFT 参数、完整 checkpoint、源轨迹、旧修复结果均保留。

## 16:28 后续：持续监控与复用完整调查

现有 `H20训练与评测流水线监控` 已临时改为 **每 15 分钟**，不是另建重复任务。
用户要求持续推进到正式 PSD 能完整启动；触发时应执行有证据的诊断、局部修复、测试和有限恢复，不能只轮询。
只有真实修复/目标/9B 更新与原生保存恢复验收通过、400×8 全部源槽与目标构建完成，且独立训练实际连续执行多个正常 optimizer step，才达到这一阶段终点。
达到后恢复原每小时监控并继续训练观察；所有原采样、思考、图片、工具、审核和有限修复预算不变。

- v14 原控制器已于约 **16:06** 结束，`paused_search_requires_resume`；下文启动 PID 是历史记录。
- `7996ba9752652f37` 实际已完成一条 **37 个 StageStep 的调查**及位置 2 的一个候选目标。原中断仅是 Gemini `review_slate` 传输超时，不该重采这条 Agent 轨迹。
- 新 `scripts/server/recover_psd_cached_review.py` 校验原输入、pending 提案、完整 continuation/episode 的绑定与哈希，持原单写者锁，仅补同材料、同模型、同审核 schema 的缺失审核。仅本次传输 deadline 改为 900 秒、传输重试 0；没有改变 high/8192 的审核配置。
- 补审核于约 **16:21 完成，结论 fail**：位置 2 的提示给出了图中文字的具体读法，不是纯方法性指导，被审核判为答案泄漏。这个失败完整保留，不通过重复审核挑选通过结果，也不把候选目标当合格目标。
- 新 `scripts/server/resume_psd_reviewed_case.py` 只恢复这一案例，复用同一条完整调查及已缓存失败审核，继续原每题 **6 次完整修复 / 12 次提案**预算，不重置。约 **16:28** 已核实完成轮数从 0 变为 1，下一次提示修订运行中；它不是正式训练。控制目录为本运行根下 `reviewed-case-continuation-v1`，启动 PID1076635 仅线索，应核验 receipt、状态和当前进程。
- 另两例没有盲重跑：`e79d4db478e45800` 的 Qwen req-000006 再现 NaN/JSON 序列化错误，backend2 同时出现 xgrammar FSM/tokens0 错误，但这还不能确定 NaN 根因；`82897bdeb31b6ade` 卡在首个 Qwen 请求超时。不能把工程故障当作模型失败或 NaN 置零放行。
- 本地恢复、slate、位置规则、pipeline 和存储 **40 项相关回归通过**，含新增 5 项恢复校验：复用完整缓存、拒绝源/episode/continuation 变更，以及续跑路径、版本和预算恢复。它们不替代 9B GPU 验收。

当前仍是 **正式 PSD 优化 0 step，400×8 尚未启动**。四副本 `/health` 正常，真实计算 idle guard 保留；没有重启全部服务、覆盖三轮 SFT 或扩充修复预算。补审核产物在本运行根的 `review-recovery-v1`；两份恢复脚本部署在 v14 根目录，**不热改不可变 `code/` 快照**。

## 原方法与错误适配

对照上游提交 `778be78bdac582b51a975ff819046583aad383e0`：

- [Agentic 提示词](https://github.com/essamsleiman/psd/blob/778be78bdac582b51a975ff819046583aad383e0/experiments/bfcl/agentic/system_prompt.md)让修复 Agent 根据可见轨迹选择少量提示，优先处理前因，保留已使对应回合通过的提示。
- [执行入口](https://github.com/essamsleiman/psd/blob/778be78bdac582b51a975ff819046583aad383e0/experiments/bfcl/repair/slate.py#L391-L419)不检查提示位置必须等于初始 first_bad_turn；读取 slate 后完整重跑。
- BFCL 的 turn 是用户消息回合，我们的 position 是单图调查中的原生决策序号，不能声称完全等价。

v12 中 28 个被拒提案的首个拒绝原因有 25 个为“非指定位置”、3 个为提示边界。
这不证明前 25 个在内容上都合格：部分可能同时泄漏答案，需要独立审核。
把它们统称为“生成器改错位置”不准确；过强的位置限制是我们的实现问题。

## 已纠正的代码边界

1. 提示词明确 `failed_position` 是诊断信息，而非排他的干预位置。
2. Provider schema 枚举公开 `decision_map` 中实际观察到的位置，允许更早位置及多个位置；不枚举不存在的动作。
3. 本地 validator 取消 `changed - {failed_position}` 拒绝条件；仍拒绝未观察位置、重复位置、非法格式，以及修改/删除已验证有效的提示。
4. 不再拒绝相同 slate 的有预算重跑：随机采样下相同提示可以再试，但必须计入完整重跑预算。中断恢复仍复用已完成的同一次生成，不静默重采。
5. 提示的私有答案泄漏检查、实际 hint 语义审核、完整轨迹 checker、精确 teacher/student token 与多图捕获均保留；没有放宽准入凑数。
6. 三层版本绑定同时更新：driver continuation v6、slate search v3、外层 canary 的 position policy。旧 terminal/pending 结果不能被新版当成同一协议继续。

## 验证与独立真实任务

- 本地 31 项定向测试通过；目标服务器 Python 3.12 全部 `training/tests` **421 项通过**。
- 测试包含诊断位置 21 而提示选择 3/4/6、多位置、未观察位置拒绝、已通过提示不变、重复提示有限重试、隐私与提示泄漏拒绝。
- 集成回归使用真实 proposal/parser/budget 路径，验证位置 21 的定位不再挡住位置 0 的完整重跑；模拟 materialization 中断后不重复生成。
- 独立代码：`/volume/ybo/wza/training-artifacts/psd-observed-positions-20260916-v14/code`。冻结 `src` 与 v12 逐文件哈希相同。
- 控制器：同部署目录的 `run_psd_observed_positions_gate.py`；真实输出：`/volume/ybo/wza/runs/psd-sft3084-captured-canary4x8-20260916/psd-observed-positions-v14`。
- 仍用 v12 的固定 32 源、有效 source checker、同任务选源；语义定位直接复用原文件并绑定哈希，其他案例保留原机械定位，不重复定位以挑选有利结果。
- 新协议每题最多 6 次完整重跑 / 12 个提案，跨案例上限 40；本 canary 仅 3 个选定失败任务，不冒称 40 条完整 Agent 并发。
- 新控制器已启动，PID1070242 只是启动线索，以 process receipt/当前进程身份和产物为准。**尚不能据此声明真实修复通过或正式 PSD 优化已开始。**

旧 v12 的 vLLM NaN/JSON 序列化异常并未因这次位置规则改动而得到修复；继续独立诊断，不能把工程故障算模型失败，也不能 NaN 置零或取消 top20 来过关。
GPU 服务和 idle 真实计算守护没有重启或停止；朋友实验不受本次代码更改影响。

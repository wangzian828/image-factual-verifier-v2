# PSD 夜间正式推进（2026-09-16）

用户已授权离开后自主推进到完整 PSD 优化训练启动。固定 400 张 × 8，
基于三 epoch SFT step3084；不扩大到 4,000 张，不覆盖原权重。

## 顺序与验收

1. 正式采集 3,200 槽，四卡、并发 40；首 40 槽在线工程验收，完整产物直接保留。
   每槽最多三次基础设施尝试，正常模型失败不重采挑成功。全量逐槽核对身份、seed、
   原图与完整 native capture。首批通过后仍保持连续 40 并发，不按 40 一批反复等长尾。
2. 全部源轨迹 checker，失败任务按上游规则选择源、生成提示并真实完整重跑；每题
   6 次完整修复/12 提案，跨题并发 40。preservation 保留真实通过轨迹。
3. 同一冻结 Qwen 的 exact IDs/pixels 原始 top20 教师评分；服务采样 logprobs
   不冒充无掩码教师目标。私有答案不进入 Agent/提示生成器。
4. 用最终数据做四卡 global batch32 短步验收：有限 loss/梯度、参数真实变化、
   原生全状态保存恢复。此前 199 目标单 GPU raw gate 与旧 DP4 gate 不替代这一步。
5. 独立输出启动 5 epoch PSD：LoRA32/alpha32/dropout0，LR4e-5，Adam(.9,.95)，
   eps1e-12，wd0，top20，目标权重1，token loss SUM。多个 optimizer step 正常
   推进才报告“正式训练已启动”，不能把采集/服务启动算成训练。

完成耗时取决于源调查和真实修复；用首批及后续完成率估算，不能保证今晚一定进入优化。
未达到目标也继续自主推进，不因为一个阶段结束而停止。

## 当前执行入口

- 不可变已验证采样代码：`/volume/ybo/wza/training-artifacts/psd-infrastructure-retry-20260916-v23/code`。
- 正式控制器：本仓库 `scripts/server/run_psd_production_collection.py`。
- 正式输出：`/volume/ybo/wza/runs/psd-production400x8-20260916-v1`。
- 网关 `http://127.0.0.1:19025`，四 stock worker 19002–19005。
- 控制器仅调度并缩小内存中的结果摘要，不删除磁盘完整轨迹、改变 Agent 或生成参数。
- 状态文件 `state.json`、`collection-progress.json`、`first40-gate.json`、`storage.json`，
  权威进程凭证 `process.json`。复查活进程和 `controller-lock`，不可重复启动。
- 后续使用 `scripts/run_psd_round.py prepare` 的完整输入绑定，repair mode slate、
  task-source-selection longest_failed、expected-rollouts-per-case8、case-concurrency40、
  defer-topk。正常模型错误与未解决基础设施失败须明确区分，不能改 manifest 掩盖错误。
- 最终 raw teacher 与 DP4 launcher 需绑定本次最终 bank；旧 canary 的硬编码入口不能直接运行。

## 无人值守与保护

恢复已有 `gemini-3-1-pro-agent` heartbeat，每15分钟检查并推进，不新建重复任务。
本机/Codex 要保持运行才能继续 AI 判断与修复；服务器 collector/GPU guard 独立存活。
正式训练验收后恢复每小时监控训练及朋友实验；旧已完成 judge 不重新付费提交。

所有服务器写入仅 `/volume/ybo/wza`，禁止服务器 Git，凭据不输出。
保留所有旧 SFT1/2/3epoch 权重和原生 checkpoint；新 PSD 单独目录。
GPU guard 继续真实计算脉冲，阶段交接排空身份明确服务，不能抢训练显存。

个人 GPFS 配额仍未知，45TiB 是共享余量而非个人配额。新 source run 设置128GiB
准入上限（不是用户配额承诺），每60秒检查新目录实际占用和共享盘16GiB最低余量。
达到上限仅停止新槽准入，让在途请求完成并保留证据，由接续任务诊断空间，不盲删旧数据。
网关重复 wire 归档关闭，native 完整 token/图片/工具轨迹不缩减。首40后估算总增长，
及时处理而非等写满。上限可能被已有在途请求越过少量，不宣称硬磁盘配额。

本文件为执行计划；实际是否启动及完成量以服务器上述状态与进程为准。

## 首次启动实况

- 代码 `fe5522d` 已从本地推送；本地及服务器6项新增定向测试均通过。
- 部署入口 `/volume/ybo/wza/training-artifacts/psd-production-collection-20260916-v24/run_psd_production_collection.py`。
- 正式控制器已启动（初始PID1216870只是线索），40个持久化槽与40个真实runtime已存在，
  四副本同时有在途请求；一次观测 GPU 利用率为100/89/100/100%。
- 首条已结束轨迹 `main-02754--d5fdfd7f--r003` 是 unusable tool response，
  `finish_reason=tool_calls,content_chars=0,reasoning_chars=1`。原始错误保留；
  不能凭该文本认定 NaN，也不能为了得到正确答案自动重采。终端输出是
  `deterministic_segment_boundary=true`，没有模型capture是预期的异常终止记录，
  在线capture检查跳过该机械段而不跳过真正模型输出。
- 后续接续要重点核对既有下游 `review_psd_sources.py`、`postprocess_psd_training.py`
  和 `psd_round.py` 的 `status == completed` 门槛：正常模型错误会让 native manifest
  写成 `completed_with_errors`。需要用完整槽位/基础设施ledger证据区分结束状态与
  模型答对率，不能篡改 manifest 为 completed、丢失败槽或重采到全部无错误来过门槛。
  这是待完善的下游接续点，尚未修改或声称已通过。

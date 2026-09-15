# 两组 Agent judge 切换普通接口

2026-09-15 晚用户授权：Batch 持续 429，未提交部分改用普通 API，接受普通计费。只改传输及调度，不改 judge 协议，不影响 H20 epoch2 Agent 推理、GPU 服务或朋友的实验。

## 唯一所有权

| 候选模型 | 已受理 Batch，保留 | 普通 API 所有权 | 可执行案例总量 |
|---|---:|---:|---:|
| Qwen3.5-9B，三 epoch 训练的 epoch3/step3084 | 74 | 1,452 | 1,526 |
| Gemini3.1Pro Agent | 75 | 1,451 | 1,526 |

原来的 28 个 Batch 不取消、不重复。旧提交器已自然达到有限重试上限退出。切换入口持有同一个 `submit.lock`，排除不确定的 Batch create 后冻结 journal SHA 与每源案例所有权；`transport-switch.json` 永久阻止旧提交器继续提交。不是将已提交的 149 条也重跑。

输入、journal、所有权文件均在服务器：

`/volume/ybo/wza/evaluation/sft3084-gemini31pro-v3-judge-20260915`

两个 gzip 候选包沿用已验证的源 SHA，不重新导出，不改变完整 v3 材料。普通请求与 Batch 逐字使用同一个 prompt、JSON 序列化和图片字节：`gemini-3.7-flash`、`low`、`include_thoughts=true`、输出 32768、相同 JSON schema，无新增温度设置。

## 普通接口运行与验收

隔离脚本：`/volume/ybo/wza/training-artifacts/run_realtime_agent_judges_20260915.py`

Python：`/volume/ybo/wza/envs/h20-qwen35-128k/bin/python`

输出：`/volume/ybo/wza/runs/eval/sft3084-gemini31pro-hybrid-judge-v3-low32k-20260915`

启动参数：`--stage run --workers 4`。2026-09-15 21:24:50（中国时间）启动 PID 945541，仅作定位线索，检查时必须核验真实身份。先每源 2 条共 4 条 pilot，必须全部正常 STOP 且通过 schema 验证，才自动放行其余案例；pilot 的有效结果直接计入全量，不重复调用。首两条约 9 秒和 14 秒；另两条首试 503，退避重试后也成功。21:26:49 四条真实 pilot 全部验收通过，`pilot-acceptance.json` 已落盘，自动进入全量 4 路并发。21:27 快照已有 7 个有效结果；这是早期进展，不是全部完成。

每个案例分别保留 request 身份摘要、逐次 attempt、原始 response（含 usage/thoughts）和规范化 result。先持久化 response 再验收；重启可从原始 response 恢复，不依靠日志猜测。已经成功或合法但质量低的 judge 不重采。

显式 408/429/500/502/503/504 最多 6 次，指数退避并加抖动；无返回的网络异常/超时保留为不确定，先核查，避免无限重复付费。非正常结束、schema 错误不能算有效 judge。连续三条最终错误停止新任务并排空已发请求。具体诊断看每案例 status/attempt，而不只看控制器是否存活。

普通接口不保证无 429/503。[官方错误处理说明](https://ai.google.dev/gemini-api/docs/troubleshooting)支持针对暂态错误采用有限退避。

两源共用一张 17,581,496 字节 GIF，Base64 后超过保守 20MB 请求阈值。脚本保留为 `deferred_file_quota`，待其归属明确的 File API URI 可用再处理；不缩图、不删图、不删除他人文件。官方[图片输入说明](https://ai.google.dev/gemini-api/docs/image-understanding)与[通用文件输入说明](https://ai.google.dev/gemini-api/docs/file-input-methods)的限制描述并不完全一致，尚未实际验证该模型更大内联请求，不据此宣称两条已解除阻塞。

## 收集、合并与监控

原 Batch collector 继续保存 28 个任务的 checkpoint：

`/volume/ybo/wza/runs/eval/sft3084-gemini31pro-batch-judge-v3-low32k-20260915/collector-state/batch-checkpoints`

混合收集器使用同一个 `_audit_record`/`_category` 定义，按 source + case_id 校验互斥并合并。运行时每分钟刷新混合输出的 `summary.json` 和每源 `audit-results.jsonl.gz`。普通控制器退出后，可运行同一脚本 `--stage merge` 收集后来完成的 Batch；运行中不能另外抢占 `submit.lock`。

只有每源 1,526 个有效 judge 齐全才写最终 SESR，统一分母 1,527。未完成时 `sesr_reported_percent=null`，不把部分结果当最终分数。Batch 失败案例不能偷偷改成普通所有权，必须先明确核实终态并更新转移记录。

原小时监控已更新，禁止复活旧 Batch submit，继续检查两路落盘与合并并回写对比大表。OpenAI Docs 用于核对现有监控的更新方式；未创建重复监控。已完成的其他 judge 不重跑。

本地验证：11 项定向单元测试、受影响 Python 编译、`git diff --check` 通过。测试包括互斥分区、重复/未知案例拒绝、源 SHA 校验、不确定 create 拒绝、与 Batch 完全相同文本、schema 验收及有限重试。真实 pilot 另行验收，离线测试不能替代真实 API 成功。

## 23:32 巡检与安全续跑

2026-09-15 23:32 巡检确认，普通控制器已于约 23:04 退出：两源各完成 109 条，另各 5 条均收到明确 HTTP 503、达到每案例 6 次上限；共 2,675 条尚未尝试。没有遗留 `in_flight` 或未回收的原始响应。原 28 个 Batch 仍全部 `JOB_STATE_RUNNING`，collector 持续更新，未新增提交。

另确认一处调度状态缺陷：连续错误触发停止派发后，在途请求成功会清零错误计数，最终可能把暂停写成 `realtime_pass_finished`。现将暂停原因锁存到排空结束；不改变连续三条错误暂停的边界。

新增显式 `--unattempted-only` 恢复模式：跳过已完成、已有明确拒绝记录以及已延期案例，不修改任何旧 receipt，不重置重试计数；遇到不确定请求、未恢复的原始响应或异常状态即拒绝启动。调用方先诊断旧失败，再将未尝试部分继续派发。旧 10 条重试用尽案例仍未完成，后续必须单独核查，不能通过重启偷偷增加次数或从最终分母删除。

新部署入口为 `/volume/ybo/wza/training-artifacts/run_realtime_agent_judges_resume_20260915.py`，SHA256 为 `db5b6dba0d2a7fec173244eb2bac930171df5f4aa52554a462f1093cfdb327b7`；原服务器脚本保留未覆盖。23:37 使用 `--stage run --workers 2 --unattempted-only` 启动 PID 958662（仅作线索，监控须查真实进程）。输入、所有权、输出目录、模型及完整 v3 审核协议不变，只有并发从 4 降为 2。

续跑选择清单为输出目录的 `resume-selection-*.json`，日志为 `resume-unattempted-20260915.log`，当前状态仍读 `process.json` / `progress.json`。清单记录跳过 218 条成功和 10 条已拒绝，选择 2,675 条未尝试案例；首批已新增 2 条有效 judge，证明实际恢复产出，但 503 仍偶发，不能承诺已解决服务繁忙。原结果没有重跑，28 个 Batch 没有转移所有权。

15 项本地定向测试及编译、diff 检查通过；新增测试覆盖旧文件字节不变、不确定状态拒绝、原始响应恢复门槛和暂停原因锁存。小时监控改用新入口，并保留当前重试及预算边界。H20 主推理、GPU 守护、朋友实验未因本次修复停止或修改。

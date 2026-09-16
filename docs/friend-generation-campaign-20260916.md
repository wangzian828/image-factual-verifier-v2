# 朋友436图实验：失败补跑及独立Pro组

2026-09-16用户明确要求失败也补跑，同时运行Gemini3.1ProPreview。这里只记录运行协议和代码，不复制朋友的图片、PPTX、私有轨迹或密钥到Git。

## 实际状态与调度

- 14:42原Gemini3.6Flash已尝试279/436：260成功、19失败、157未尝试。旧控制器952726/952728为线索，以`state/fallback-keep-going-controller-20260915.json`与实际命令为准，原运行不改。
- 新Flash补跑控制器1063897在等待同一`flash-controller.lock`。原436条全部尝试并排空后自动接手所有未成功项，已有成功不重跑。不是这19条已经补完。
- 独立Gemini3.1ProPreview控制器1063898已真实启动；首条全Agent通过模型、原图、工具返回、输出预算、思考配置和PPTX结构检查，自动进入余下435条。不是只做API ping。
- 两模型各串行1，总并发最多2；Flash原进程和补跑队列共享锁，不能并发抢同一组。此跨模型并发差异依据本次“一起跑”授权记录，未恢复此前4路样本并发。
- 每次完整尝试仍900秒，原runner的503最多一次重试/60秒退避不变。新的失败补跑写独立wave目录，不覆盖旧attempt。每次campaign最多三轮额外补跑，仍失败则进入诊断状态，由小时任务继续处理；不把“尝试完”冒称全部成功，不无限刷同一失败消耗费用。

## 模型/档位实际映射

交接配置写的是`gemini-3.1-pro-preview`和CLI档位`gemini-3.1-pro-high`。对安装的AGY1.2.3做**零外部调用**本地拦截，发现该档位实际请求`gemini-3.1-pro-preview-customtools`，`maxOutputTokens=65535`，`thinkingConfig={includeThoughts:true,thinkingBudget:-1}`；不是Flash的4000/65536。

新可信父进程仅将这个精确customtools路径映射回用户指定的普通`gemini-3.1-pro-preview`，请求正文、prompt、工具、thought signatures均原样转发。其他模型仍被精确allowlist拒绝，不开放CLI的辅助Flash-Lite请求。wire记录实际转发模型以及alias适配事件。原Flash broker/源代码完全未改。

high在此CLI中实际编码为动态预算-1，不应声称wire显式发了`thinkingLevel=high`。[Google说明](https://ai.google.dev/gemini-api/docs/generate-content/gemini-3)称Gemini3未指定thinking level时默认high；本次不强行修改原CLI正文，保留实际协议差异。输出65535与Flash65536的差异同样显式验收。

## 位置与验收

所有路径以下列ROOT为前缀：`/volume/ybo/wza/external-projects/wanglinhaotrain-20260915`。

- 只读旧runner：`project/src_linux_fallback`；原Flash结果`project/outputs/gemini-3.6-flash-medium-linux-full-20260915`。
- 新控制器：`friend_generation_campaign_v1.py --lane flash|pro --execute --thinking-json ...`。
- 进程receipt：`state/{flash,pro}-campaign-20260916-process.json`；外层日志同前缀`.log`。
- 新结果：`project/outputs/friend-{flash,pro}-campaign-20260916/`，含`identity.json`、输入绑定、`state.json`、`aggregate.json`、`config-wave-XX.json`、`wave-XX/`、`wire.jsonl`、`runner.log`。Pro另有`pro-canary.json`。
- 汇总按每个sample的首个结构成功结果去重，记录所选status路径/hash；不按质量择优。不同模型绝不合并统计。中断的running/retry_wait不自动跳过，需要明确恢复。
- 原图/固定prompt/工具/原尺寸单页可编辑PPTX生成任务不变；没有渲染、judge或视觉质量评价。结构成功不证明视觉质量。

新增调度8项单测通过；真实Pro首例七项验收均通过。小时监控按官方自动化更新流程纳入这两个队列，未新建重复任务。新增代码从本地提交，服务器不执行Git。

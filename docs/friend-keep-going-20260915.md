# 朋友 FigureEditBench 续跑调度更新

2026-09-15 晚用户明确要求 3.6 实验不要因失败停整批，尽量继续。原14条已尝试（8成功6失败）不重跑，继续未尝试422条。新独立控制器不再执行三次连续基础设施失败熔断；每条最多两次尝试和串行1保持，模型仍Gemini3.6Flash，原配置/提示/图片/工具/medium预算全部不变。

服务器项目根目录：`/volume/ybo/wza/external-projects/wanglinhaotrain-20260915`。

- 入口 `resume_full_keep_going.py --execute`。
- 新状态 `state/fallback-keep-going-controller-20260915.json`。
- 执行日志 `state/gemini-36-flash-keep-going-20260915.log`。
- 初始14条状态哈希、原run元数据及用户新策略保存在 `state/fallback-keep-going-authorization-20260915.json`。
- 结果继续原 `project/outputs/gemini-3.6-flash-medium-linux-full-20260915`，同一冻结config hash，不混模型。

新入口位于冻结 adapter 之外；未修改原 `src_linux_fallback`，不删除历史held记录、不把部分PPTX或API错误标成成功。利用原runner的终态跳过逻辑继续，保存每条真实结果；直到436条均已尝试，允许最终状态为completed_with_failures。凭据、磁盘或执行器损坏等无法正常落盘的故障仍需诊断。

6项定向测试通过，服务器预检确认14条终态/配置/pilot哈希有效、422未尝试。原小时监控通过官方OpenAI文档核对后同步为新状态路径与新策略。没有更换模型、增加并发或改变IFV/PSD/GPU任务。

运行代码及测试保存在本地 `work/friend-wanglinhaotrain-20260915`，隔离部署在上述服务器项目根目录。朋友项目源码未并入IFV仓库；本仓库仅记录操作交接，不包含其数据、凭据或产物。

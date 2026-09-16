# 感知工具跨图片缓存事件（2026-09-16）

## 已确认的问题和边界

06:39小时巡检发现PSD 4×8小样出现length和1200秒网关504。深入检查时，`main-05485`的调查主体为发布会，而`perceive_scene`却返回SunRail列车；不同图片的结果字符串SHA相同、`cache_hit=true`。

根因是冻结runtime的缓存身份缺陷，不是已经证明SFT权重损坏：`RuntimeToolAdapter`将公开schema里的`image_input`隐藏，实际调用时才通过`_provider_args()`绑定图片。`StageRunner._build_cache_args()`却只给compare/anomaly类工具补图，perception/OCR没有得到隐式图片身份。因此空参数`perceive_scene`可以跨图命中同一个键。`TOOL_CACHE_ENABLED=0`只关闭web缓存，独立的`PERCEPTION_CACHE_ENABLED`默认仍为1；上一版PSD启动器遗漏后者。

这与vLLM APC/混合状态缓存是两层不同问题。关闭APC不会关闭工具感知缓存；当前证据也不能证明所有length/超时都由感知缓存引起。

## 只读历史审计

`scripts/server/audit_perception_cache_identity.py`逐个核对冻结选中trace SHA、图片文件SHA和原始工具结果SHA。以“不同图片文件SHA收到同一结果且存在cache hit”筛查候选组；相同文件的正常重复使用不会入组。**不同文件SHA仍可能是同图重编码，不能仅凭该筛查逐条定性为语义错配。** 两个独立无缓存结果恰好相同也不计作缓存候选。

| 已审计来源 | 冻结轨迹数 | 感知/OCR调用数 | cache hit调用数 | 跨图结果组 | 命中跨图组的唯一case |
|---|---:|---:|---:|---:|---:|
| SFT epoch2/2056 | 1,524 | 2,745 | 1,247 | 8 | 1,246 |
| SFT epoch3/3084 | 1,526 | 2,578 | 1,206 | 7 | 1,205 |

这些数量是需进一步核实的候选范围，不等于已逐条目视确认错配、所有case最终标签错误，或每条轨迹同样程度受影响。原BAcc仍是对原输出正确计数，但不能单独解释为排除了工具错误的模型能力。其余Agent尚未审计，不能宣称安全；不将这一故障泛化到不走工具的Direct QA。

### 09:30补充：训练源只读核验与实际图片反例

只读扫描本次训练绑定的4,873条canonical源（4,872 train＋1 val），源trace/binding SHA无错误：8,824次感知/OCR调用，其中1,485次标记cache hit；407个跨文件SHA共享结果候选组涉及1,191个case。**1,191不是已确认污染条数**。冻结训练文件、index与media bindings哈希均与门禁记录一致，没有更改数据或权重。

直接查看两张原图确认了一组不能用“同一图片复用”解释的反例：`main-02736`是海边道路、人群与车辆（360×480）；`main-02766`是办公室内景与玻璃办公楼外景拼图（627×231）。两者`perceive_scene`均为空参数、cache_hit=true，却得到完全相同的4,571字符结果，描述日食状太阳、白色曲面建筑及玻璃穹顶，与两张原图均不相符。原图SHA分别为`1e8437cb5211eb9fa1848f776343933ba4f77f9f7345e742a983d2eba48026c6`和`74d6e187caca4d789effce7d2d193b25d6d1279a87e61cd8c160abfd806c6c9c`，共享工具结果SHA为`c4417d801a7755877e0707ebf0696f3657167fb0dabe9cf70f107ba1b7f6d657`。源轨迹与图片都在initial交付包，不是本次查看生成的结果。

这确认至少存在真实内容错配，但不证明1,191条全部错误，也不等于模型训练已报废。先保持新PSD双缓存关闭，不擅自丢弃旧数据、重训或重跑全量评测。

证据在服务器`/volume/ybo/wza/evaluation/perception-cache-identity-audit-20260916`，包含完整组内case、图片和trace/result哈希；报告没有修改原轨迹、图片、gold、预测或已完成judge。两个选中索引SHA分别为`104bd65fec08f13966156beea4e012363d2ec02ddc67ec7a8f50066efcc210b0`、`28c8f296564180a7496e75afb793ea8ee803f6ba075978f7c0357e975937e388`。

## 隔离和修复验证

- 原PSD小样`/volume/ybo/wza/runs/psd-slate-canary4x8-20260916`已核对PID/完整命令/脚本SHA后发送SIGINT，进程正常退出，19012在途归零。24条已落盘轨迹（含失败）及未完成请求的runtime archive均保留，`source_bank_admissible=false`，不进入checker/目标或训练，不补跑它来挑成功结果。证据和停止收据在该目录`cache-contamination-hold`。
- 不修改冻结`src`、工具schema、prompt、权重或历史环境；新PSD诊断进程显式设置`TOOL_CACHE_ENABLED=0`及`PERCEPTION_CACHE_ENABLED=0`，并从实际Orchestrator断言`tool_cache.enabled=false`、`cacheable_tools=[]`。通用PSD隔离env也增加两个独立开关。
- 新两案例诊断`/volume/ybo/wza/runs/psd-real-runtime-gate-no-perception-cache-20260916`由`run_psd_runtime_gate_v4.py --cache-free`运行，同一固定输入，203.28秒和165.74秒，2/2有最终报告，感知/OCR全部cache miss，两张图返回不同感知结果。
- strict source-policy审查无failure，但有1条`PROTOCOL_CORRECTION`：重复visit被runtime拒绝后继续调查。原样保留，不声称零警告；它是已记录的模型动作纠正，不是把非法调用静默视为成功。现场`tool_llm_api_calls`未透传感知VLM用量，报告为0；不据此宣称没有VLM调用，也不把cache miss自身冒充独立wire级图像证明。`cache-identity-audit.json`明确`provider_subcall_attested=false`、`full_psd_gate_passed=false`。
- 接着只验证修正配置下同一固定4图×8小样，独立路径`/volume/ybo/wza/runs/psd-slate-canary4x8-no-perception-cache-20260916`，入口`run_psd_grouped_canary_v2.py --cache-free`。不改温度、seed、think、总输出、动作数、128K、多图或权重，不与已污染的小样拼接。所有后续PSD门槛仍需通过；400×8及正式优化器训练未启动。

## Judge与下一步权限

06:41，在识别跨图缺陷前按既有冷却/收据续交epoch2 Batch一次，第二次仍明确HTTP429、0任务受理、0case提交，原收据未重置。确认上游缺陷后，在其准备目录写入`data-quality-hold.json`；新的`submit_sft2056_judge_guarded_20260916.py`在启动及每片付费请求前检查该标记，现场验证在任何API初始化前拒绝提交。旧入口不得再使用；不能因为下一小时额度恢复就付费提交此批旧输入。

此前已授权且正在运行的epoch3/Pro普通judge与旧28个Batch未中断，不重发、不取消有在途的不确定请求；这些结果仍对应原冻结轨迹，只能带工程缺陷限定解释，不能包装成干净协议的最终对比。朋友实验和GPU守护继续。

历史大规模重跑及费用/对比口径需用户确认；本次没有擅自重跑旧测试集，没有改模型权重，也没有丢弃原证据。OpenAI Docs用于更新既有小时监控的阻断条件，保持原频率和重要变化才通知的设置，不建重复任务。技术测试覆盖缓存审计、judge提交阻断、固定分组、collection，以及两个独立缓存开关被环境默认值重新打开的回归防护。

## 07:01 续进状态

修正4×8小样已完成prepare并于07:01:40启动，`process.json`记录PID1005036；现场核验完整命令及两个实际进程环境开关均为0，manifest为4图32槽、8并发、同一epoch2模型。仅为诊断采样，不是已通过checker或正式PSD训练。

07:02普通judge汇总为epoch3有效1,178、Pro有效1,177，仍2 worker运行且连续错误计数0；这是未完成快照，不报最终SESR。朋友实验112成功、15失败、309未尝试，按既有keep-going继续。GPU守护进程995299身份已复核，原权重及全部checkpoint未动。

07:06，新小样已有1个无顶层错误的canonical trace，感知工具无缓存命中；网关8个在途，仍在采集，不能据此宣称32条全部完成。GPU2实测100%，其他三卡的守护近期各完成约13秒计算脉冲。旧Batch collector进程915205身份及输出路径核验，`collector-progress.json`显示28个RUNNING、0已回收、无错误；只有完成任务后才会产生`collector-state/batch-checkpoints/*.jsonl.gz`，该目录当前不存在不等于collector没运行。32项定向测试、受影响脚本编译及diff检查通过。

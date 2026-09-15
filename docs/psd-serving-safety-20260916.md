# PSD 隔离推理安全候选（2026-09-16）

此候选只服务后续 PSD 验收，没有部署进 epoch2 测试集推理。该评测及其有限重试现已结束，保留 1,524 成功、2 失败及 1 缺图。主 Agent、旧网关源码、旧请求协议、旧模型权重均未修改。已结束的正式评测里 `thinking_token_budget=8192` 仍只是请求值，不能追溯称为已生效。

## 预算控制

vLLM 0.18.1 支持通过自定义 logits processor 和 `vllm_xargs` 接入逐请求参数，无需升级或改写安装包。[版本对应文档](https://github.com/vllm-project/vllm/blob/v0.18.1/docs/features/custom_logitsprocs.md)

新增 `scripts/server/psd_qwen_thinking.py`，插件入口为 `scripts.server.psd_qwen_thinking:PSDThinkingBudget`，启动参数为 `--logits-processors`。隔离网关将原来的显式 `thinking_token_budget` 映射为 `vllm_xargs.ifv_thinking_budget`；不修改 messages、工具 schema、图片字节、采样温度、seed 或输出总额度。

插件从所绑定模型的 `tokenizer.json` 读取 token ID，并验证真实 assistant generation prefix；不硬编码其他 Qwen 版本的 token ID。检查过 epoch2 导出 tokenizer：`<think>` 为 248068、`</think>` 为 248069。它只统计当前助手回合实际新生成的思考 token，达到预算时强制生成结束思考标记，不强制 EOS；正常提前结束思考后不再干预答案。调用方总输出预算必须为结束标记和答案留出空间，不把整个响应硬截断冒充思考预算。

逐请求状态跟随 vLLM 的移除、添加、移动及交换操作；保留引擎提供的输出 token 列表引用。实现按公开接口约定处理 slot 重用，不使用历史轨迹里上一轮思考的长度作为当前预算。[v0.18.1 接口约定](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/v1/sample/logits_processor/interface.py)

未知生成前缀、缺少精确 prompt token、非法预算、输出额度不足及 speculative decoding 均拒绝，不静默降级。上限 8192；更小预算只用于隔离边界探针，正式 PSD 不因此缩短预算。

## 网关与超时链

新增 `scripts/server/psd_qwen_gateway.py`，保留 least-inflight 分配，不替换旧网关。只运行一个 uvicorn worker。

- 每个请求只派发一次；连接异常、读超时、HTTP 错误均不自动重发 POST。
- 客户端断开、网关总 deadline 到达或外层取消时，取消并等待上游 HTTP 任务收尾，再释放副本计数。
- HTTP 响应 body 保持原样；不把一次不确定生成改成另一个副本上的隐式重跑。
- 健康端点明确标记 `gpu_boundary_validated=false`；候选启动或 CPU 测试成功均不等于 GPU 验收通过。

专用进程配置为 `training/configs/psd/isolated-serving-safety.env`：网关总 deadline 1200 秒、模型客户端 1230 秒、阶段执行器 1260 秒，客户端重试为 0。配置要求同时应用到隔离网关及 PSD Agent 进程；仅设置网关环境不能改变已经启动的 Agent。网关启动检查这些值的严格大小关系。该配置没有缩短 action 总数、图片或上下文长度。

上游 HTTP 连接关闭不等于已证明 vLLM 的 GPU 请求立即释放。后续必须检查实际引擎 running/waiting 与请求取消，不能用 mock 或 socket 检查代替这一步。

## 隔离位置与验证边界

服务器候选目录：`/volume/ybo/wza/training-artifacts/psd-serving-safety-20260916`。只上传新增代码、测试、非敏感配置和供导入的旧网关副本；未覆盖 `/volume/ybo/wza/image-factual-verifier-v2` 或原 PSD 准备快照，未修改现有 Python 环境。

本地只运行不依赖 FastAPI 的 11 项状态机测试并通过；本地 Python 缺少 FastAPI，完整测试改在服务器已有依赖环境中运行。服务器通过 `CUDA_VISIBLE_DEVICES=''` 禁用 GPU，在 vLLM 0.18.1 环境加载真实 tokenizer、SamplingParams 与 BatchUpdate，并使用 CPU tensor 测试 1/3/8192 token 边界。pytest 从已有训练环境的 site-packages 追加解析，vLLM 和 torch 优先使用推理环境，未安装或升级软件。

网络测试使用临时 loopback HTTP 服务，验证响应头已经返回、body 尚未完成时，deadline 会关闭上游连接且只收到一次请求；不调用 Gemini，也不对正在运行的模型发送诊断请求。首次该测试的 0.15 秒 deadline 在冷启动 HTTP backend 初始化时提前到期、尚未发送请求，因此测试失败；已调整为 5 秒并显式断言响应头确已发出，没有降低生产超时/取消验收要求。

最终服务器定向回归 **35 passed，19.05 秒**：包含新状态机/网关测试、真实 vLLM CPU tensor 测试与旧网关 3 项回归。真实 vLLM 请求转换已确认 `vllm_xargs` 到达 `SamplingParams.extra_args`，而非被 API 当作无效 extra 接收。未加载模型权重，未执行新 GPU 生成；后面的 gate 仍全部待验证。

## 尚未放行的 GPU gate

1. 当前 epoch2 推理及其有限补跑结束，核验请求排空；按原计划先停 guard，再交接卡。
2. 绑定明确的 PSD 策略 epoch、权重及代码 SHA，启动隔离服务加载插件；原测试结果和旧服务配置留存。
3. 真正模型生成测到思考边界，确认 `</think>` 后仍能生成 native tool 或最终答案；包含多图、并发、提前结束、正式 8192 上限。保存实际 token 和请求参数，不能只观察字符数。
4. 实测 HTTP 超时/客户端取消后 GPU 请求能释放、没有第二次 POST；同步核对 Agent、客户端、网关三个时间参数。
5. 以上通过再走新多位置 PSD 真实 canary、top20、GPU 更新和 native 保存/恢复 gate，全部通过才采 400×8。CPU 验证完成不自动放行 3200 条采样。

## 03:47 后的实际交接与 GPU 验证

`freeze_sft2056_budgeted.py` 逐条核验 1,524 个选中 trace 的 SHA、case、有效终态和冻结 gold，结果单独保存在 `/volume/ybo/wza/evaluation/qwen35-sft2056-epoch2-agent-budgeted1524-20260916`，不复制原始图片或轨迹。`transition_to_psd_probe.py` 先只读预检 PID/进程组、导出 SHA、终态及空闲锁，再停止明确归属的旧 guard、核实队列排空、正常停止旧服务。旧权重及所有结果均保留。

新服务目录为 `/volume/ybo/wza/inference/psd-sft2056-safety-20260916`，四卡 TP1、vLLM 0.18.1、BF16、128K、多图 32、同一 epoch2 权重，网关端口 19001，后端 19002–19005。新的 guard 进程记录是该目录根下的 `guard.json`，采样在 `idle-guard`，不能继续读取旧服务目录来判断当前守护是否运行。

初版真实生成探针保存在 `generation-probes-v1`：机械强制边界的 1/3/8192 均在对应位置生成 `</think>`，8192 探针共生成 8205 token 并正常 STOP；正常提前结束探针在 50 token 关闭思考并 STOP。3-token 极端机械探针在关闭后重复答案至 length，因此只说明边界和后续 token 放行，不当作正常 Agent 行为通过。

同轮双图 native-tool 探针失败：19003 副本输出 32767 个 token 0（`!`），加上强制的一个结束思考 token，最终 length。该失败没有被藏掉，也没有用于放行采样。

### 短输入缓存复用异常及隔离尝试（未证明解决长输入）

对照证据保存在 `multimodal-diagnosis-v1` 和 `multimodal-diagnosis-v2`：另一副本同输入有/无预算均能调用工具；在故障的同一 19003 副本上，关闭预算仍输出全 token 0，开启预算也失败。只加入独立 `cache_salt`、保持图像/文本/工具/采样不变后，正常输出 412 token、两图描述和一次 native tool call。因而已定位到前缀缓存复用这条路径；尚未用中间 tensor 证明具体 NaN 来源，不能把上游别版本 issue 当成已确认的内核根因。

[上游相关报告](https://github.com/vllm-project/vllm/issues/55766)描述了 hybrid GDN/align 前缀命中后从首 token 输出 `!`、隔离 cache salt 后恢复的类似症状，但其版本为 0.28.0、模型及硬件不同。这里只将其作为诊断线索，没有升级 vLLM 或套用未经验证的内核补丁。

隔离网关 v2 为每次请求生成不同 `cache_salt`，绕开跨请求前缀复用，不改变消息、图像字节、工具、思考预算、采样或权重；代价是当前 PSD 请求不能享受之前的跨请求 APC 加速，不能继续引用热缓存 511–515 token/s 作为这一配置吞吐。新网关源码镜像在 `/volume/ybo/wza/training-artifacts/psd-serving-safety-20260916/gateway-v2`，旧候选保持原样。只重启了已排空的隔离网关，四个模型后端未重载；`gateway-before-cache-isolation.json` 保存原进程记录。

新增缓存隔离和派发计数测试。首次回归因 pytest 将旧回归文件所在目录前置，实际导入了旧网关，出现两项失败；改用 `--import-mode=importlib` 并断言真实模块路径后，**36 passed，13.48 秒**，确认验证的是 gateway-v2，不是通过修改断言忽略缓存检查。

### 真实取消链路

初版/第二版测试的 `request_success_total{finished_reason="abort"}` 断言均未通过，虽然 GPU 队列和网关在途数已经归零；延长 30 秒仍不递增，不能把这个计数当作客户端断连的可靠完成信号。原始失败分别保存在 `cancellation-probes-v1/v2`。

v3 使用网关在每次实际派发前递增的 POST 计数，并同时观察 GPU 进入 running、取消后 running/waiting 清零、网关 reservation 释放以及随后持续无排队。正式 19001 网关的客户端断连测试、独立 19011 网关的 5 秒 deadline 测试均通过：各派发一次 POST、GPU 释放；后者返回 504。正式网关的 1200/1230/1260 超时没有缩短。证据在 `cancellation-probes-v3`；临时测试网关已正常停止，guard 已恢复。该结果不等于 PSD 真实 canary 或 GPU 训练已通过。

缓存隔离后的混合 64/8192 预算双图四副本复验由 `probe_psd_cache_isolation.py` 执行，产物 `cache-isolation-probes-v2`。64 只用于边界诊断，正式仍为 8192；真实 PSD 多位置 canary、top20、更新及 native 保存恢复仍须后续逐项验证，不得从这些合成协议探针推断全流程成功。

这 8 个请求已全部返回，覆盖四副本，均不再出现 token 0 连发，耗时 2.4–5.0 秒；这是同一双图协议探针的耗时，不是完整 Agent 的加速倍数。但严格参数检查仅 2/8 通过：4 个正式 8192 预算探针中 2 个通过，4 个强制 64 预算探针均未通过。失败包括缺少 native call、数组字符串损坏，以及数组末尾多逗号。

已直接对比原始生成 token 与解析结果：例如 `descriptions` 原文为 `["...", "..."],`，不是合法 JSON；vLLM 0.18.1 的 qwen3_coder parser 在 JSON 失败后执行 `ast.literal_eval`，得到包含一个数组的 tuple，序列化成嵌套数组，违反期望的 `array[string]`。该结果是模型原始格式错误与 parser fallback 的共同表现，不能靠偷偷展开数组来宣称成功。也不能因为这种人工新工具探针失败，就断言原有 Agent 的所有工具都失效。

当时结论：这组短输入的缓存隔离回归和真实取消链路通过；完整多图 native-tool/PSD gate 仍未通过。后续长输入再次失败，见下方记录，不能把这一局部结果描述为缓存异常全部解决。400×8 尚未开始。

## 04:44 真实 Agent 两案例诊断已启动

入口 `scripts/server/run_psd_runtime_gate.py` 从已冻结 32 条训练 canary 的前两条取固定案例，不按结果挑样，使用未修改的真实 Agent 工具、prompt、图片及 24-action 原生预算，T=0.7、think8192、输出32768、并发2、每例1条、seed0。每例一条仅用于运行协议诊断，不进入 400×8 的源轨迹银行，也不以此代替原方法的八次采样。

新运行目录 `/volume/ybo/wza/runs/psd-real-runtime-gate-20260916-v2`，入口在服务器代码根 `run_psd_runtime_gate_v2.py`，启动 PID 990475 只是线索，检查时须核验 `process.json` 和真实进程。运行前逐文件比较隔离快照 `src` 与冻结 Agent 的规范化文本一致，原始字节差异来自 CRLF；所有源文件 SHA、导出 SHA、选中 case、凭据存在性（不含值）均单独记录。未调用服务器 Git，也未加载私有 gold。

发现一个配置陷阱并在启动器里规避：冻结 runtime 通过模型名称是否含 `qwen3.5` 选择专用采样参数，因此不能直接用不带该字符串的隔离后端 alias。Agent 保持公开策略名 `ifv-qwen3.5-9b-sft-2056`，网关按既有固定映射转到同一导出的 `ifv-psd-sft2056-safety`；不是更换权重。`effective-stage-config.json` 已确认两阶段实际配置都是 T0.7、top_p0.95、top_k20、presence_penalty1.5、think8192，模型/阶段超时仍1230/1260，重试0。原人工工具探针没有完整使用这组采样参数，也不因此修改或删除旧失败记录。

第一次启动因同时指定 provider profile 与独立 `--model` 被参数校验拒绝，尚未进入 Agent。已保留其失败目录，v2 去掉冲突 CLI 项并由 profile 环境变量传入策略名，随后正常创建真实 run manifest，当前在执行。该启动不表示工具验收、PSD checker/repair、top20 或训练已通过；完成后必须审查 canonical trace 与真实外部 subcall。

### 两条真实诊断的完成结果

两条均已结束，但没有有效最终报告。第一条耗时40.35秒、4个工具调用，最后收到 `finish_reason=tool_calls` 却没有可用调用；第二条耗时417.19秒、12个工具调用，最后是 `finish_reason=length`、无可用答案。因此真实运行 gate 仍未通过，不能把它们作为成功 preservation 或把服务标记为完全验收。

已逐项解析 canonical `state.all_steps`：16个已完成工具结果均为 `status=success`，涉及 scene perception、OCR、图搜、文本搜索、反搜、访问网页和视觉检查。实际子调用包括 OCR3次、图搜1次、文本查询7次、候选重排7次、上传/反搜/页面获取/提取各1次。不是“没注入外部凭据”的重现。

原 API backend 对“不可用响应”只留错误摘要，没有把该响应体写入 context ledger，不能据此臆测原始 token 或擅自修正格式。在 `failed-request-diagnostic` 对第一条的 `req-000005` 做了一次独立的归档请求诊断，要求返回 token IDs；它不是重跑完整轨迹，不计入任何评测/训练结果。归档 JSON 字符串可能规范化，因此明确不声称 exact-token 重放。该请求的 SSH 输出连接中断，但服务器最终写出了 `response.json`：32768 个 token，其中 32767 个是 token 0（`!`），另一个为强制 `</think>`，最终 length、无工具。请求经过 unique-salt 网关，故此前短输入的缓存隔离不足以解决长上下文；未因 SSH 断开重复发送同一请求。

## 05:07 单卡 cache-off 对照

`probe_psd_no_apc_backend.py` 核实归属与空队列后，只重启 GPU2/19004，将 APC 关闭、mamba cache mode 从 align 改为 none，其余权重、BF16、128K、多图32、vLLM版本不变。其他三副本未重启。旧后端/guard 记录保存在 `no-apc-backend-probe-v1`，新后端 PID992562 为线索，以 `replica-2.json` 实查为准；当前隔离服务是混合配置，不能当作四卡同配置的正式服务。

该次只诊断首 token：使用同一个归档上下文，关闭预算插件的逐请求开关、最大输出256，两副本都正常返回 native tool，未出现 token0：cache-off GPU2 139token/13.78秒，cache-on GPU1 229token/3.22秒。这不是完整 Agent 成功，也不能据此认定关缓存是唯一修复：重启/副本状态及请求预算均可能影响结果。未用不同输出长度比较性能。测试结束 guard 已恢复。

下一组 `archived-context-budget-ab-v2` 在这两副本上各重复两次，保留真实8192预算、32768输出、Qwen采样参数及 unique salt，同时检查真实工具 schema。v1 因训练 Python 环境缺少 jsonschema 在发出任何生成请求前失败，guard 已自动恢复；v2 改用已经具备依赖的 vLLM Python 环境，无安装/升级，并在暂停 guard 前验证依赖。结果必须逐项读取，不以诊断启动代替通过；真实多位置PSD、top20、GPU更新及保存恢复仍未完成。

05:20结果：cache-off GPU2两次均139token、在108关闭think、正常native tool且schema通过，耗时2.44/2.43秒；cache-on GPU1第一次32767个token0加endthink、length，耗时250.61秒，第二次251token但工具参数带空字段名、schema失败。两种失败均保留，不删除字段、改写参数或把快速返回算成成功。guard已恢复（995299仅作进程线索）。这支持优先测试cache-off配置，但仍不足以唯一确认底层内核根因，或证明所有完整轨迹都正常。

完整Agent后续测试仍用同样两个固定训练canary和相同采样协议，只通过独立loopback网关19012定向到cache-off 19004副本。网关源码仍是gateway-v2，保留1200/1230/1260和禁止POST重放；原四副本网关19001未改。入口`run_psd_runtime_gate.py --single-no-apc --launch`（服务器版本`run_psd_runtime_gate_v3.py`），独立目录`/volume/ybo/wza/runs/psd-real-runtime-gate-no-apc-20260916`，所有旧失败与权重保留。是否通过必须检查新目录的完整轨迹和外部调用，不能将归档单请求的2/2通过直接算作完整Agent通过。

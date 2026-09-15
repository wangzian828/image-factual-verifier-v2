# PSD 隔离推理安全候选（2026-09-16）

此候选只服务后续 PSD 验收，未部署进正在运行的 epoch2 测试集推理。主 Agent、旧网关、旧请求协议、旧模型权重均未修改。当前正式评测里的 `thinking_token_budget=8192` 仍只是请求值，不能追溯称为已生效。

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

# PSD 首 token 无效概率：执行路径诊断

这是推理工程诊断，不是 PSD 优化 loss 异常。测试返回的动作不执行，输出不进入目标银行。
原三轮 SFT epoch3/step3084 权重及完整 optimizer/RNG checkpoint 不修改。

## 已核实环境与请求

- H20 四个独立 TP1 副本；vLLM 0.18.1、torch 2.10.0、transformers 4.57.6、triton 3.6.0。
- 加载合并后的完整模型，BF16；`quantization=None`、未启用动态 LoRA、`speculative_config=None`。
- 共同参数：context 131072、memory 0.94、max-num-seqs 16、max-num-batched-tokens 32768、throughput、GDN prefill triton、reasoning parser qwen3、单工具 parser、xgrammar、多图32、mm-processor-cache-gb0、APC关闭、mamba-cache-mode none。
- 请求保留原图片、14个工具、required tool choice、T0.7、think8192/out32768、原生 token IDs 与 top20。仅为诊断改为 streaming 并加唯一 cache salt；是存档重建，不是原始 wire 字节重放。
- 去除 salt 后请求 SHA256：`39bd97d1002739ea98120e3ae7adc15c66bdf9e053cfb09ca1e732eecbeb79b9`。已核对初次坏请求、eager A/B 坏请求、新四模式测试一致。

## 异常的准确含义

初次 8 请求中 1 次、随后原配置 16 请求中 1 次，从首个生成 token 即无效：

```json
{"token":"token_id:0","logprob":null,
 "top20_ids":[0,19,18,16,17,1,2,3,11,10,8,9,13,12,14,15,7,6,4,5],
 "top20_logprobs":"全部 null"}
```

正常对照首步 top-3：760=-0.6804068685，40=-1.4304068089，9764=-2.0554068089。
因此不是只比较随机抽中 token 的概率，也不是两个近似候选换序。
同类非流式原请求有 `Out of range float values are not JSON compliant: nan` 错误；
流式 null 本身不能区分 NaN 与 Inf，更不等于已抓到 softmax 前 logits 或定位到具体算子。
检测器立刻关闭诊断连接，不将 null 置零、忽略 top20、或者把异常当有效训练数据。

## 执行模式与版本陷阱

|GPU|torch.compile|CUDA Graph|显式变化|
|---|---|---|---|
|0|原 VLLM_COMPILE|原 FULL_AND_PIECEWISE|无，基线|
|1|VLLM_COMPILE (3)|NONE|`--compilation-config '{"mode":3,"cudagraph_mode":"NONE"}'`|
|2|NONE (0)|FULL_DECODE_ONLY|`--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`|
|3|NONE (0)|NONE|`--enforce-eager`|

安装版本 `vllm/config/vllm.py:808–878` 明确：enforce_eager 同时关闭两者；mode0 与默认 piecewise 图不兼容时会自动将图改 NONE。
所以不能只传 mode0 就声称保留了原图模式。GPU2 显式使用支持的 decode-only 图。
这也意味着该四组不是完全正交的因子实验：GPU0 的图模式与 GPU2 不同，关闭编译还改变 custom ops 的默认路径；不能只凭这一组把责任定给编译器。
参考 [0.18.1 故障排查](https://docs.vllm.ai/en/v0.18.1/usage/troubleshooting/)、[安装版本源码](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/config/vllm.py)。

## 统计边界与后续

17:29 实测更新：两批均已完成，每组共100次，第二批前四副本逐卡以相同启动命令重新启动，编译缓存未清理。

|配置|首 token 全null|非tool的stop|其余tool完成|
|---|---:|---:|---:|
|原编译+图|1/100|2/100|97/100|
|编译、无图|1/100|0/100|99/100|
|无编译、decode-only图|0/100|0/100|100/100|
|eager，两者均关|0/100|0/100|100/100|

第一批每组50次全部tool完成；异常均在重启后的第二批出现。两次全null仍从首个token开始；
原配置另两次虽然logprob有限，但首步分布明显偏离常见top20，并以stop而非tool_calls结束，不计作Agent成功。
结果支持优先排查编译及其伴随custom-op/fusion路径，**不证明具体编译器bug或永久修复**。
仅关闭图仍能出错；关闭编译+保留decode图值得进入真实repair候选验证。
随后四卡40路混合长短请求两波及回到固定输入的控制请求共96次完成，每卡24次：原配置1次首token全null、保留编译但关闭图1次；无编译decode图及eager均24/24工具调用完成。

固定100次还比较了同一个token760：eager的logprob全为-0.78991；其他路径存在波动（不能把所有波动都当错误）。
两种无编译候选均未复现NaN，但先选**纯eager**进入真实repair验证，继续保留其他模式证据，不宣称已定位具体编译算子。
临时候选控制器为`promote_psd_execution_candidate.py --kind eager`，逐卡核验身份/排空后修改执行模式，原始checkpoint、Agent、输入和预算均不变；这不是永久根因结论，也不是正式训练放行。
网关将启用有限2GiB、单响应64MiB的原始wire诊断，保留实际非流式repair请求字节，弥补存档重建不是原wire的限制；不记录密钥header。
恢复脚本`resume_psd_pending_cases.py`只续原3个pending案例，保持6/12预算与已经完成的两轮调查/审核哈希；该脚本须等candidate状态就绪才允许启动。
新增恢复和诊断在本地共57项定向回归通过；真实repair、目标、9B优化及恢复验证仍未通过，不把脚本已写当作训练成功。

旧 A/B 是原配置1/16、eager0/16，不是永久修复证明。上述扩展诊断已完成两批每组50次固定请求（每卡并发2），两批之间同参数逐卡重启。
每组都绑定启动命令与进程 receipt，记录首步完整top20和规范化请求hash。
原配置也可能一段时间不复现，因此需要考虑请求历史/调度状态，不能简单以候选零失败宣告成功。
已覆盖混合长度和两道不同图片任务的请求历史；多图完整 Agent 仍需真实验证。这些是诊断，不扩大训练采样预算。

17:38：四卡 eager 候选已全部就绪，网关原始 wire 捕获已启用。仍只放行原3个pending案例的工程恢复，尚未开始3200条正式采集或PSD优化。
固定请求正常完成部分的耗时中位数：原配置6.71秒、仅关图7.10秒、无编译decode图5.07秒、纯eager6.69秒。
输出长度、调度和服务历史未严格匹配，不能当作正式吞吐量或加速倍数。关闭CUDA Graph主要减少CPU启动开销优化，性能影响取决于实际瓶颈；不是退回CPU，也不改变BF16或权重。
参考 [PyTorch CUDA Graph说明](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)。

服务根：`/volume/ybo/wza/inference/psd-sft3084-20260916`。
初次坏请求：`nan-stream-sweep-v1/wave1-gpu3-copy0`；第二次：`nan-stream-eager-ab-v1/wave4-gpu2-copy0`。
新矩阵：`nan-execution-matrix-part1-v1`；重启控制器：`nan-execution-matrix-restarts-v1`；第二批：`nan-execution-matrix-part2-v1`；混合请求：`nan-mixed-history-v1`。

[上游 #52568](https://github.com/vllm-project/vllm/issues/52568)涉及动态LoRA，不能直接套用本任务。
编译缓存、前缀缓存、线性注意力状态是不同层次；本次已在APC关闭、mamba none下复现，不把“关前缀缓存”重述成新修复。
诊断期间自动化保持 PAUSED；当前任务现场持续处理。

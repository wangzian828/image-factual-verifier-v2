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

旧 A/B 是原配置1/16、eager0/16，不是永久修复证明。当前加做每组50次固定请求（每卡并发2），随后同参数逐卡重启，再做第二批50次。
每组都绑定启动命令与进程 receipt，记录首步完整top20和规范化请求hash。
原配置也可能一段时间不复现，因此需要考虑请求历史/调度状态，不能简单以候选零失败宣告成功。
若固定请求对照不再触发基线，需另外覆盖混合长度/多图的真实请求历史；这些是诊断，不扩大训练采样预算。

服务根：`/volume/ybo/wza/inference/psd-sft3084-20260916`。
初次坏请求：`nan-stream-sweep-v1/wave1-gpu3-copy0`；第二次：`nan-stream-eager-ab-v1/wave4-gpu2-copy0`。
新矩阵：`nan-execution-matrix-part1-v1`；重启控制器：`nan-execution-matrix-restarts-v1`；第二批预定：`nan-execution-matrix-part2-v1`。

[上游 #52568](https://github.com/vllm-project/vllm/issues/52568)涉及动态LoRA，不能直接套用本任务。
编译缓存、前缀缓存、线性注意力状态是不同层次；本次已在APC关闭、mamba none下复现，不把“关前缀缓存”重述成新修复。
诊断期间自动化保持 PAUSED；当前任务现场持续处理。

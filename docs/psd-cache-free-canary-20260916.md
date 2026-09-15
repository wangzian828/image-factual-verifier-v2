# 无感知缓存 PSD 小样验收（2026-09-16 07:39 巡检）

## 结论：尚未通过，不能启动 400×8

固定4图×8共32槽全部结束，21条有最终报告，11条顶层错误：8条`finish_reason=length`且无可用content，2条ReadTimeout，1条最终fact-check报告契约失败。所有原轨迹、错误和runtime archive保留，不挑21条当整组通过、不修改模型或Agent，也未启动checker、修复目标、top20或PSD优化器。

结果根：`/volume/ybo/wza/runs/psd-slate-canary4x8-no-perception-cache-20260916`。50次感知/OCR调用均无cache hit，证明本轮没有复现上次工具缓存命中；不能据此声称工具判断全部正确或完整GPU安全门槛通过。APC也已在此单卡后端关闭，但剩余错误根因尚未确认。

## 独立的采集校验器错误

进程最后的`ValueError: PSD collection slot seed/index/policy binding mismatch`不是11个模型请求错误的成因。原生`src/eval/result_records.py::run_result_record`不写冗余`group_size`字段，旧PSD校验器却要求逐行存在，导致完整采集被误报。

仅修正训练侧校验器：仍按manifest生成完整预期槽位，严格比较case/episode/分组hash/index/seed，拒绝缺槽、重槽及不匹配的显式group_size。group hash本身已绑定group_size、policy、model、base seed；缺少冗余字段不等于缺少分组证明。使用真实原生writer构造回归输入，保留失败结果，未改冻结src或给历史行补造字段。

服务器独立加载`training-artifacts/psd-serving-safety-20260916/psd_collection_validator_v2.py`，只读验证32槽通过；原`state.json`的held和原ValueError日志保留。验证报告在下述串行诊断目录`collection-integrity.json`，同时明确`psd_gate_passed=false`。

## 原始输出回放

选取真实失败的`main-05485--d4337e0a--r001 / req-000015`，归档上下文含15个图像输入，重建后实际prompt 47,427 tokens。不裁剪图片、历史、工具schema或输出预算；不是声称已保存原请求的逐字节wire副本。冻结Agent在抛unusable-response异常前没有落盘完整raw response，因此原失败不能直接认定为NaN或token0重复。

串行诊断：`/volume/ybo/wza/inference/psd-sft2056-safety-20260916/cache-free-failure-diagnostic-v1`。同一重建输入，两个请求均保留T0.7和32768输出上限，仅对比8192预算插件开/关，增加return_token_ids作观测，均不执行返回的工具、不产生训练目标：

| 变体 | 总输出tokens | think闭合位置 | 耗时 | token0数 | 结束原因 |
|---|---:|---:|---:|---:|---|
| 预算8192开启 | 500 | 425 | 10.72秒 | 0 | tool_calls |
| 预算插件关闭对照 | 628 | 552 | 11.77秒 | 0 | tool_calls |

每条都返回一个native工具调用；未将其数量检查冒充完整schema/Agent验收。单次串行均正常不能排除负载相关、采样或内核问题，也不能证明插件无问题。

07:53启动独立8并发回放（4个预算开、4个关闭），新目录`SERVICE/cache-free-concurrency-diagnostic-v1`，入口`training-artifacts/psd-serving-safety-20260916/probe_psd_cache_free_failure_v2.py --concurrent`，PID1010443仅线索，以process.json/真实命令/summary为准。固定8请求不重试，只用当前GPU2，不重启后端、不改其余服务/guard；不把合成回放当8条完整Agent成功或速度承诺。

07:54该8请求全部结束：500–825输出tokens，61.51–66.80秒，全部tool_calls且各1个调用，think闭合位置425–584，token0数均0。未复现原故障，不足以锁定或排除并发、预算插件或引擎数值根因；原`length`响应未保存完整raw，不能用事后重建回放替代。下一步需要在隔离serving留存真实失败响应/请求关联，再作有限对照；不热改冻结Agent、不重复全量评测、不把同图回放当跨图正确性证明。

研究核对：[vLLM自定义logits processor接口](https://docs.vllm.ai/en/v0.18.0/features/custom_logitsprocs/)及现场0.18.1源码要求remove→add→move和live output list；当前插件顺序一致。[上游异步占位token问题](https://github.com/vllm-project/vllm/issues/52461)针对未被CLI计入的processor，我们显式CLI加载，不能仅凭相似现象归因。[上游GDN前缀缓存问题](https://github.com/vllm-project/vllm/issues/55766)的版本与配置也不同，且本轮APC已关闭，不能套用其根因；这些只作为排查线索，不升级vLLM、不更换模型。

## Judge当前尾项及GIF新边界

普通judge于07:47结束该轮未尝试队列，真实进程退出、submit.lock释放：epoch3有效1,440、Pro有效1,437，共2,877。未完成部分共175条：旧28个Batch负责149条，24条各已6次明确HTTP503失败，两源各1条共用GIF延后。不能恢复旧Batch submit、重置失败预算、把`realtime_pass_finished`解释为每源1526齐全，或报告最终SESR。

GIF SHA `72aed244f121f6599e0a863674d77e2d72ec800109bf5935a9c37f5d7c2f0078`，17,581,496 bytes，360×640、268帧。现在不能再仅归因File API配额：[Google官方支持的图像格式](https://ai.google.dev/gemini-api/docs/image-understanding#supported-image-formats)列PNG/JPEG/WEBP/HEIC/HEIF，未列GIF。只读File API抽查100个文件合计122,840,039 bytes，不能据此推断全部配额或已恢复；未上传、删除远端文件或新调这两条judge。不能偷偷只取第一帧、丢帧、改成视频或修改已有request hash；多帧输入口径需明确后再处理。24条耗尽重试也不无限拉起。

epoch2仍因跨图片缓存data-quality-hold禁止任何新judge提交。以上judge只对应原冻结输出，其缓存缺陷限定仍保留。小时监控继续，后续优先读取8并发诊断和Batch状态，不能因为仅修复了分组校验器就放行PSD。

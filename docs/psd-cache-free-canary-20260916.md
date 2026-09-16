# 无感知缓存 PSD 小样验收（2026-09-16 07:39 巡检）

## 09:52更新：真实原始响应已保留，仍有两条生成失败

### 10:10：复现工具数组循环，隔离候选已通过精确请求对照

`SERVICE/exact-wire-tool-choice-v1`的4次串行对照均有调用；不能由此宣称正常。接着`exact-wire-tool-choice-concurrent-v1`固定8并发（两原请求×required/auto×两份）复现1条length：`1-required-1`在token459已经`</think>`，之后反复生成完整`finish_investigation`对象并继续数组，最终32,768tokens、0个token0、无可解析完整数组，耗时301.26秒。另一条required生成了5个工具对象，API却只返回第一个。因此这次复现不是思考超8192，也不是全token0/NaN；不要推成旧11条错误均已同因确诊。

现场vLLM0.18.1源码证实：`tool_parsers/utils.py::_get_json_schema_from_tools`只设置`minItems=1`，没有上限；`entrypoints/openai/utils.py::maybe_filter_parallel_tool_calls`直到生成完成后才在`parallel_tool_calls=false`时截取第一条。若数组循环到length，required解析器会抑制完整JSON解析错误并清空content，表现就是客户端“无答案”。这是所见故障的具体机制，不是仅凭相似上游issue归因。

新增可选`psd_single_tool_parser.py`继承原Qwen3CoderToolParser，只在**required且明确parallel_tool_calls=false**时给原工具数组加`maxItems=1`；名称/参数schema、prompt、images、采样温度、8192/32768及auto/named/parallel模式不变。不把正式required偷偷换成auto，也不升级或原地patch安装包。使用官方[tool parser plugin入口](https://docs.vllm.ai/en/v0.18.1/features/tool_calling/#how-to-write-a-tool-parser-plugin)。9项真实vLLM请求对象/xgrammar编译与schema测试通过。

GPU1/19003以同epoch2 BF16/128K、多图、APC-off配置加载候选，原GPU2/19004保留对照。原GPU1启动receipt保存在`SERVICE/single-tool-backend-v1/backend-before.json`，新receipt在`SERVICE/replica-1.json`；guard995299只在排空切换时短暂暂停，随后恢复，没停其他后端、改权重或执行服务器Git。

候选`single-tool-backend-v1/exact-wire`的8并发对照全部tool_calls，295–729tokens、46.37–52.69秒、0 token0。启动完整轨迹前还会解码原始token序列，检查每条required确实仅有1个JSON对象，而非只看经截断的API字段。该结果支持修复这一复现故障，但不代表完整PSD已验收，也不是严谨吞吐量基准。

两组旧judge已于09:35收尾，不再按下方07:57快照重试；见[主表](evaluation-comparison-20260909.md)及[普通接口最终合并](judge-realtime-switch-20260915.md)。

新增独立诊断`/volume/ybo/wza/runs/psd-wire-diagnostic4x2-20260916`：固定原4图各2条、共8条完整Agent（这是诊断，不是把正式每图8条采样预算改为2）。同epoch2权重、T0.7/think8192/out32768/128K、多图及工具保持。09:49结束，6条完整报告、2条length错误；6条严格结构审计通过，1条重复visit纠正警告保留。未启动source checker、top20或PSD训练，不能挑6条视为整体验收。

捕获目录`/volume/ybo/wza/inference/psd-sft2056-safety-20260916/wire-diagnostic-v1/wire`保存全部131次请求/响应原文gzip与SHA，0归档失败、0未决，压缩body约146MiB。其中97个tool_calls、32个stop、2个length。两个异常分别为prompt27,575/30,395tokens，均生成满32,768tokens、content为空、tool_calls为空；返回思考仅1,487/2,096字符。由此**不能再仅归因思考过长**，也不能仅凭API解析后的响应认定NaN/token0。上下文、工具参数和图片原字节均有原始请求可查，不是事后重建。

发现上一节07:54回放的明确局限：旧`probe_psd_cache_free_failure.py`重建时写死`tool_choice=auto`；新捕获的两条真实失败均为`required`。旧回放虽然保留历史/图片/采样预算，但不是该工具选择路径的等价对照。旧成功结果仍保留，不能据此排除required/结构化输出路径。

下一组仅4次GPU诊断：两份精确失败请求，各required/auto一次，均加`return_token_ids=true`观测；仅auto是诊断对照，**不修改正式Agent的required策略**。图片、messages、tools、seed、T0.7、8192/32768全部不变，串行一次派发、不自动重试、不执行返回工具、无PSD目标。入口`training-artifacts/psd-serving-safety-20260916/probe_psd_exact_wire_v1.py`，输出`SERVICE/exact-wire-tool-choice-v1`；09:53已启动PID1023190（只作定位，状态应重新读真实文件）。原8条失败/成功均不重写，不与本回放混作新训练样本。

新增`audit_psd_wire_capture.py`只读验证已提交receipt的解压字节及SHA、区分在途/归档失败/length/不可用response/工具参数JSON语法；不把此检查当完整tool schema、checker或PSD门槛。39项原定向回归＋4项wire audit＋5项精确请求差异测试通过；服务器38项serving/capture检查通过。

另外核对上游[H20动态LoRA问题](https://github.com/vllm-project/vllm/issues/52568)和[required与投机解码问题](https://github.com/vllm-project/vllm/issues/38106)：本机是全量BF16权重且未启用动态LoRA/投机解码，不直接套用这些故障结论，也不据此升级vLLM。

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

07:57尾项更新：旧Batch已回收2个任务、26个仍RUNNING，无collector错误。在确认普通runner退出和锁释放后执行只读API的结果合并，每源新增5条有效审核，总有效epoch3为1,445、Pro为1,442，仍不完整。两源尚缺165条：139条仍由旧Batch负责、24条重试耗尽、2条多帧GIF。朋友任务133成功16失败、287未尝试，仍keep-going；四卡守护近期均完成约13秒计算脉冲。39项定向回归、编译和diff检查通过，代码及主体记录已从本地提交推送`453f9c3`。

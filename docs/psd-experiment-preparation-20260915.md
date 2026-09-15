# PSD 实验准备：4,000 条统一训练输入

## 结论与位置

数据、源轨迹语义准入、真实审查诊断、固定抽样清单和启动计划已准备。没有启动全量 PSD，没有改动主 SFT / Gemini 任务、旧轨迹或权重。准备完成不等于新 SFT 上的 PSD 已训练成功。

- 输入：`/volume/ybo/wza/data/psd-candidate-pool-4000-20260914-v3`
- 计划与清单：`/volume/ybo/wza/runs/psd-experiment-preparation-4000-20260915-v1`
- 真实源审查：`/volume/ybo/wza/runs/psd-source-review-canary-20260915-r1`
- 隔离代码与测试：`/volume/ybo/wza/training-artifacts/psd-preparation-20260915`

服务器操作限于 `/volume/ybo/wza`；提交仅从本地 Windows 工作树进行。

## 数据口径

| 用途 | 数量 | real / fake | 训练边界 |
|---|---:|---:|---|
| PSD 输入池 | 4,000 | 1,421 / 2,579 | 只有后续准入的本轮目标才用于训练 |
| 固定训练 canary | 32 | 16 / 16 | 来自上述池，不额外剔除或留出 |
| 固定开发观察 | 400 | 200 / 200 | 来自正式测试集，只评测、不做梯度训练 |
| 正式测试 | 1,527 | 377 / 1,150 | 图可用 1,526；分母仍为 1,527 |

按固定 seed 在看到结果前抽样。开发清单只有 ID、test 标识和 `training_prohibited=true`，不复制测试图或 gold 到训练。它不是独立留出集。内部保留来源，论文统一称 4,000 条 PSD 训练样本。

## 修正与验证

- `review_psd_sources.py` 审完整原轨迹、成功/失败/空工具响应及归档多图；标签、strict audit、语义均通过才进 preservation。
- 正确标签但错误证据能进 repair，并贯穿完整 causal verifier / offline finalize。契约失败由绑定到原轨迹的 strict audit 证明，不假装是标签错误。
- 网络/解析异常和 unresolved 保持待处理；入口暂停，不伪造失败或静默反复抽 judge 直到通过。已完成响应、审查决定、原图/轨迹/参考、prompt/schema/model 均有绑定。
- 私有审查只用于准入，不进入 student 输入或 token target。原有 exact-token、多图、局部 loss、完整续跑、procedural hint 检查保留。
- `audit_psd_source_reviews.py` 离线检查绑定和路由；`prepare_psd_experiment_plan.py` 固定清单、资源与启动门槛。`run_psd_round.py prepare` 和新的 canary 已接入源审查。历史已完成 bank 保持不可变，不原地重写。

真实诊断使用原有 8 条训练轨迹，只新调用 Gemini 3.1 Pro Preview 源审查，不生成新 Qwen 轨迹：8/8 完成、无待处理；原标签正确 5/8，新审查 pass 2、fail 6，其中 3 条标签正确。它们均能通过新的修复源验证，源 run 未改变。离线审计新增 API 为 0。summary SHA-256：`e96abc2eae735c2e468018bee6c48c66358a0b6a04c61e413610cbaa98b94518`。

这不是 judge 准确率或 PSD 能力提升。失败理由包括证据与结论冲突、把反搜误匹配当作原图主张、OCR 归因不准确。最后一类可能涉及轻微重建与实质错误的边界，需在新 canary 中校准；不能把 judge fail 都宣称为人工确认错误。

## GPU 释放后的执行顺序

1. 完成主 SFT 保存、推理和正式评测顺序；不抢卡、不停服务。
2. 选择并实际导出/加载新 SFT checkpoint，绑定 serving profile 和 checkpoint manifest。计划 snapshot 目前故意为 null，不退回 Base 或旧 SFT。
3. 固定 32 条训练 canary 采新策略轨迹，验证工具实际执行、多图、原始 token capture、thinking/length 停止情况。
4. 源审查 → 错误定位 → procedural hint → 同一冻结策略续跑 → PSD 修复 judge → exact top-20 targets。没修好就不训练该失败目标，不删除输入。
5. 测真实 targets 的典型/最长长度；SP4 稳定配置作退路，DP4 作吞吐候选。同一有效 batch=32 比较 targets/s、监督 tokens/s、延迟、显存峰值、错误率，确认保存峰值后选最快的稳定配置。短样本结果不能充当 128K 容量证明。
6. 实做 GPU optimizer 更新、native full-state 保存、加载及下一步续跑。旧 CPU resume 和 H20 无保存一步测试不能替代此项。
7. 冻结本轮 snapshot，采完整 4,000 条输入，分阶段保存/retry，确认完整覆盖后构建准入 bank。使用按完成顺序的有界并发，不混用边更新边生成的过期策略目标。
8. 核实本用户配额或明确可用预算后，启动一轮 PSD。先按现有 recipe：5 epochs、LoRA r32、LR 4e-5、top20、128K。step 取决于准入 assistant-step targets，不是 4,000 张图直接除 batch；最终以真实 dataloader 为准。
9. PSD 后 smoke、固定开发观察、正式测试；报告 BAcc、SESR、完成率、工具错误、思考/输出截断、耗时和资源。最终评测 judge 使用大对比表冻结口径，与 PSD judge 分开。

## 可重复执行的准备命令

只生成/验证清单与计划，不启动 GPU 或外部 API：

```bash
python scripts/prepare_psd_experiment_plan.py \
  --pool /volume/ybo/wza/data/psd-candidate-pool-4000-20260914-v3 \
  --test-manifest /volume/ybo/wza/data/factcheck-test-1527-filtered-frozen-20260909/test-manifest.jsonl \
  --test-runtime /volume/ybo/wza/evaluation/factcheck-formal1527-available1526-20260912/runtime-release/runtime_input/cases.jsonl \
  --storage-root /volume/ybo/wza \
  --output /volume/ybo/wza/runs/psd-experiment-preparation-4000-20260915-v1
```

新 rollout 和实测 snapshot 就绪后按 `run_psd_round.py prepare --help` 准备。训练显式指定 `--training-data-root /volume/ybo/wza`，通过现有 idle-GPU、环境、target、初始化、resume、存储门槛。不能用假 snapshot 路径使文档显得可启动。

本次只新增代码、清单和文本审查缓存，不复制图片或权重。现有 8 GB checkpoint 配置只是估计，需考虑新旧两份峰值及至少 1.25 倍余量。共享卷剩余不是用户 quota。每小时监控保留，读取当前准备状态，不重复已完成审查或抽样。研究协议与资料核对见 `research/psd-preparation-20260915/`。

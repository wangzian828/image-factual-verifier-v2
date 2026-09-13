# SFT Canonical Release 工作流

当前版本（2026-09-13）的唯一训练发布链路：

```text
teacher rollout
  + strict trace audit
  + frozen SFT judge
  -> accepted-teacher-release
  -> accepted-dataset
  -> ms-swift policy/perception conversion
  -> real Qwen processor verification v4
  -> raw-data gate v3
  -> one-step weighted-loss canary
  -> swift sft
```

## 1. accepted release

`accepted-teacher-release/` 是训练前的冻结来源，至少包含：

- `accepted_release_manifest.json`
- `selected_episodes.jsonl`
- `traces/`
- `eligibility/`
- `runtime-stores/`
- `runtime_store_index.jsonl`
- 质量分桶和来源校验记录

它固定每条入选 episode 的 trace SHA-256、judge 结果、split、runtime commit 和
图像路径。后续导出只读这个 release，不回到原始 rollout 目录重新拼接数据。

原始 trace、拒绝项和 holdout 不删除；它们只是不进入本次训练输入。

`ifv-accepted-teacher-release-v4` 同时冻结每条入选轨迹的 runtime context
和 artifact store。原始 trace 字节及 `source_trace_sha256` 保持不变；
converter 只在内存中把 `state.runtime_store.runtime_path` 重绑定到 release
内副本，因此删除原 rollout 目录后仍可恢复初始图片、候选图片、裁剪图和聚焦图。

禁止从 `attempt/traces` 手工拼装 preview 或训练样本。10 条 smoke 的审阅产物
必须直接取自 `<smoke-pipeline>/accepted-release/trajectory_sft.jsonl` 或
`<smoke-pipeline>/sft-training-package/ms-swift-policy/`。

## 2. 生成训练数据

```powershell
python scripts/trajectory/export_dataset.py `
  --accepted-release <accepted-teacher-release> `
  --case-split <case_split.jsonl> `
  --output-dir <accepted-dataset>

cd training
python -m ifv_training convert-policy `
  --input <accepted-dataset> `
  --output <ms-swift-policy>

python -m ifv_training convert-accepted-perception `
  --input <accepted-dataset> `
  --output <ms-swift-perception>
```

如果接手的是已发布的 canonical raw-trace 归档，而不是仍可访问的
`accepted-teacher-release`，禁止从旧版 ms-swift JSONL 反推目标。应使用原始轨迹
提供 thought、工具调用、真实工具结果和最终答案，只从经过校验的旧交付包复用媒体
文件及 `<image>` 标记位置：

```powershell
python scripts/trajectory/export_canonical_trace_bundle.py `
  --raw-root <verified-raw-traces> `
  --reference-delivery <verified-reference-delivery> `
  --raw-archive <raw-traces.tar.gz> `
  --raw-archive-sha256 <published-sha256> `
  --reference-archive <reference-delivery.tar.gz> `
  --reference-archive-sha256 <published-sha256> `
  --output <new-empty-output>

python scripts/trajectory/audit_canonical_trace_rebuild.py `
  --rebuilt-root <new-output> `
  --raw-root <verified-raw-traces> `
  --reference-delivery <verified-reference-delivery> `
  --verify-source-archives `
  --output <new-output>/independent-audit.json
```

独立审计器不调用导出函数；它从原始轨迹重新构造每一轮并逐项比对 thought、调用、
实际参数、结果、图片绑定和最终证据 ID。两个来源归档的 SHA-256、交付包内部校验和、
零丢行以及最终证据对先前未掩码成功观察的因果可见性必须同时通过。

`ms-swift-policy` 和 `ms-swift-perception` 是两个独立数据集：

- policy：完整 ReAct episode，保留 provider 原生 `<think>`；
- perception：图片观察 JSON，不伪造 thinking。

两者均使用标准 `messages` 和 `images` 字段。policy 额外提供 episode 实际使用的
`tools` schema。

## 3. 发布前检查

```powershell
python -m ifv_training audit --strict --input <ms-swift-policy>
python -m ifv_training audit --strict --input <ms-swift-perception>

python scripts/probe/verify_ms_swift_agent_dataset.py `
  --model <目标 Qwen checkpoint> `
  --policy-dir <ms-swift-policy> `
  --perception-dir <ms-swift-perception> `
  --loss-scale ifv_agent+ignore_empty_think `
  --output <processor-verification.json>
```

processor 的上下文、图片、截断、padding-free、sequence-parallel 和 thinking 参数必须
逐项取自实际训练 profile。当前四卡 H20 使用 SP4；不能复用八卡 SP8 的报告。

JSON 审计只检查结构；真实 processor 验证还必须通过：

- `<think>` token 出现在 assistant labels；
- 工具调用和工具结果进入编码后的输入；
- 图片没有被 processor 丢失；
- 每行存在可训练 assistant token；
- 编码长度不超过目标上下文上限。
- provider 原生非空 `<think>` 完整保留并统一使用 1.0 权重，不按长度截断或降权；
- 完整 `<tool_call>` 与最终 `<answer>` 均使用 2.0 权重；
- tool response 与 `loss=false` 历史坏动作的 label/loss weight 均为 0；
- 可训练 token 中只出现上述允许的正权重。

不能用 ms-swift 4.4.2 内置的 `qwen` loss scale 代替：该配置仍匹配旧
`✿FUNCTION✿` 语法，不会命中 Qwen3.5 实际渲染的
`<tool_call><function=...>`。正式训练必须加载
`training/plugins/ifv_sft_agent_plugin.py`，并让 processor-v4 以真实 token 证明规则
已经生效。历史 processor-v3 报告和 `ignore_empty_think` 吞吐实验不能作为下一轮启动
依据。

目标模型变化时必须重新跑 processor 验证，不能复用旧模型的通过结果。

全量训练前还必须在完全相同的 H20、FSDP2、SP4、数据模板和相对 loss 权重下运行一步
canary，检查四卡 loss 有限、无 rank 退出、显存不过界。训练结束后先跑小规模 Agent
行为 canary：工具调用可解析、报告满足结构契约、没有 `length/abort`；通过后才进入冻结
1,527 分母的全量推理与 judge。训练 loss 下降本身不构成能力提升证据。

## 4. 数据边界

- evaluator private gold、judge 字段、source access policy 和内部缓存不得进入模型可见
  消息；
- provider wire instruction 不进入训练输入；
- `action_only` 不进入 reasoning policy SFT；
- 失败轨迹可保留在 audit/holdout，但不能通过转换器伪装成高质量训练样本；
- 训练包中的每一行是一条完整 episode，不拆成 step-level policy 样本。
- 两批轨迹必须在 canonical raw-trace 层合并、重新绑定各自媒体并统一重新审计；不得把两
  个已经转换的 ms-swift JSONL 直接拼接后声称是正式发布。缺少任一 policy 图片时暂停
  发布，不删除该轨迹，也不退化为文本训练。

## 5. 当前暂停点

本次收尾完成代码、文档、格式审计和真实 processor 验证后，停在完整教师 rollout
启动之前。不会自动启动 8,490 条训练数据的全量 rollout。

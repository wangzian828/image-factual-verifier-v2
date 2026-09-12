# SFT Canonical Release 工作流

当前版本（2026-09-08）的唯一训练发布链路：

```text
teacher rollout
  + strict trace audit
  + frozen SFT judge
  -> accepted-teacher-release
  -> accepted-dataset
  -> ms-swift policy/perception conversion
  -> real Qwen processor verification
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

目标模型变化时必须重新跑 processor 验证，不能复用旧模型的通过结果。

## 4. 数据边界

- evaluator private gold、judge 字段、source access policy 和内部缓存不得进入模型可见
  消息；
- provider wire instruction 不进入训练输入；
- `action_only` 不进入 reasoning policy SFT；
- 失败轨迹可保留在 audit/holdout，但不能通过转换器伪装成高质量训练样本；
- 训练包中的每一行是一条完整 episode，不拆成 step-level policy 样本。

## 5. 当前暂停点

本次收尾完成代码、文档、格式审计和真实 processor 验证后，停在完整教师 rollout
启动之前。不会自动启动 8,490 条训练数据的全量 rollout。

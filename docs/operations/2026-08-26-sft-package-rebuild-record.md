# 2026-08-26：紧凑版 SFT 包重打记录

## 目的

用 `ifv-trajectory-sft-v3` 的紧凑全轨迹导出重新生成训练包。只改变训练视图；
不重跑原始 rollout，不删除原始 trace，不启动训练。

## 输入与来源

| 项目 | 绝对路径 / 值 |
| --- | --- |
| 正式 teacher pipeline | `/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/gemini37-r2-train8490-p10-20260824` |
| 原始新初始轨迹 merged run | `/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/gemini37-r2-train8490-p10-20260824/rollouts/initial/merged` |
| 原始 trace 数 | 940 |
| 冻结 SFT eligibility | `/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/gemini37-r2-train8490-p10-20260824/sft-eligibility/initial` |
| 冻结 accepted release | `/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/gemini37-r2-train8490-p10-20260824/sft-eligibility/initial/automatic-storage` |
| accepted / rejected | 475 / 465 |
| rollout 与 judge 模型 | `gemini-3.7-flash` |
| 代码 checkout | `/gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01` |
| 代码提交 | `ac25f92`（并发 SFT 导出） |

`main-05917` 的 judge gate 通过但未进入 accepted release。原因是严格轨迹导出发现
`ClaimAssessment/Evidence` 方向不一致；它保留在 rejected 归档，不作为训练样本。

## 本次输出

| 项目 | 绝对路径 / 值 |
| --- | --- |
| 新训练包 | `/gsdata/home/wza/image-factual-verifier-v2-data/generated/sft/packages/gemini37-r2-train8490-initial940-accepted475-compact-v3-r2-20260826` |
| 构建日志 | `/gsdata/home/wza/image-factual-verifier-v2-data/generated/sft/package-logs/gemini37-r2-train8490-initial940-accepted475-compact-v3-r2-20260826.log` |
| PID 记录 | `/gsdata/home/wza/image-factual-verifier-v2-data/generated/sft/package-logs/gemini37-r2-train8490-initial940-accepted475-compact-v3-r2-20260826.pid` |
| 训练 | 不启动 |
| split | `--all-train` |
| 导出并发 | `--export-concurrency 8` |

## 最终结果

| 项目 | 结果 |
| --- | --- |
| 包状态 | 成功完成 |
| provider-neutral policy 数据 | 475 条，全部 `train` |
| provider-neutral perception 数据 | 475 条，全部 `train` |
| long holdout | 0 条 |
| excluded episode | 0 条 |
| ms-swift policy 严格审计 | 通过，475 条 |
| ms-swift perception 严格审计 | 通过，475 条 |
| Qwen3.5 processor 审计 | 通过，950 条，无 processor error |
| Qwen3.5 最大输入长度 | 83,243 tokens |
| Qwen3.5 p50 / p90 / p95 / p99 | 7,653 / 46,630 / 61,177 / 74,821 tokens |
| policy 最大输入长度 | 83,243 tokens |
| policy p50 / p90 / p95 / p99 | 28,114 / 61,177 / 71,625 / 78,475 tokens |
| perception 最大输入长度 | 7,653 tokens |
| 超过 128K | 0 条 |
| 训练是否启动 | 否 |

真实 processor 审计文件：

`/gsdata/home/wza/image-factual-verifier-v2-data/generated/sft/packages/gemini37-r2-train8490-initial940-accepted475-compact-v3-r2-20260826/audits/processor-qwen35-9b.json`

最终包内的直接训练输入：

```text
/gsdata/home/wza/image-factual-verifier-v2-data/generated/sft/packages/gemini37-r2-train8490-initial940-accepted475-compact-v3-r2-20260826/ms-swift-policy/train.jsonl
/gsdata/home/wza/image-factual-verifier-v2-data/generated/sft/packages/gemini37-r2-train8490-initial940-accepted475-compact-v3-r2-20260826/ms-swift-perception/train.jsonl
```

包总目录：

`/gsdata/home/wza/image-factual-verifier-v2-data/generated/sft/packages/gemini37-r2-train8490-initial940-accepted475-compact-v3-r2-20260826`

## 失败的 r1 记录

r1 已保留，未覆盖：

`/gsdata/home/wza/image-factual-verifier-v2-data/generated/sft/packages/gemini37-r2-train8490-initial940-accepted475-compact-v3-r1-20260826`

r1 的导出数量也是 475 条，但旧审计器把 5 条工具错误观察中的字符串
`Gemini Interactions` 误判成 provider wire instructions，导致严格审计失败。
修复审计规则后生成 r2，实际审计通过。

## 相关代码提交

- `ac25f92`：`export_dataset` 增加 `--export-concurrency`，默认包构建使用 8 个导出 worker；结果按 episode ID 稳定落盘。
- `c39c96a`：审计器只拦截真正的 `Native Gemini Interactions protocol:` 提示词标记，不误伤工具错误观察。

导出并发只作用于 trace → full-trajectory SFT 转换；judge 的 `--concurrency` 是另一套参数。
worker 不直接写共享 JSONL，单条导出异常会记录到 `episode_metadata.jsonl` 并继续处理其他条目。

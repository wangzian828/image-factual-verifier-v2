# GPU-13 版本登记

最后核对日期：2026-08-16

这份表是 GPU-13 Agent 运行版本的唯一登记入口。分支名不是版本号；每次实验必须记录 commit、benchmark release、benchmark SHA-256、模型和 OCR 后端。

## 当前运行基线

| 项目 | 值 |
|---|---|
| canonical branch | `codex/gpu13-canary-20260804-plan-relaxation-01` |
| 当前服务器 checkout HEAD | `34a2eeb` |
| 最近一次行为测试 commit | `d34cf63` |
| 模型 | `gemini-3.7-flash` |
| 当前 OCR | EasyOCR `1.7.2`，共享 CPU reader |

`7c08a0a` 及其前面的 `0000541`、`d9a18df`、`c016fe6` 只包含运行路径、轮询和运维 guard；最近一次行为测试使用的是 `d34cf63`，不能把当前服务器 HEAD 直接当成该次 Agent 行为版本。

## 已接受的历史结果基线

| 基线用途 | commit | benchmark release | OCR | 结果 | 备注 |
|---|---|---|---|---:|---|
| construction 高基线 | `fe43bdd` | `five-construction-parallel-20260815T105457Z` | 本地 CPU PaddleOCR | 5/5 | 历史接受结果；不是当前服务器上的最新 checkout |
| EF3 高基线 | `0738e59` | `ef3-five-parallel-20260815T155441Z` | PaddleOCR Cloud API | 4/5 | 服务器 run：`ef3-five-parallel-paddleocr-api-r1-20260815` |
| 组合高基线 | `fe43bdd` + `0738e59` | construction + EF3 | 两种 OCR 环境 | 9/10 | 这是能力分组基线，不是一个 commit 的单次 10 条 run |

## 非高基线版本

| commit | construction | EF3 | 组合 | 结论 |
|---|---:|---:|---:|---|
| `f9d3a28` | 4/5 | 4/5 | 8/10 | 旧的较差版本，不得作为高基线 |
| `d34cf63` | 2/5 | 未重跑 | — | 最近一次 construction canary；5 条中 1 条 OCR 超时，另外出现两条 false-real |

`f9d3a28` 的两个服务器 run 使用同一批 benchmark：

- `gemini37-latest-v5-construction-five-20260816T052835Z`
- `gemini37-latest-v5-ef3-five-20260816T052835Z`

它们虽然都使用 Gemini 3.7，但 manifest 中没有当前 API OCR 的 cache version；不能把它们与 PaddleOCR API-only 运行简单视为同一环境。

## 差异审计结论

### `0738e59` → `d34cf63`

两者之间的 Agent 编排代码基本没有变化；主要差异是：

- OCR API 实现增加了提交重试和队列退避；
- 运行结果、trace 路径和评估后处理发生变化；
- 当前 construction run 出现了一次 PaddleOCR API 请求超时。

因此，EF3 高基线与当前 construction 结果不能直接互相替代。要判断当前 Agent 是否损坏 EF3，必须用 `d34cf63` 在同一 EF3 release 上重新跑一组。

### `fe43bdd` → `d34cf63`

construction 基线的差异更大，包含：

- 本地 CPU PaddleOCR → PaddleOCR Cloud API；
- OCR 工具实现的大幅替换；
- Evidence/Findings 汇聚和评估导出改动；
- 后续运行时修复和 OCR cache 版本变化。

当前 construction 的两条错误轨迹显示，问题不只是 OCR：

- 一条在 `meaningful_routes_exhausted` 后没有 Finding/Evidence，仍被判为 `real`；
- 一条 refuted 图被视觉连贯性推成 `real`；
- 另有一条明确的 OCR API timeout。

所以不能把 9/10 高基线归因成“Gemini 3.7 本身”；它是两个能力分组、两个 commit、两种已登记 OCR 环境的组合结果。

## 运行登记规则

1. 论文或报告中引用结果时，必须同时写 commit 和 benchmark release。
2. `f9d3a28` 标记为旧较差版本，不得写成高基线。
3. `fe43bdd` 只作为 construction 高基线，`0738e59` 只作为 EF3 高基线。
4. 当前 canary 的行为结果使用行为测试 commit，不使用仅包含运维改动的服务器 HEAD。
5. 任何“高结果”结论必须保留 `summary.json`、`run_manifest.json` 和对应 trace 目录。

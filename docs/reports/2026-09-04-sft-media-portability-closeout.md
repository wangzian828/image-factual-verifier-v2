# SFT 图片投影与可移植性收尾记录

日期：2026-09-04

## 1. 范围

本次只完成训练包、图片投影、真实 processor 探针、文搜图链路和跨服务器交付入口。
不启动 8,490 条全量教师 rollout，不修改训练/测试数据集内容。

## 2. 正式小包

来源：

```text
unified-react-v1-gemini37-train10-sftformat-20260904-6f2de3f/
accepted-release-media-v2-36a8c3d/
```

输出：

```text
unified-react-v1-gemini37-train10-sftformat-20260904-6f2de3f/
sft-training-package-media-v2-36a8c3d/
```

数量：

- selected release：3；
- reasoning policy：2；
- perception：3；
- action-only：1；
- long holdout：0；
- policy/perception 严格结构审计：通过。

`--minimum-accepted-cases` 按 reasoning policy 行计算，因此本包使用 2，而不是
selected release 总数 3。

## 3. 图片投影

- 初始 policy 图片进入初始 user `<image>`；
- 工具后首次出现的候选图、裁剪图和聚焦视图挂到对应 `tool_response`；
- 相同图片按 SHA-256 全 episode 去重；
- 发布包图片写为 data URI，不依赖原服务器路径；
- marker 数与顶层 `images` 数严格相等；
- 原始 runtime trace、图片和 accepted release 不改写。

真实小包：

| Case | 图片数 | Marker 数 | 用途 |
| --- | ---: | ---: | --- |
| `main-02731` | 3 | 3 | reasoning |
| `main-02733` | 3 | 3 | reasoning |
| `main-02728` | 12 | 12 | action-only |

## 4. Processor 探针

旧探针错误地要求源 `tool_call` JSON 在 Qwen 编码文本中逐字出现。真实 ms-swift
模板会将其渲染成：

```text
<function=工具名>
<parameter=参数名>
...
</parameter>
</function>
```

新探针按真实 function/parameter token 检查，并继续检查：

- `<think>` 位于训练 labels；
- `tool_response` 位于编码输入；
- 图片未被 processor 丢弃；
- 训练 label 非空；
- 总长度不超过 128K。

服务器最终 processor 报告在提交同步后补记。

## 5. 文搜图

`text_image_search` 已完成以下链路核对：

- Serper 图片搜索真实返回 `results`、页面 URL 和图片 URL；
- active ReAct tool schema 和英文 prompt 可见；
- reducer 只记录为 Discovery；
- 最多 3 张候选图进入紧邻工具结果的下一轮请求；
- runtime context artifact 可由 SFT 导出器恢复；
- 候选图不会因进入上下文自动升级为 Evidence。

## 6. 可移植性

新增通用入口：

```text
configs/runtime.env.example
scripts/server/ifv_env.sh
scripts/server/doctor.py
scripts/server/bootstrap_runtime.sh
scripts/server/run_ifv.sh
scripts/server/start_eval.sh
scripts/server/start_gemini_eval.sh
scripts/server/eval_worker.sh
scripts/server/poll_eval.sh
```

完成：

- active Python/runtime/训练/服务入口不再写死当前账号、挂载点、模型目录或代理；
- teacher autopilot 使用 `run_ifv.sh` 或 `IFV_SERVER_RUNNER`；
- GPU allowlist 改为可选环境配置；
- 模型 checkpoint、vLLM 环境和数据根目录由环境或 CLI 指定；
- Jupyter 客户端新增正式文件下载功能；
- gpu-13 wrapper 只保留机器身份边界，复用通用实现。

## 7. 本地验证

```text
主仓：489 passed
training：114 passed
compileall：通过
git diff --check：通过
active 代码机器路径/IP 扫描：0 条
```

服务器 shell 语法、真实 processor、样本下载和小规模 smoke 在正式提交同步后补记。

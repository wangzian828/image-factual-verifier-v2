# IFV Agent 教师轨迹 API 交接

Updated: 2026-09-07

## 目标

在新的 Linux 服务器上，通过目标服务器已有的大型开源 Qwen
OpenAI-compatible API，完成：

```text
8,490 条训练集
  -> unified-react-v1 teacher rollout
  -> raw-history canonical trace
  -> strict trace audit
  -> independent Qwen SFT judge
  -> quality reroll
  -> accepted release
  -> policy/perception SFT package
```

本交接不部署模型，不携带 API key，也不启动当前机器上的模型服务。目标服务器的
Codex 负责确认 endpoint、model ID、并发和配额。

当前交付到“代码、启动入口、数据定位和验收契约均已准备完成”为止。由于当前机器
没有大型 Qwen teacher/judge API，真实 endpoint 预检、10 条 smoke、全量 rollout
和 SFT judge 均有意留给目标服务器上的 Codex 执行；这不是缺失项，也不得在当前
机器上改用小 Qwen 或其他模型代跑。

## 固定代码版本

```text
repository: git@github.com:wangzian828/image-factual-verifier-v2.git
branch: codex/gpu13-canary-20260804-plan-relaxation-01
minimum commit: 5317d9a
```

交接包的 `MANIFEST.json` 和 `bootstrap.sh` 会固定生成该包时的精确 commit；实际
接手必须使用该精确 commit，而不是只停留在 minimum commit。

服务器不得直接修改源码。需要修改时，在开发工作树提交并推送，再在服务器
fast-forward。

## 固定训练数据

```text
ModelScope dataset: jiashuhong/factcheck_train
archive: factcheck_train-8490-20260907.tar.gz
bytes: 24127361006
sha256: 2fda3ca7144d899e355fcbbaa4e6b93350878dbf5225ab423402123d5e37a448
manifest rows: 8490
image count: 8490
private-gold rows: 8490
```

数据归档只用于训练集。不得使用 `jiashuhong/factcheck_test`、测试图片或测试
private gold 生成 teacher trajectory、SFT 或 RL 数据。

## 大 Qwen Teacher API 契约

目标 endpoint 必须：

- 提供 OpenAI-compatible `/v1/models` 和 `/v1/chat/completions`；
- 接受图片输入和原生 tool calls；
- 使用支持图像输入的大型 Qwen，而不是待训练的小 Qwen student；
- 保留 provider 返回的真实 reasoning/`<think>`；不得补写不存在的 thought；
- 对同一 interaction history 支持多轮工具调用；
- 返回的 model ID 与 `/v1/models` 一致。

配置写入目标服务器未跟踪的
`~/.config/image-factual-verifier/runtime.env`：

```bash
IFV_DATA_ROOT=/absolute/server/data/root
IFV_OMP_NUM_THREADS=1
MODELSCOPE_API_TOKEN=provided-out-of-band

QWEN_TEACHER_BASE_URL=http://teacher-host:port/v1
QWEN_TEACHER_API_KEY=none
QWEN_TEACHER_MODEL=served-teacher-model
QWEN_TEACHER_VISION_MODEL=served-teacher-model
IFV_TEACHER_ROLLOUT_PROFILE=teacher-qwen-server
IFV_TEACHER_ROLLOUT_MODEL=served-teacher-model
```

如果 teacher 服务需要鉴权，把 `QWEN_TEACHER_API_KEY` 设置为目标服务器提供的
凭据；不要写入 Git、命令行、日志、trace 或交接包。

## SFT Judge API 契约

judge 可以与 teacher 共用 endpoint，也可以独立部署。它必须支持图片输入和严格
JSON schema 输出。独立配置示例：

```bash
IFV_SFT_ELIGIBILITY_PROVIDER=qwen_local
IFV_SFT_ELIGIBILITY_BASE_URL=http://judge-host:port/v1
IFV_SFT_ELIGIBILITY_MODEL=served-judge-model
IFV_SFT_ELIGIBILITY_WIRE_API=chat_completions
IFV_SFT_ELIGIBILITY_ENABLE_THINKING=true
IFV_SFT_ELIGIBILITY_API_KEY_ENV=IFV_SFT_ELIGIBILITY_API_KEY
IFV_SFT_ELIGIBILITY_API_KEY=provided-out-of-band
```

pipeline state 只记录 `IFV_SFT_ELIGIBILITY_API_KEY` 这个变量名，不记录变量值。

## 工具侧配置

Teacher API 不是唯一外部依赖。目标服务器还必须配置项目实际启用的搜索、网页读取、
OCR、图片上传和视觉搜索服务。至少运行：

```bash
source scripts/server/ifv_env.sh
scripts/server/run_ifv.sh python scripts/server/doctor.py --require-provider local --json
```

随后用一个真实 case 验证所有 required tools。工具访问失败、空搜索或 endpoint
错误必须原样保留在 raw history 中，不能改写为事实证据。

## 执行顺序

下载并验证数据：

```bash
scripts/server/prepare_factcheck_dataset.sh --split train
```

默认启动 10 条后台 smoke：

```bash
scripts/server/start_teacher_rollout_portable.sh
```

命令返回 `pid`、`output_dir`、`log_file` 和 `pid_file`。检查：

```bash
cat <output_dir>/pipeline-state.json
find <output_dir>/rollouts -type f -path '*/traces/*.json' | wc -l
tail -n 100 <log_file>
```

只有 10 条 smoke 完成以下验收后，才运行全量：

```bash
scripts/server/start_teacher_rollout_portable.sh --full
```

## 10 条 Smoke 验收

- 输入严格来自训练集；
- 10/10 case 都生成终态 trace 或明确工程失败；
- 每轮最多一个原生 tool call；
- 原始连续搜索结果完整进入 `state.all_steps`；
- 无匹配搜索、工具错误、访问失败没有被静默删除；
- Judgment 能读取完整 raw history；
- strict trace audit 无 failure；
- SFT judge 逐条完成，拒绝样本不会进入 accepted release；
- policy/perception 数据审计通过；
- endpoint、model、commit、dataset SHA 和并发记录完整；
- 没有 private gold 泄漏到模型输入或训练行；
- 没有持续增长的连接泄漏或遗留 rollout 子进程。

## 输出与停止条件

主要输出位于：

```text
<output_dir>/pipeline-state.json
<output_dir>/rollouts/
<output_dir>/sft-eligibility/
<output_dir>/classification/
<output_dir>/accepted-release/
<output_dir>/sft-training-package/
<output_dir>/audits/pipeline-summary.json
```

一键入口默认只生成并审计 SFT package，不自动启动 GPU SFT。只有真实 Qwen
processor verification 通过后，才允许显式向 `run_teacher_sft_pipeline.sh`
传递 `--run-training` 及训练 profile。

## 给接手 Codex

以下步骤全部在具备大型 Qwen API 的目标服务器执行。当前机器无需、也不应执行
teacher/judge API 调用。

接手后按顺序执行：

1. 阅读本文件、`AGENTS.md` 和 `docs/portable-deployment-handoff.md`。
2. clone 固定 branch，确认 `HEAD` 不低于固定 commit。
3. 配置未跟踪的 `runtime.env`，不得输出任何凭据值。
4. 验证 ModelScope 归档 SHA、8,490 条 manifest/private gold 和 8,490 张图片。
5. 调用 `/v1/models` 确认 teacher/judge model ID。
6. 运行 10 条 smoke 并逐条审计完整轨迹。
7. 只有 smoke 验收通过后才运行 `--full`。
8. 全量完成后生成 accepted release、双路 SFT package、processor verification
   和最终哈希清单。

不得以缺少模型配置为由改用小 Qwen、其他用户的 endpoint 或未获授权的公共模型。

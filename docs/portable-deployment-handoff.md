# IFV 新服务器部署与 Codex 交接

目标：在一台没有现有 IFV 环境的新 Linux 服务器上，使用正式 Git 提交完成 runtime
安装、数据验证、小规模 rollout、SFT 包审计和训练前检查。本文不包含任何现有机器的
账号、地址、密码、挂载点或密钥。

## 1. 交付物

- Git 仓库及固定 commit；
- `configs/runtime.env.example`；
- 已完成的 SFT training package；
- 数据包 `MANIFEST.json` 和 `audits/`；
- 目标 Qwen checkpoint；
- 本文和 `qwen-ms-swift-deployment.md`。

数据、模型、日志和 checkpoint 不放入 Git checkout。

## 2. 路径约定

部署者自行选择：

```bash
export IFV_REPO_ROOT=/absolute/path/to/image-factual-verifier
export IFV_DATA_ROOT=/absolute/path/to/ifv-data
export IFV_MODEL_ID=/absolute/path/to/qwen-checkpoint
```

通用代码不假设这些路径。默认数据目录仅用于个人开发；生产必须显式配置
`IFV_DATA_ROOT`。服务器专属限制可选配置：

```bash
export IFV_EXPECTED_REPO_ROOT="$IFV_REPO_ROOT"
export IFV_EXPECTED_BRANCH=<approved-branch>
export IFV_ALLOWED_GPU_IDS=<comma-separated-physical-gpu-ids>
```

`IFV_ALLOWED_GPU_IDS` 为空表示不额外施加机器专属 allowlist；仍由
`CUDA_VISIBLE_DEVICES` 选择本次任务的 GPU。

## 3. Runtime 安装

```bash
git clone <repository-url> "$IFV_REPO_ROOT"
cd "$IFV_REPO_ROOT"
mkdir -p ~/.config/image-factual-verifier
cp configs/runtime.env.example \
  ~/.config/image-factual-verifier/runtime.env
chmod 600 ~/.config/image-factual-verifier/runtime.env
# 编辑文件，填入本机路径、provider key、代理和 endpoint。

source scripts/server/ifv_env.sh
scripts/server/bootstrap_runtime.sh
scripts/server/run_ifv.sh python scripts/server/doctor.py \
  --require-provider gemini --json
```

若不用 Gemini，将 `--require-provider` 改成 `qwen`、`local` 或 `none`。

## 4. ModelScope 数据归档

训练集使用单一权威归档，避免托管平台逐张审核时产生静默缺图：

```text
dataset: jiashuhong/factcheck_train
file: factcheck_train-8490-20260907.tar.gz
sha256: 2fda3ca7144d899e355fcbbaa4e6b93350878dbf5225ab423402123d5e37a448
```

配置 `MODELSCOPE_API_TOKEN` 后，可自动下载、校验、解压并验证 8,490 条
manifest、私有 gold 和图片引用：

```bash
source scripts/server/ifv_env.sh
scripts/server/prepare_factcheck_dataset.sh --split train
```

测试集归档位于 `jiashuhong/factcheck_test`，文件
`factcheck_test-1682-20260907.tar.gz`，SHA-256 为
`485dba3b8b3947913f372b57f85dbe30b456894022b3fa467c9460dcb8847ea5`。

## 5. 一键教师 Rollout

教师 API、10 条 smoke、全量 rollout、SFT judge 和 SFT package 导出的完整交接流程见
[`teacher-rollout-and-sft-handoff.md`](teacher-rollout-and-sft-handoff.md)。该接手任务在
导出完成后停止，不包含 processor verification 或 GPU SFT。
`teacher-rollout-api-handoff.md` 只保留旧版 API 摘要。

大型开源 Qwen 通过 OpenAI-compatible endpoint 提供服务。配置：

```bash
QWEN_TEACHER_BASE_URL=http://teacher-host:port/v1
QWEN_TEACHER_API_KEY=none
QWEN_TEACHER_MODEL=served-teacher-model
QWEN_TEACHER_VISION_MODEL=served-teacher-model

# The launcher reuses this same Qwen endpoint for webpage Evidence extraction.
BROWSE_EXTRACT_PROVIDER=qwen_local
BROWSE_EXTRACT_MODEL=served-teacher-model
BROWSE_EXTRACT_BASE_URL=http://teacher-host:port/v1
BROWSE_EXTRACT_WIRE_API=chat_completions

SERPER_API_KEY=provided-out-of-band
BROWSE_FETCH_PROVIDER=jina
JINA_API_KEY=provided-out-of-band
OCR_BACKEND=baidu
BAIDU_OCR_API_KEY=provided-out-of-band
BAIDU_OCR_SECRET_KEY=provided-out-of-band
VISUAL_SEARCH_PROVIDER=serper_lens
IMAGE_UPLOAD_PROVIDER=oss
OSS_ACCESS_KEY_ID=provided-out-of-band
OSS_ACCESS_KEY_SECRET=provided-out-of-band
OSS_ENDPOINT=https://oss-endpoint
OSS_BUCKET_NAME=private-upload-bucket

IFV_SFT_ELIGIBILITY_PROVIDER=qwen_local
IFV_SFT_ELIGIBILITY_BASE_URL=http://judge-host:port/v1
IFV_SFT_ELIGIBILITY_MODEL=served-judge-model
IFV_SFT_ELIGIBILITY_ENABLE_THINKING=true
```

judge 使用独立凭据时，只保存环境变量名：

```bash
IFV_SFT_ELIGIBILITY_API_KEY_ENV=IFV_SFT_ELIGIBILITY_API_KEY
IFV_SFT_ELIGIBILITY_API_KEY=provided-out-of-band
```

无参数默认命令支持自动执行 `smoke -> full`，但本次交给另一个 Codex 时不要直接使用
自动衔接模式。必须先单独完成并检查 10 条 smoke：

```bash
scripts/server/start_teacher_rollout_portable.sh \
  --foreground \
  --smoke-only \
  --output-dir <smoke-output-dir>
```

该单个命令内部自动完成初始 10 条 rollout、逐轮 strict trace audit 与 SFT judge、
对拒绝或未完成 case 最多 3 轮 quality reroll，以及 accepted release 和
policy/perception SFT package 导出。另一个 Codex 不得手工逐轮启动、单独调用 judge、
手工触发 SFT 导出或从中间目录拼接结果。`rollouts/initial/attempt-*` 和从其中手工
复制的 trace preview 都不是 accepted release，也不是 SFT package。

smoke 完成后必须检查正式 `accepted-release/trajectory_sft.jsonl` 和
`sft-training-package/ms-swift-policy/*.jsonl`：顶层 `tools/messages/images`、
真实 `<think>`、最终 `<answer>`、tool-call/tool-response 时序、字符串化 arguments、
`<image>` marker 数量和过程图片均必须正确。runtime archive 必须能在不依赖原 rollout
目录的情况下恢复 candidate、crop 和 focused-view 图片。

只有这些检查全部通过，才启动全部 8,490 条：

```bash
scripts/server/start_teacher_rollout_portable.sh \
  --full \
  --output-dir <full-output-dir>
```

全量命令返回 PID、日志和输出目录后，另一个 Codex 仍必须持续检查任务直到进程退出，
完成全量 SFT judge、最多 3 轮 quality reroll、accepted release 和双 SFT package，
并确认 `<full-output-dir>/audits/final-delivery.json` 中 `final_delivery=true`。不要把
endpoint 凭据写进命令行、Git、trace 或 pipeline state。

本流程不使用 Gemini API，不允许 EasyOCR、本地 OCR 或 OCR fallback。图片上传服务
可以按目标服务器条件显式选择 OSS、custom 或 temp。

## 6. Rollout 小测

本任务不要使用低层 `start_teacher_rollout.sh` 只生成 raw rollout。唯一 smoke 入口是：

```bash
scripts/server/start_teacher_rollout_portable.sh \
  --foreground \
  --smoke-only \
  --output-dir <smoke-output-dir>
```

该入口必须连续完成 Agent rollout、拒绝样本 reroll、SFT judge、accepted release、
双 SFT package 和轨迹格式检查所需的全部正式产物。

## 7. SFT 包

从 accepted release 构建：

```bash
python scripts/trajectory/build_sft_training_package.py \
  --accepted-release <accepted-release> \
  --output-dir <new-package-dir> \
  --minimum-accepted-cases <reasoning-row-minimum> \
  --export-concurrency 8
```

注意：`--minimum-accepted-cases` 约束 reasoning policy 行，不是
`selected_release_cases` 总数；action-only 不计入该门槛。

结构审计和真实 processor 验证：

```bash
python -m ifv_training audit --strict --input <package>/ms-swift-policy
python -m ifv_training audit --strict --input <package>/ms-swift-perception
python training/scripts/probe/verify_ms_swift_agent_dataset.py \
  --model "$IFV_MODEL_ID" \
  --policy-dir <package>/ms-swift-policy \
  --perception-dir <package>/ms-swift-perception \
  --output <package>/audits/processor-verification.json \
  --max-context 131072
```

只有报告 `passed=true` 后才允许训练。

## 8. Qwen 教师与 `<think>`

更换大型 Qwen 教师时：

1. 使用目标模型的原生 tool-call 模板；
2. 每个 ReAct assistant 动作保留模型真实 `<think>`；
3. 不为缺失 thought 的动作补写文本；
4. Judgment 没有 thought 不影响已有 ReAct reasoning supervision；
5. 使用目标 checkpoint 的 processor 重新编码整包；
6. 确认 `<think>` token 位于 labels，而工具结果只作为条件输入。

## 9. 交给另一个 Codex

新的 Codex 首先读取：

1. `docs/teacher-rollout-and-sft-handoff.md`
2. `docs/portable-deployment-handoff.md`
3. `docs/agent-structure.md`
4. `docs/agent-prompt-and-runtime-guide.md`
5. `docs/sft-training-and-data-construction.md`
6. `docs/qwen-ms-swift-deployment.md`
7. `docs/trajectory-artifact-guide.md`

然后执行：

```bash
git status --short --branch
git log -1 --oneline --decorate
source scripts/server/ifv_env.sh
scripts/server/run_ifv.sh python scripts/server/doctor.py --json
python -m pytest -q
(
  cd training
  PYTHONPATH=. python -m pytest -q
)
```

其唯一业务任务按以下顺序执行：

1. 只调用一次正式 `--smoke-only` 入口，完整运行 10 条训练集 smoke；
2. 等待该一体化流水线自动完成拒绝/未完成样本最多 3 轮 reroll、逐轮 SFT judge、
   accepted release 与双 SFT package；
3. 不得手工拆分 rollout、reroll、judge 或 SFT 导出步骤；
4. 检查完整轨迹内容、`<think>` 格式、tool 时序、顶层
   `tools/messages/images` 以及全部过程图片；
5. smoke 全部确认通过后启动 8,490 条全量；
6. 持续运行至全量 accepted release、SFT 导出和最终审计完成；
7. 仅在 `audits/final-delivery.json` 的 `final_delivery=true` 后结束。

不得在服务器 checkout 直接改源码。修改应在开发工作树完成、提交并推送，再由目标
服务器 fast-forward。不得把真实密码、API key 或私有 gold 写入 prompt、trace、
脚本、文档或提交。

## 10. 明确停止点

本交接任务已经获得全量运行授权。默认入口会在 10 条 smoke 通过后自动继续运行全部 8,490 条，
完成 SFT eligibility judge、quality reroll、accepted release、policy/perception SFT
package 和严格审计。只有 `audits/final-delivery.json` 中
`final_delivery=true` 才能停止和交付；不得用 smoke、单个 attempt、手工预览包或仅有
raw trace 的目录代替。

停止点不包含真实 checkpoint processor verification、GPU SFT、RL、Direct QA 或测试集
实验；这些由后续训练或评测负责人执行。

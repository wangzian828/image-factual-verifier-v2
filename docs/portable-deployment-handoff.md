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

大型开源 Qwen 通过 OpenAI-compatible endpoint 提供服务。配置：

```bash
QWEN_TEACHER_BASE_URL=http://teacher-host:port/v1
QWEN_TEACHER_API_KEY=none
QWEN_TEACHER_MODEL=served-teacher-model
QWEN_TEACHER_VISION_MODEL=served-teacher-model

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

默认命令运行 10 条 smoke，并在后台完成 rollout、strict trace audit、
SFT judge、quality reroll、accepted release 和 SFT package：

```bash
scripts/server/start_teacher_rollout_portable.sh
```

显式启动全部 8,490 条：

```bash
scripts/server/start_teacher_rollout_portable.sh --full
```

命令返回 PID、日志和输出目录。不要把 endpoint 凭据写进命令行、Git、trace
或 pipeline state。

## 6. Rollout 小测

先确认 checkout、数据、API 和工具，不直接启动全量：

```bash
cd "$IFV_REPO_ROOT"
source scripts/server/ifv_env.sh
scripts/server/start_teacher_rollout.sh \
  --benchmark <runtime-cases.jsonl> \
  --output-dir "$IFV_DATA_ROOT/runs/eval/handoff-smoke-10" \
  --limit 10 \
  --concurrency 10 \
  --rollouts-per-case 1
scripts/server/poll_eval.sh handoff-smoke-10 --tail 100
```

验收：

- `run_manifest.json` 存在；
- `traces/*.json` 持续增加；
- `summary.json` 最终存在；
- 无启动参数错误；
- 工程错误逐条归因；
- `text_image_search` 的结果含真实候选时，下一轮 context archive 有对应图片；
- 工具结果不是仅剩 transport status；
- 最终 Judgment 能看到完整 provider interaction 历史。

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

1. `docs/portable-deployment-handoff.md`
2. `docs/agent-structure.md`
3. `docs/agent-prompt-and-runtime-guide.md`
4. `docs/sft-training-and-data-construction.md`
5. `docs/qwen-ms-swift-deployment.md`
6. `docs/trajectory-artifact-guide.md`

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

不得在服务器 checkout 直接改源码。修改应在开发工作树完成、提交并推送，再由目标
服务器 fast-forward。不得把真实密码、API key 或私有 gold 写入 prompt、trace、
脚本、文档或提交。

## 10. 明确停止点

完成环境、10 条 smoke、SFT 导出、严格审计和真实 processor 验证后停止。全量教师
rollout 必须由数据负责人单独确认模型、并发、预算、输入 manifest 和输出目录后启动。

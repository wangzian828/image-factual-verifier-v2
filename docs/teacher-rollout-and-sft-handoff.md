# IFV 教师轨迹与 SFT 完整交接

更新日期：2026-09-08

本文是交给目标服务器上的 Codex 和实验人员的权威执行手册。代码从 GitHub 拉取，
训练集从 ModelScope 下载；图片、轨迹、日志、accepted release、SFT package 和
checkpoint 都留在目标服务器，不放入 Git，也不经当前工作站中转。

## 1. 目标与硬边界

完整链路：

```text
8,490 条训练集
  -> 大型开源 Qwen teacher API
  -> unified-react-v1 Agent / 视觉工具 / 网页 Evidence 抽取
  -> raw-history canonical trace
  -> strict trace audit
  -> Qwen SFT eligibility judge
  -> 最多三轮 quality reroll
  -> accepted release
  -> policy/perception SFT package
```

接手 Codex 的**唯一任务**止于已审计的 `policy/perception` SFT package 导出。
它不负责 GPU SFT 训练、RL、Direct QA、测试集实验、模型部署、源码改动或任何与
训练集教师轨迹无关的工作。导出包和审计产物交回后，由后续训练负责人另行决定是否
做目标 checkpoint 的 processor verification 与实际训练。

**10 条 smoke、任意 `rollouts/initial/attempt-*` 目录、手工抽取的成功 trace 和本地
preview 都不是最终交付。**最终交付必须覆盖全部 8,490 条训练 case，并生成机器验收
文件 `audits/final-delivery.json`，其中 `final_delivery` 必须为 `true`。

- 仅 `jiashuhong/factcheck_train` 可进入 teacher rollout、SFT 或 RL。
- 测试集、测试图片和测试 private gold 绝不能进入上述链路。
- teacher 必须是支持图像输入的大型开源 Qwen API；不得改用小 Qwen、Gemini 或未授权模型。
- 同一个 teacher API 同时承担主 Agent、视觉工具和网页 Evidence 抽取。
- OCR 只允许百度 OCR；禁止 EasyOCR、本地 OCR 和 OCR fallback。
- 空搜索、无匹配、工具错误和访问失败必须保留在 raw history，不能改写成事实证据。
- private gold 仅供服务器端 frozen judge 和审计使用；不得进入 Agent 输入或模型可见训练行。
- 服务器 checkout 只允许运行已提交代码；源码修改必须在开发机提交、推送后再 fast-forward。

## 2. 固定来源

```text
GitHub: git@github.com:wangzian828/image-factual-verifier-v2.git
Branch: main
```

接手时记录准确 `git log -1 --format=%H`；不得以历史实验分支替代 `main`。

训练集归档：

```text
ModelScope dataset: jiashuhong/factcheck_train
file: factcheck_train-8490-20260907.tar.gz
bytes: 24127361006
sha256: 2fda3ca7144d899e355fcbbaa4e6b93350878dbf5225ab423402123d5e37a448
manifest records / images / private-gold rows: 8490 / 8490 / 8490
```

以下测试集归档仅用于评测，禁止用于本手册的链路：

```text
ModelScope dataset: jiashuhong/factcheck_test
file: factcheck_test-1682-20260907.tar.gz
sha256: 485dba3b8b3947913f372b57f85dbe30b456894022b3fa467c9460dcb8847ea5
```

## 3. 新服务器安装

目标服务器自行选择绝对路径；不得照搬旧机器的用户名、挂载点、代理、GPU 编号或模型目录：

```bash
export IFV_REPO_ROOT=/absolute/path/to/image-factual-verifier-v2
export IFV_DATA_ROOT=/absolute/path/to/ifv-data

git clone --branch main git@github.com:wangzian828/image-factual-verifier-v2.git "$IFV_REPO_ROOT"
cd "$IFV_REPO_ROOT"
git status --short --branch
git log -1 --oneline --decorate
mkdir -p "$HOME/.config/image-factual-verifier"
cp configs/runtime.env.example "$HOME/.config/image-factual-verifier/runtime.env"
chmod 600 "$HOME/.config/image-factual-verifier/runtime.env"
# 编辑 runtime.env；只填目标服务器实际拥有的配置。
source scripts/server/ifv_env.sh
scripts/server/bootstrap_runtime.sh
scripts/server/run_ifv.sh python scripts/server/doctor.py --json
```

`scripts/server/ifv_env.sh` 会读取未跟踪的
`~/.config/image-factual-verifier/runtime.env`，设置数据根目录、缓存及 loopback
`NO_PROXY`。所有项目进程必须满足 `OMP_NUM_THREADS=1`。

## 4. 运行时 API 配置

### 4.1 大型 Qwen teacher

将下列值写入目标服务器的未跟踪 `runtime.env`：

```bash
QWEN_TEACHER_BASE_URL=http://teacher-host:port/v1
QWEN_TEACHER_API_KEY=provided-out-of-band
QWEN_TEACHER_MODEL=served-teacher-model
QWEN_TEACHER_VISION_MODEL=served-teacher-model
IFV_TEACHER_ROLLOUT_PROFILE=teacher-qwen-server
IFV_TEACHER_ROLLOUT_MODEL=served-teacher-model
```

endpoint 必须支持：`/v1/models`、`/v1/chat/completions`、图片输入、原生 tool calls、
同一 interaction history 的多轮调用和真实 reasoning/`<think>` 输出。不得由 runtime
补写不存在的 thought，`/v1/models` 中的 model ID 必须与配置一致。

同一个 endpoint 固定承担：主 Agent ReAct/最终 Judgment、`perceive_scene` 和局部视觉
检查、以及 `visit` 抓取网页后的结构化 Evidence 抽取。启动器将其设置为：

```bash
BROWSE_EXTRACT_PROVIDER=qwen_local
BROWSE_EXTRACT_MODEL=served-teacher-model
BROWSE_EXTRACT_BASE_URL=http://teacher-host:port/v1
BROWSE_EXTRACT_WIRE_API=chat_completions
```

主 Agent、视觉工具和网页 Evidence 抽取必须使用同一个 teacher model ID。便携启动器
会拒绝与 `IFV_TEACHER_ROLLOUT_MODEL` 不同的 `QWEN_TEACHER_VISION_MODEL`，并拒绝
隐式回退到旧的本地 Qwen endpoint 或凭据。

不要配置 Gemini 作为 fallback，也不要混用 `NECODEX_*`、`LMDEPLOY_*` 或小 Qwen 变量。

### 4.2 Agent 工具 API

```bash
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
OSS_KEY_PREFIX=image-search
```

- Serper：文本搜索、图片搜索和 Lens。
- Jina：网页抓取和搜索候选 rerank。
- 百度 OCR：唯一 OCR 后端。
- OSS：为 Lens 提供可访问的临时图片 URL。

若不用 OSS，必须显式选择 `IMAGE_UPLOAD_PROVIDER=custom` 并配置
`IMAGE_UPLOAD_API_URL`，或显式选择 `temp`；不能静默切换 provider。

### 4.3 SFT eligibility judge

默认复用同一个 teacher endpoint：

```bash
IFV_SFT_ELIGIBILITY_PROVIDER=qwen_local
IFV_SFT_ELIGIBILITY_BASE_URL=http://teacher-host:port/v1
IFV_SFT_ELIGIBILITY_MODEL=served-teacher-model
IFV_SFT_ELIGIBILITY_WIRE_API=chat_completions
IFV_SFT_ELIGIBILITY_ENABLE_THINKING=true
```

启动器会将 `QWEN_TEACHER_API_KEY` 映射给复用 endpoint 的 judge，因此复用时无需在
命令行再次传入密钥。使用独立 judge 时才配置：

```bash
IFV_SFT_ELIGIBILITY_BASE_URL=http://judge-host:port/v1
IFV_SFT_ELIGIBILITY_MODEL=served-judge-model
IFV_SFT_ELIGIBILITY_API_KEY_ENV=IFV_SFT_ELIGIBILITY_API_KEY
IFV_SFT_ELIGIBILITY_API_KEY=provided-out-of-band
```

pipeline state 只能记录 API key 的环境变量名，不能记录密钥值。

### 4.4 其他配置

```bash
IFV_DATA_ROOT=/absolute/server/data/root
IFV_OMP_NUM_THREADS=1
MODELSCOPE_API_TOKEN=provided-out-of-band
```

`MODELSCOPE_API_TOKEN` 只用于下载训练集归档；若仓库匿名可读则不需要。

## 5. 下载与数据隔离

```bash
cd "$IFV_REPO_ROOT"
source scripts/server/ifv_env.sh
scripts/server/prepare_factcheck_dataset.sh --split train

wc -l "$IFV_DATA_ROOT/datasets/factcheck_train-8490-20260907/train-manifest.jsonl"
find "$IFV_DATA_ROOT/datasets/factcheck_train-8490-20260907/images" -type f | wc -l
test -f "$IFV_DATA_ROOT/datasets/factcheck_train-8490-20260907/evaluator_private/private-gold-v1/train-private-gold.jsonl"
```

验收必须为 8,490 条 manifest 和 8,490 张图片。测试集不得出现在 rollout 的
`dataset-root`、`train-manifest`、gold sidecar 或 SFT package 中。

在第一次 teacher 请求发出前，autopilot 会从完整训练候选集冻结
`case-split/case_split.jsonl`。默认 full run 固定 8 个 validation case；10 条 smoke
固定 1 个。`preparation.json` 同时记录 runtime、private gold 和 case split 的 SHA-256，
恢复运行时任何一个文件或 split manifest 发生漂移都会直接失败。SFT 导出只消费这份
预先冻结的 split，不得在 accepted release 之后重新挑 validation case。

## 6. 启动顺序

### 6.1 endpoint 预检

不要将 token 写入 shell history、日志或命令参数。先确认模型列表：

```bash
source scripts/server/ifv_env.sh
curl -fsS -H "Authorization: Bearer $QWEN_TEACHER_API_KEY" \
  "$QWEN_TEACHER_BASE_URL/models" | python -m json.tool
scripts/server/start_teacher_rollout_portable.sh --help
```

若 endpoint 不鉴权，按服务要求设置 `QWEN_TEACHER_API_KEY=none`。预检失败时，先修
endpoint、model ID 或权限；不得改代码绕过检查。

### 6.2 10 条训练集 smoke

本次交接必须分成两个阶段执行，**不得一开始直接使用无参数默认命令自动衔接全量**。
先以前台方式完整跑完 10 条训练集 smoke：

```bash
scripts/server/start_teacher_rollout_portable.sh \
  --foreground \
  --smoke-only \
  --output-dir <smoke-output-dir>
```

上述单个命令会自动执行完整 smoke 流水线。另一个 Codex 不得手工拆分、逐轮启动
rollout、单独触发 reroll、重复调用 judge 或手工拼接 SFT 导出。流水线内部自动完成：

1. 初始 10 条 teacher Agent rollout；
2. 每轮 rollout 后自动执行 strict trace audit 和 SFT judge；
3. 对上一轮被 SFT judge 拒绝或未完成的 case 自动执行最多 3 轮 quality reroll；
4. 每个未解决 case 最多产生 4 轮候选，不要对已经通过的 case 强制重复 rollout；
5. 自动生成 `accepted-release/`；
6. 自动生成 policy/perception 两套 `sft-training-package/`；
7. 完成转换后格式审计和 runtime media 完整性检查。

检查：

```bash
cat <smoke-output-dir>/pipeline-state.json
cat <smoke-output-dir>/accepted-release/accepted_release_manifest.json
cat <smoke-output-dir>/sft-training-package/MANIFEST.json
find <smoke-output-dir>/rollouts -type f -path '*/traces/*.json' | wc -l
find <smoke-output-dir>/audits -maxdepth 3 -type f -name '*.json' -print
```

逐条验收：

1. 输入 case 全来自 train manifest；
2. 每个 case 有终态 trace 或明确工程失败；
3. 每轮最多一个 native tool call；
4. `state.all_steps` 保留连续搜索、空搜索、访问失败和工具错误；
5. Judgment 读取 raw history，而不是 reducer 后的摘要；
6. 无匹配不能单独变成 fake 证据，工具错误不能变成事实证据；
7. strict trace audit 无 failure；
8. SFT judge 逐条完成，拒绝样本不进入 accepted release；
9. policy/perception 两套 SFT 数据结构审计通过；
10. 无 private-gold 泄漏、持续连接泄漏或遗留 rollout 子进程。

轨迹格式还必须单独检查：

1. 只检查正式 `accepted-release/trajectory_sft.jsonl` 和
   `sft-training-package/ms-swift-policy/*.jsonl`，不得从 `attempt/traces`
   手工制作 preview；
2. policy 行顶层使用 `tools`、`messages`、`images`；
3. assistant reasoning 使用真实 `<think>...</think>`，最终输出包含
   `<answer>...</answer>`；
4. tool call 与 tool response 按实际时序交替，tool arguments 保持目标
   Qwen/ms-swift 契约要求的字符串化 JSON；
5. `messages` 中 `<image>` 数量与顶层 `images` 数量一致；
6. 除初始图片外，实际调查中出现的 candidate、crop、focused view 必须能从
   archived runtime store 恢复；
7. `accepted-release` schema 必须为
   `ifv-accepted-teacher-release-v4`，并满足
   `runtime_store_archive.selected_count == accepted_case_count`；
8. SFT 只导出通过 judge 的 accepted 轨迹，不得把拒绝轨迹写入训练数据。

### 6.2.1 全量前 smoke 结果包

以上一体化 smoke 流水线结束后，另一个 Codex 必须先制作并交付一个完整结果包，
供用户检查实际轨迹和 SFT 格式。**结果包交付并得到用户明确确认之前，不得启动全量。**

结果包必须来自正式 smoke pipeline，不得重新运行 converter、手工修改轨迹或从
`attempt/traces` 拼接 preview。结果包至少包含：

```text
SMOKE-RESULTS.md
pipeline-state.json
preparation.json
rollouts/
classification/
accepted-release/
sft-training-package/
audits/
readable-episodes-zh/
```

具体要求：

1. `rollouts/`：保留 10 条首轮以及流水线自动产生的 reroll 轨迹、run manifest、
   score 和工程失败记录；
2. `accepted-release/`：必须完整包含 selected/rejected trace、正式冻结的
   selected/rejected eligibility 审计、
   `trajectory_sft.jsonl`、`runtime-stores/` 和 `runtime_store_index.jsonl`；
3. 不要复制顶层 `sft-eligibility/cache/`、judge 原始请求缓存或 private-gold
   sidecar；结果包所需的 judge 结论以 accepted release 内冻结的 eligibility
   artifact 为准；
4. `sft-training-package/`：完整包含 provider-neutral accepted dataset、
   ms-swift policy/perception 数据、manifest 和严格审计；
5. `readable-episodes-zh/`：只能由正式
   `accepted-release/trajectory_sft.jsonl` 使用
   `render_sft_episodes_zh_readable.py` 生成；
6. `audits/`：包含 strict trace audit、pipeline summary、policy/perception
   audit 和其他 smoke 阶段已产生的审计文件；
7. `SMOKE-RESULTS.md`：记录 commit、训练集版本、teacher/judge model、
   并发、初始与 reroll 轮次、成功/拒绝/工程失败数量、accepted SFT 行数、
   每条 SFT 的 messages/tool calls/images 数量及格式检查结论。

可读视图必须从正式导出生成：

```bash
python scripts/trajectory/render_sft_episodes_zh_readable.py \
  --input <smoke-output-dir>/accepted-release/trajectory_sft.jsonl \
  --output-dir <smoke-review-dir>/readable-episodes-zh
```

结果包不得包含：

- private gold、evaluator sidecar 或构造标签；
- API key、token、密码、完整环境变量转储；
- 手工修改后的 trace、手工补写的 thought 或非正式 preview；
- 测试集数据或测试集结果。

最终交付给用户时必须给出：

```text
smoke output directory
review directory
review tar.gz path
review tar.gz byte size
smoke commit
teacher model
judge model
accepted SFT row count
audit/format result
```

任一 rollout、reroll、SFT 导出、结构审计、过程图片或轨迹格式检查失败，都应停在
smoke，修复代码、提交并更新服务器 checkout 后从已有输出恢复。结果包交付后等待
用户检查；只有用户明确回复可以开始全量，才能进入下一节。

### 6.3 全量 8,490 条 rollout

用户检查 smoke 结果包并明确批准后，使用与 smoke 相同的 commit、teacher model、
judge model 和工具配置，启动完整 8,490 条训练集：

```bash
scripts/server/start_teacher_rollout_portable.sh \
  --full \
  --output-dir <full-output-dir>
```

当前默认 `rollout-concurrency=10`、`sft-concurrency=10`、最多三轮 quality reroll。
目标服务器确认 API 配额后，可显式调高：

```bash
scripts/server/start_teacher_rollout_portable.sh --full \
  --rollout-concurrency 16 \
  --sft-concurrency 64 \
  --quality-reroll-rounds 3
```

恢复已有任务时使用相同输出目录；不要删除已有 attempt：

```bash
scripts/server/start_teacher_rollout_portable.sh --full \
  --output-dir <existing-pipeline-dir>
```

成功 trace 和已完成 judge 不应被无意义重跑。失败 attempt、工程错误和 judge 拒绝由
入口的 retry/reroll 规则处理，所有原始 trace 和 interaction archive 都必须保留。

## 7. 产物、审计与导出验收

完整 pipeline 至少包含：

```text
pipeline-state.json
preparation.json
case-split/case_split.jsonl
case-split/manifest.json
rollouts/
sft-eligibility/
classification/
accepted-release/
sft-training-package/
audits/
```

`accepted-release/` 必须是 `ifv-accepted-teacher-release-v4`，其中每条入选
轨迹都要有对应的 `runtime_store_path`，且
`runtime_store_archive.selected_count == accepted_case_count`。这保证原始
rollout runtime 删除或迁移后，正式 converter 仍能恢复完整过程图片。

不要从 `rollouts/*/attempt-*/traces` 手工制作 preview。10 条 smoke 审阅包
直接使用 `<smoke-output>/accepted-release/trajectory_sft.jsonl`，或从
`<smoke-output>/sft-training-package/ms-swift-policy/` 生成可读视图。

```bash
test -f <pipeline-dir>/accepted-release/selected_episodes.jsonl
test -f <pipeline-dir>/accepted-release/runtime_store_index.jsonl
test -d <pipeline-dir>/accepted-release/runtime-stores
test -f <pipeline-dir>/case-split/case_split.jsonl
test -d <pipeline-dir>/sft-training-package/ms-swift-policy
test -d <pipeline-dir>/sft-training-package/ms-swift-perception
python -m ifv_training audit --strict \
  --input <pipeline-dir>/sft-training-package/ms-swift-policy
python -m ifv_training audit --strict \
  --input <pipeline-dir>/sft-training-package/ms-swift-perception
cat <pipeline-dir>/audits/pipeline-summary.json
cat <pipeline-dir>/audits/final-delivery.json
```

默认 `start_teacher_rollout_portable.sh` 和直接 `--full` 都会在完整 pipeline 结束后自动运行最终交付
门禁。门禁要求：`preparation.limit=null`、case 数为 8,490、最终 classification
完整覆盖全部 case、accepted release 与 selection 数量一致、policy train/validation
均非空、双 SFT 审计通过，且本任务没有误启动 GPU 训练。

也可手工复核已有完整目录：

```bash
python scripts/trajectory/verify_teacher_sft_delivery.py \
  --pipeline-dir <pipeline-dir> \
  --expected-case-count 8490
```

交接任务在以上审计和导出完成后停止。需要训练时，后续训练负责人用最终目标
checkpoint 另行执行 processor verification：

```bash
python training/scripts/probe/verify_ms_swift_agent_dataset.py \
  --model "$IFV_MODEL_ID" \
  --policy-dir <pipeline-dir>/sft-training-package/ms-swift-policy \
  --perception-dir <pipeline-dir>/sft-training-package/ms-swift-perception \
  --output <pipeline-dir>/sft-training-package/audits/processor-verification.json \
  --max-context 131072
```

该验证必须确认图片可接收、工具调用被目标模板正确渲染、工具结果进入上下文、
`<think>` 位于可训练 labels、每行存在非空 labels 且没有超出上下文；但它不属于
本次接手 Codex 的任务。换 checkpoint、processor、chat template 或 ms-swift 版本后，
由训练负责人重新验证。

当前 reasoning policy SFT 必须使用
`training/configs/models/qwen3.5-9b.env` 中的
`IFV_ADD_NON_THINKING_PREFIX=false`。历史缓存配置已归档到 Git 标签
`pre-deep-cleanup-20260910`，不得用于新导出的 reasoning SFT package。仓库当前只提供 mock GRPO 工程 smoke，
不应把它描述为已经打通的正式 RL 训练链路。

## 8. 失败处理

- `/v1/models` 不含配置 model ID：停止，修正模型配置，不硬编码绕过。
- teacher 请求失败：保留原错误和已有 trace，按工程 retry 规则恢复。
- 搜索为空：保留该轮原始结果；不能删除、压缩成“无证据”或自动判 fake。
- Jina/Serper/OCR/OSS 失败：是工具访问失败，不是事实证据。
- judge 非法 JSON 或请求失败：该条不能进入 accepted release；修复后用同一 trace/cache 重试。
- strict audit failure：查看对应 trace 和 audit JSON；不绕过审计、不手改产物。
- 服务器 checkout dirty：停止运行，确认差异来源；不得在服务器直接编辑源码。

## 9. 给另一个 Codex 的检查清单

```bash
cd "$IFV_REPO_ROOT"
git status --short --branch
git log -1 --oneline --decorate
source scripts/server/ifv_env.sh
scripts/server/run_ifv.sh python scripts/server/doctor.py --json
scripts/server/start_teacher_rollout_portable.sh --help
```

另一个 Codex 的执行任务必须严格按以下顺序完成：

1. 配置 API、仓库和训练集路径，确认 checkout 干净且使用交接指定 commit；
2. 只调用一次 `--foreground --smoke-only`；该命令内部自动完成 10 条初始
   rollout、拒绝样本最多 3 轮 quality reroll、逐轮 SFT judge、accepted release
   和双 SFT package，不得手工拆分执行；
3. 检查完整 trace，确认 raw history、工具结果、
   `<think>`、tool-call/tool-response 时序及二元 Judgment 均正确；
4. 检查正式 SFT 导出格式及所有过程图片，确认 runtime archive 可独立恢复
   candidate、crop 和 focused-view 图片；禁止使用手工 preview 代替；
5. 运行 policy/perception 严格结构审计；
6. 按第 6.2.1 节制作完整 smoke 结果包，生成中文可读轨迹、汇总说明和压缩包，
   并把路径和大小交给用户；
7. 停止并等待用户检查结果包；没有用户明确批准时不得执行 `--full`；
8. 用户批准后，使用同一 commit、模型和 API 配置启动 8,490 条全量 rollout，并持续监控，不得
   只启动进程后立即结束任务；
9. 等待全量工程 retry、SFT judge、最多 3 轮 quality reroll、accepted release
   和双 SFT package 全部完成；
10. 验证 `audits/final-delivery.json` 中 `final_delivery=true`，再交付完整审计与
   哈希清单。

到此停止；不要启动 processor verification、GPU SFT、RL、Direct QA 或测试集实验。
最终记录准确 commit、数据集版本、teacher model ID、endpoint、并发以及 smoke/full
产物路径；不得记录任何密钥、密码、private gold 或私有服务器地址。

最终汇报必须引用 `audits/final-delivery.json`。如果该文件不存在或
`final_delivery=false`，任务仍未完成，不能把中间 preview 回传为最终结果。

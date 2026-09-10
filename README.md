# Image Factual Verifier

## 并行任务交接

当前有两个并行任务，必须使用不同的输出目录和运行锁，不能互相覆盖：

1. **教师训练轨迹任务**：继续阅读并执行 `docs/teacher-rollout-and-sft-handoff.md`、`docs/teacher-rollout-api-handoff.md` 和 `docs/portable-deployment-handoff.md`。该任务已有进度，不得删除、重置或覆盖已有 rollout；按原流程完成 smoke、SFT 导出和后续全量工作。
2. **测试集 Agent 任务**：使用以下两个模型分别完整运行测试集 Agent，并分别交付 raw-history 轨迹和 `agent-results.jsonl`：
   - `GPT-5.5`
   - `Qwen3.5-397B-A17B`
   该任务不运行 SFT、RL、训练、SFT eligibility judge 或 private-gold judge；测试集 private gold 不得进入 Agent 请求。

测试集任务只能使用 `agent-test-*` 输出目录；教师任务使用其既有 pipeline 目录。启动前先检查是否已有同一 release/model 的运行进程，禁止重复启动。

### 代码入口

- Agent runtime：`src/orchestrator/react_runtime.py`
- Agent pipeline：`src/orchestrator/pipeline.py`
- 当前 prompt：`src/orchestrator/unified_prompts.py`
- 测试 rollout：`scripts/run_agent_test_rollout.py`
- 轨迹审计：`scripts/audit_real_trace.py`

测试 rollout 必须保留 raw tool history、工具响应、候选图片和完整 trace。不得手工拼接 trace、删除失败观察或从 preview 目录导出结果。

### 必需 API 配置

目标模型为 `GPT-5.5` 和 `Qwen3.5-397B-A17B`。两个模型必须分别配置、分别预检、分别运行；具体 endpoint、服务端 model ID 和 API 使用方式由任务发起人提供。收到配置后，确认 `/v1/models` 返回的 model ID 与实际服务要求一致；不要猜测 model ID，也不要把 key 写入仓库。

```bash
# 大 Qwen：主 Agent、视觉工具、网页 Evidence 抽取共用
QWEN_TEACHER_BASE_URL=https://<teacher-endpoint>/v1
QWEN_TEACHER_API_KEY=provided-out-of-band
QWEN_TEACHER_MODEL=<GPT-5.5_OR_QWEN3.5-397B-A17B_SERVER_MODEL_ID>
QWEN_TEACHER_VISION_MODEL=<SAME_SERVER_MODEL_ID>

# 文本搜索、图片搜索和 Lens
SERPER_API_KEY=provided-out-of-band

# 页面抓取
BROWSE_FETCH_PROVIDER=jina
JINA_API_KEY=provided-out-of-band

# 唯一允许的 OCR；禁止本地 OCR 和任何 fallback
OCR_BACKEND=baidu
BAIDU_OCR_API_KEY=provided-out-of-band
BAIDU_OCR_SECRET_KEY=provided-out-of-band

# Lens 临时图片上传
VISUAL_SEARCH_PROVIDER=serper_lens
IMAGE_UPLOAD_PROVIDER=oss
OSS_ACCESS_KEY_ID=provided-out-of-band
OSS_ACCESS_KEY_SECRET=provided-out-of-band
OSS_ENDPOINT=https://<oss-endpoint>
OSS_BUCKET_NAME=<private-bucket>
OSS_KEY_PREFIX=image-search
OSS_USE_SIGNED_URL=1
OSS_SIGNED_URL_EXPIRY_SECONDS=3600

IFV_DATA_ROOT=/absolute/server/data/root
OMP_NUM_THREADS=1
```

大 Qwen 同时承担主 Agent、视觉工具和网页 Evidence 抽取；不要启用 Gemini、本地 LMDeploy、本地 EasyOCR 或 OCR fallback。凭据只放服务器未跟踪 env 文件或进程环境。

### 启动前检查

```bash
set -euo pipefail
export OMP_NUM_THREADS=1
source scripts/server/ifv_env.sh
git status --short --branch
git log -1 --oneline --decorate
curl -fsS -H "Authorization: Bearer $QWEN_TEACHER_API_KEY" "$QWEN_TEACHER_BASE_URL/models" | python -m json.tool
```

确认 checkout 为交接 commit 且工作树干净，并确认 model ID 与两个教师变量一致后才能启动。服务器只允许拉取、fast-forward、安装和运行已提交代码，不得直接改服务器 checkout。

### 测试集运行

`--benchmark` 必须是标记为 `release_stage=development_subset` 且 `training_prohibited=true` 的测试 release，不得使用训练集。

```bash
python scripts/run_agent_test_rollout.py \
  --benchmark /absolute/server/test-release \
  --output-dir /absolute/server/runs/agent-test-<model>-<date> \
  --profile teacher-qwen-server \
  --concurrency 10 \
  --maximum-attempts 4
```

上面的命令必须对 `GPT-5.5` 和 `Qwen3.5-397B-A17B` 各执行一次；`<model>` 使用对应模型名的安全目录名，两个输出目录不得复用。启动前检查同一模型是否已有运行进程，禁止重复启动。

入口只做工程重试并保留所有 attempt；工程失败不能伪装成 `real/fake`。运行结束检查：

```bash
cat /absolute/server/runs/agent-test-<model>-<date>/summary.json
wc -l /absolute/server/runs/agent-test-<model>-<date>/agent-results.jsonl
find /absolute/server/runs/agent-test-<model>-<date>/rollouts -type f -name '*.json' | wc -l
```

交付 `summary.json`、`agent-results.jsonl`、`run-config.json`、`target-case-list.txt`、`rollouts/` 和重试记录，并注明 commit、model ID、endpoint（不含 key）、release、case 总数、成功数、工程失败数和结果目录。不得上传数据、图片、private gold 或任何 key。

测试集任务在完整结果包交付后结束；结果的二分类/LLM judge 由主任务另行处理。教师任务仍以其原交接文档规定的最终交付条件为准。

# gpu-13 / Image Factual Verifier 私有运维配置档

This project profile records the verified Image Factual Verifier deployment,
data, evaluation, and replay details for the `wza` account on `gpu-13`.

Internal only. Do not include this file in an external handoff package.

The shareable host-independent guide lives in
[`remote-jupyter-operations.md`](remote-jupyter-operations.md). It documents
only the generic SSH tunnel and Jupyter client workflow; it does not include
this private host or project configuration.

This profile intentionally contains no Jupyter password, API key, private key,
or other credential.

## Non-Negotiable Rules

1. Edit project source only in the local Windows checkout.
2. Commit locally and push the commit to GitHub.
3. On gpu-13, only clone, fetch, fast-forward, install, and run committed code.
4. Never edit source in the server checkout. The update and bootstrap scripts reject
   a dirty worktree.
5. Every project process on gpu-13 must run with `OMP_NUM_THREADS=1`.
6. Do not use `47.104.232.153` to transfer project files or datasets. It is only the
   SSH control endpoint. GitHub, Hugging Face, and Google Drive downloads originate
   from the server.
7. Keep credentials in an untracked server `.env` or process environment. Never put
   them in Git, shell scripts, notebooks, traces, or this document.

## Active runtime branch

The only supported branch for current Agent evaluation, Gemini teacher runs, and
gpu-13 rollout work is:

```text
codex/gpu13-canary-20260804-plan-relaxation-01
```

The maintained local source checkout is
`C:\Users\wangza\ifv-gpu13-canary-20260804-01`. The maintained server checkout is:

```text
/gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01
```

Do not use `codex/image-factual-verifier-v3`,
`codex/image-factual-verifier-v4`, `codex/image-factual-verifier-v4-minimal`, or
`feature/visual-fact-search-agent` as the current runtime baseline. Those branches
are historical or experimental lines and must not be substituted for the canary
without an explicitly recorded reproduction plan.

Before every server run, verify the checkout before starting any provider or
evaluation process:

```bash
git branch --show-current
git status --short --branch
git log -1 --oneline --decorate
```

The first command must print the active canary branch. If it does not, stop and
repair the checkout with the maintained update script; do not continue on the
wrong branch. The script's default branch is intentionally the active canary:

```powershell
python scripts/server/jupyter_remote.py --kernel-name ifv-agent --shell `
  'cd /gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01 && git branch --show-current && git status --short --branch'
```

Historical commits such as `fe43bdd` and `0738e59` are useful for reproducing
older measurements, but they are not the current branch baseline. Record the
exact commit in every run manifest and do not infer the runtime version from the
model name alone.

### SFT eligibility behavior

`176a8a6` changes frozen SFT eligibility from an all-or-nothing
`protocol_rejections` exclusion to an automatic recovered-trajectory assessment.
The judge receives compact rejected-turn history and returns `clean`,
`recovered_minor`, `degraded_repetition`, or `unresolved`. Only clean and
materially recovered trajectories may pass; repeated or unresolved blocked behavior
remains in rejected trace storage. `0961205` isolates automatic SFT storage by
eligibility output version, so a rerun cannot collide with an earlier scored version
of the same rollout. These are SFT export changes, not Agent rollout-version claims.

The historical result mapping is maintained in
[`gpu13-version-registry.md`](gpu13-version-registry.md). In particular,
`fe43bdd` is the accepted construction high baseline and `0738e59` is the
accepted EF3 high baseline; `f9d3a28` is an older 8/10 result and must not be
described as the high baseline. The combined 9/10 figure is a capability-group
aggregate across those two commits, not one 10-case run from a single version.

The maintained gpu-13 wrappers automatically set `IFV_ENV_FILE` to the following
private file when it exists:

```text
/gs/home/wza/.config/image-factual-verifier/runtime.env
```

The Python runtime loads that dotenv file without overriding explicitly supplied
process variables. The wrapper exports only the file path; it does not print or copy
credential values. Use mode `600` for this file.

### 当前统一数据集与教师 rollout（2026-08-25）

当前唯一可用于教师 rollout 的统一数据集是：

```text
/gsdata/home/wza/image-factual-verifier-v2-data/datasets/route-aware-hrc-stage2-10563-final-organized-20260824-r2/unified-dataset
```

总账与隔离边界：

- 总记录：10,174；
- `test-manifest.jsonl`：1,684 条，禁止进入 teacher rollout、SFT 或 RL；
- `train-manifest.jsonl`：8,490 条，是当前教师自动链的唯一输入；
- 图片均已存在；7,972 条有生图 prompt，2,202 条网页图等不需要 prompt。

当前正式教师自动链目录为：

```text
/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/gemini37-r2-train8490-p10-20260824
```

目录名中的 `p10` 是最初创建时的命名，不代表现在的实际并发。以
`pipeline-state.json` 和运行中的命令行为准：2026-08-25 已确认 rollout 并发为
24、SFT judge 并发为 10。当前代码提交为 `75e7181`。

自动链已完成一次完整 runtime/private-gold 投影与图片 SHA-256 校验；每次
`attempt-XX` 调用都显式跳过重复的 `run_cases` 全量图片预检。不要把这一优化复制到
普通手工 `run_cases` 命令中，除非该命令使用的正是已经验证过的同一投影。

当前全量任务只能通过
`scripts/trajectory/run_teacher_rollout_autopilot.py` 恢复。它会从不可变的已有
attempt 中收集成功 trace，只运行未完成或工程失败的 case；成功 case 不会被重跑。
完整链路、质量桶与 reroll 规则见
[`../teacher-rollout-autopilot.md`](../teacher-rollout-autopilot.md)。

检查任务是否真的在运行：

```bash
RUN=/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/gemini37-r2-train8490-p10-20260824

cat "$RUN/pipeline-state.json"
ps -eo pid,ppid,etime,args | grep -E '[r]un_teacher_rollout_autopilot|[s]rc\.eval\.run_cases'
find "$RUN/rollouts/initial" -type f -path '*/attempt-*/traces/*.json' | wc -l
ss -tan state close-wait | tail -n +2 | wc -l
```

恢复扫描期间只有 autopilot 进程、但暂时没有 `run_cases` 子进程是正常的。只要扫描完成，
它会新建后续 attempt 和 case list。`CLOSE-WAIT` 短暂存在不等于故障；重点是它是否随
完成 trace 数持续无界增长。相关生命周期事故和修复记录见：

- [`2026-08-23-gemini-rollout-resource-lifecycle.md`](2026-08-23-gemini-rollout-resource-lifecycle.md)
- [`2026-08-24-close-wait-session-leak-followup.md`](2026-08-24-close-wait-session-leak-followup.md)

### 历史：1051 条 archive 训练池与重跑（2026-08-23）

下列内容记录的是 2026-08-23 的 1051 条 archive 训练池和针对 archive 的排障，
不是当前 8,490 条统一训练集的正式输入。

The registered training pool for that historical teacher rollout was:

```text
/gsdata/home/wza/image-factual-verifier-v2-data/training/remaining-after-qwen35-route-balanced-bacc68p7-1687-20260821-r6
```

It contains 1051 training cases. The associated test set is
`test_sets/qwen35-route-balanced-bacc68p7-1687-20260821-r6` and must not be used
as rollout input.

The pool metadata records the logical archive source
`/gsdata/home/wza/image-factual-verifier-data-pipeline-data/archives/route-aware-hrc-final-3000-20260816`.
That exact directory is not materialized on gpu-13. The physical archive for this
3000-case source is:

```text
/gsdata/home/wza/image-factual-verifier-data-pipeline-data/archives/history/2026-08-17/route-aware-hrc-final-3000-20260816
```

The physical archive has 2800 candidate rows and 3000 archived images. Before a
rollout, select the case IDs from the training pool and verify that every selected
`archive_source_version_id` exists in `human-review-candidates.jsonl` and that its
`archive_image_path` exists under `artifacts/images/`. Do not substitute the
`route-aware-hrc-stage2-10563-20260821-baseline7583-v4-full` archive: it is a
different aggregate archive, even though it contains related upstream records.

The rollout must receive only the archive adapter's runtime projection. For the
post-rollout frozen SFT judge, build a separate private adapter from the matching
candidate rows, adding `case_id=archive_source_version_id` as the judge key.
That adapter may contain factual status, claims, and evidence, but it must never
be passed to the Agent or copied into model-visible training rows.

Important selection detail: `src.eval.run_cases` validates every ID supplied by
`--case-list` before applying `--limit`. Therefore, when running a bounded prefix
such as the first 100 training cases, first materialize a temporary case list
containing exactly those 100 IDs. Passing the full 1051-case list together with
`--limit 100` still validates all 1051 IDs and can fail before any rollout starts.

### Historical archive rollout entrypoint (重要)

历史 archive 重跑必须使用 `src.eval.run_cases --archive-root`，不能使用
`scripts/server/start_gemini_eval_gpu13.sh`。后者内部固定启动
`python -m src.eval.run_eval`，只接受 v0.3 release 的 `--benchmark`，不接受
`--archive-root`。把 archive 参数传给该 wrapper 会出现两种容易误判的现象：
启动器先打印 PID/run_id，但后台进程随后在 argparse 阶段以状态码 2 退出；
不会生成 `traces/*.json`，也不会发出 Gemini 请求。

正确的 archive 启动方式是直接调用 `run_gpu13.sh`，并让 archive runner 自己
获取 Gemini 全机锁：

```bash
cd /gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01
source scripts/server/gpu13_env.sh
export GEMINI_EVAL_MAX_CONCURRENCY=16
export GEMINI_MAX_INFLIGHT_REQUESTS=16
run_id="archive-rerun-$(date -u +%Y%m%dT%H%M%SZ)"
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m src.eval.run_cases \
  --archive-root /path/to/archive \
  --output-dir "$IFV_DATA_ROOT/runs/eval/$run_id" \
  --profile teacher-gemini \
  --concurrency 4 \
  --rollouts-per-case 1 \
  --base-sampling-seed 982451653 \
  --timeout 1800 \
  --case-id <case-id-1> \
  --case-id <case-id-2>
```

`start_gemini_eval_gpu13.sh` 仍然适用于 v0.3 release 的正式后台评测：

```bash
scripts/server/start_gemini_eval_gpu13.sh \
  --benchmark "$IFV_DATA_ROOT/releases/<release-id>/runtime_input/cases.jsonl" \
  --output-dir "$IFV_DATA_ROOT/runs/eval/<run-id>" \
  --concurrency 4
```

两类输入不得混用。启动前先做入口检查：

```bash
case "$INPUT_MODE" in
  archive)  test -f "$ARCHIVE_ROOT/human-review-candidates.jsonl" ;;
  release)  test -f "$RELEASE_ROOT/manifest.json" && \
            test -f "$RELEASE_ROOT/runtime_input/cases.jsonl" ;;
esac
```

后台启动器打印 PID 只代表 shell 已经 fork，不能代表 rollout 已开始。实际运行
必须同时检查：

```bash
test -f "$RUN_DIR/run_manifest.json"
find "$RUN_DIR/traces" -maxdepth 1 -type f -name '*.json' | wc -l
test -f "$RUN_DIR/summary.json" && cat "$RUN_DIR/summary.json"
tail -n 80 "$LOG_FILE"
```

判定规则：

- 只有 `traces/*.json` 数量开始增加，才算进入实际 rollout；
- 只有 `summary.json` 存在且 `num_errors` 已确认，才算批次结束；
- `run_manifest.json` 缺失或后台日志包含 `run_eval.py: error` / `exit_status=2`，
  归类为启动参数错误，不归类为 Gemini/API 失败；
- `Lmod` 关于 `gnu12` 与 `gnu9` 的提示来自 conda/module 环境初始化。它本身
  不是 rollout 结果；仍需继续检查 Python 进程、trace、summary 和最终退出码。

2026-08-23 实际排查记录：对 6 条被 SFT LLM judge 拒绝的 archive case 重跑时，
前两次误用了 `start_gemini_eval_gpu13.sh --archive-root`，后台分别在参数解析
阶段退出，没有生成 trace，也没有消耗 Gemini 请求。第三次改为
`src.eval.run_cases --archive-root`，使用全新 sampling seed 和独立 output 目录，
并发 4；该任务的正确 run id 是
`gemini-sft-retry6-seed982451653-20260823`。旧的失败目录/日志不得当作 rollout
失败样本，后续比较只读取正确 archive runner 生成的 trace。

该批次中 case `...:0006:baseline_historical_generated_drain-generated-no-prototype-0000`
首轮因 `perceive_scene` 的 Gemini HTTP 504 产生工程错误。单独使用正确的
archive runner、并发 1、VLM timeout 180 秒、`VLM_TOOL_REQUEST_MAX_RETRIES=3`
重跑后成功：

```text
run_id=gemini-sft-retry0006-seed982451653-r1-20260823
termination=success
verdict=fake
engineering_error=0
llm_api_calls=41
tool_calls=16
time_taken_sec=372.33
```

这次对照证明该 504 是可恢复的外部视觉请求失败，不应直接归类为轨迹质量失败；
但重跑后的完整轨迹仍必须重新经过 frozen SFT LLM judge，不能只依据最终 verdict
判定可训练。

该 6-case 重跑的完整配对结果如下。5 条首轮成功轨迹先统一经过
`score_sft_eligibility.py`，随后 0006 的工程错误单独修复重跑并再次经过同一
LLM judge：

```text
case 0001  rollout=fake  judge=pass
case 0002  rollout=fake  judge=pass
case 0003  rollout=real  judge=reject
case 0005  rollout=real  judge=reject
case 0006  rollout=fake  judge=pass  # 首轮 perceive_scene 504，重试后恢复
case 0010  rollout=real  judge=reject
```

因此新 seed 的 6 条重跑结果是：

```text
最终有效轨迹：6/6
正确 verdict：3/6
frozen SFT LLM judge 通过：3/6
仍拒绝：3/6
最终工程错误：0
```

重跑 judge 产物：

```text
5-case judge:
/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/sft-eligibility-gemini-sft-retry6-seed982451653-20260823

0006 judge:
/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/sft-eligibility-gemini-sft-retry0006-seed982451653-r1-20260823
```

结论：换 sampling seed 确实能产生不同轨迹，并且本批次把 0006 从工程错误恢复为
可用轨迹；但对模型本身判断能力的改善不是稳定的，3 条错误 verdict 仍被 judge
正确拒绝。因此“重新 roll”可以作为保留轨迹、提高正样本产率的策略，但不能替代
SFT LLM judge；生产流程应保留每个 case 的多个独立 rollout，再按 judge 结果和质量
桶选择训练候选。

## Verified Topology

Verified on 2026-07-11:

```text
Windows workstation
  -> SSH 47.104.232.153:2429 as wza (control endpoint; observed host gpu-16)
  -> local 127.0.0.1:8333 forwards to jump-host 127.0.0.1:8333
  -> existing jump-host forward reaches Jupyter on gpu-13:8333
  -> Jupyter REST/WebSocket executes as wza on gpu-13
```

The Jupyter endpoint returned Tornado login responses, accepted the configured
password, and successfully executed a temporary kernel on `gpu-13`. The base
`python3` kernel is only a control-plane kernel. Project work must use the
`ifv-agent` kernel registered by `bootstrap_gpu13.sh`; it starts the isolated
Python 3.11 environment through `ifv_agent_kernel_gpu13.sh`, which applies the
same proxy, data-root, cache, and `OMP_NUM_THREADS=1` guard as project wrappers.
The wrapper also appends `127.0.0.1`, `localhost`, and `::1` to both `NO_PROXY`
and `no_proxy`. Without this loopback exemption, inherited proxy settings can turn
a healthy local vLLM `/health` or `/v1/models` request into a proxy-generated 503.

The old local `9814` route belongs to the previous server workflow. On the current
path the service behind remote `127.0.0.1:9814` was unavailable; use local `8333`.

## Start The Local Tunnel

Run this on Windows. `-F NUL` avoids unrelated forwards from the user's SSH config.

```powershell
ssh -F NUL -N `
  -L 127.0.0.1:8333:127.0.0.1:8333 `
  -p 2429 `
  -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=30 `
  -o ServerAliveCountMax=3 `
  wza@47.104.232.153
```

Then open `http://127.0.0.1:8333/tree`. Supply the Jupyter password out of band.

For non-interactive control, install the optional local dependency and use the
committed client:

```powershell
python -m pip install websocket-client
$env:JUPYTER_REMOTE_BASE = "http://127.0.0.1:8333"
python scripts/server/jupyter_remote.py --kernel-name ifv-agent --shell "hostname; id -un"
```

The client intentionally has no built-in server URL. Set
`JUPYTER_REMOTE_BASE` in every new PowerShell process or pass `--base`
explicitly; this keeps the shareable Python client free of private topology.

The Windows workstation also has a local companion copy at:

```text
D:\wangza\Desktop\jupyter_remote.py
```

Treat `scripts/server/jupyter_remote.py` in the repository as the maintained source
of truth. The desktop copy is an older convenience entry point: it starts `python3`
when no kernel ID is supplied and does not support `--kernel-name`. Therefore:

- prefer the repository client and always pass `--kernel-name ifv-agent` for project
  work;
- if the desktop copy must be used, first create or list an `ifv-agent` kernel with
  the repository client, then pass that existing ID through `--kernel`;
- never let the desktop copy create its default `python3` kernel for project tests,
  evaluation, installation, or data work;
- verify `hostname` prints `gpu-13`, `id -un` prints `wza`, and
  `OMP_NUM_THREADS` is `1` before running project commands.

Example using a known `ifv-agent` kernel ID:

```powershell
python "D:\wangza\Desktop\jupyter_remote.py" `
  --kernel "<ifv-agent-kernel-id>" `
  --shell `
  "hostname; id -un; echo `$OMP_NUM_THREADS"
```

The client prompts for the password without echo. For unattended automation, inject
`JUPYTER_REMOTE_PASSWORD` from a secret manager for that process only; do not persist
it in a profile or script.

### PowerShell-to-Bash quoting rule

When invoking `jupyter_remote.py --shell` from PowerShell, pass the complete remote
Bash command as one single-quoted PowerShell argument. Never place Bash variables,
command substitutions, or escaped double quotes inside a double-quoted PowerShell
string. PowerShell expands `$release`, `$out`, `$env`, and `$(...)` locally before
the command reaches gpu-13, which can split the positional command and silently
replace remote paths with empty strings.

Forbidden:

```powershell
python scripts/server/jupyter_remote.py --kernel-name ifv-agent --shell `
  "release=/remote/release; out=/remote/run; command --input `"$release/file`" --output `"$out`""
```

Required—prefer explicit absolute paths:

```powershell
python scripts/server/jupyter_remote.py --kernel-name ifv-agent --shell `
  'cd /gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01 && scripts/server/run_gpu13.sh command --input /absolute/remote/input --output /absolute/remote/output'
```

If remote Bash variables are genuinely useful, the outer PowerShell argument must
still be single-quoted:

```powershell
python scripts/server/jupyter_remote.py --kernel-name ifv-agent --shell `
  'release=/absolute/remote/release; out=/absolute/remote/run; command --input "$release/file" --output "$out"'
```

Inside that outer single-quoted PowerShell argument, write Bash double quotes
normally as `"..."`. Do not write `\"...\"`: the backslashes reach Bash literally,
prevent the quotes from grouping metacharacters, and can turn `|`, `&&`, or spaces
inside an intended argument into separate shell syntax.

Before executing a long or destructive remote command, first use the same quoting
form with a harmless `printf` or path-existence check. Treat any
`unrecognized arguments` error from `jupyter_remote.py` as a local quoting failure;
the intended remote command did not run.

### Long Jupyter commands

`jupyter_remote.py` uses a 120-second execution timeout by default. Pass a larger
explicit timeout for a foreground replay, provider probe, or other command that can
legitimately exceed that control-plane limit:

```powershell
@'
cd /absolute/remote/checkout
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent python scripts/replay_snapshot_discrepancy.py ...
'@ | python scripts/server/jupyter_remote.py --kernel-name ifv-agent --shell `
  --stdin --timeout 240
```

The timeout is for the Windows-to-Jupyter control client, not permission to run
unbounded work. Keep the runtime's own bounded action, provider, and stage limits
in effect.

On 2026-07-15, the `8333` path was authenticated and verified with
`hostname=gpu-13`, `id -un=wza`, and the `ifv-agent` kernel. Credentials were supplied
only to the controlling process and were not written to the checkout, logs, or this
document.

`jupyter_remote.py` defaults to `python3` to preserve control-plane access before
the environment is bootstrapped. Set `--kernel-name ifv-agent` or
`JUPYTER_REMOTE_KERNEL=ifv-agent` for every project command. The bootstrap process
installs `ipykernel` only when missing and registers/replaces the `ifv-agent`
kernelspec; no Jupyter server restart is required.

## Required Server Runtime Environment

The server's `.bashrc` and an older Jupyter process were observed to contain the
obsolete proxy port `47894`. Do not rely on either environment. The committed
gpu-13 wrappers ignore the legacy `IFV_SERVER_PROXY` variable and use:

```bash
export http_proxy=http://100.10.1.210:47899
export https_proxy=http://100.10.1.210:47899
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export OMP_NUM_THREADS=1
export IFV_DATA_ROOT=/gsdata/home/wza/image-factual-verifier-v2-data
```

Only an explicitly announced proxy migration should set
`IFV_SERVER_PROXY_OVERRIDE`. This distinct name prevents a stale inherited
`IFV_SERVER_PROXY=...:47894` value from silently breaking GitHub and provider access.

GPU selection is dynamic. Do not assume or reserve specific physical GPU IDs;
before starting a task, inspect current utilization, free memory, and running
processes with `nvidia-smi`, then choose an available card without terminating
another user's process.

The following were verified through that proxy:

- GitHub HTTPS and repository `git ls-remote`;
- Hugging Face HTTPS;
- Google Drive HTTPS.

OCR is selected explicitly with `OCR_BACKEND`; the default is `baidu`.
Baidu uses the general OCR API with an in-memory access-token cache. EasyOCR
is an explicit local diagnostic backend only; it uses one process-shared CPU
reader and serializes calls because a reader instance is not assumed to be
thread-safe. There is no silent backend fallback: a missing credential, failed
request, malformed result, or runtime exception is an explicit
`ocr_with_position` tool failure.

For the local backend:

```bash
OCR_BACKEND=easyocr
EASYOCR_GPU=false
```

For Baidu general OCR, keep all values in the untracked runtime environment
file or the process environment:

```bash
OCR_BACKEND=baidu
BAIDU_OCR_API_KEY=<api-key>
BAIDU_OCR_SECRET_KEY=<secret-key>
BAIDU_OCR_CONNECT_TIMEOUT_SECONDS=10
BAIDU_OCR_READ_TIMEOUT_SECONDS=120
BAIDU_OCR_MAX_RETRIES=0
BAIDU_OCR_MAX_EDGE=4096
BAIDU_OCR_MAX_UPLOAD_BYTES=4500000
```

`BAIDU_OCR_ACCESS_TOKEN` may be supplied for a controlled run instead of the
API key/secret pair. Tokens are never written to traces or source files.

The Baidu path converts the image or requested crop to JPEG before submission,
limits its longest side to 4096 pixels, and bounds the uploaded JPEG to 4.5 MB
so the base64/form-encoded request remains below the provider limit. OCR
coordinates are mapped back to the original image. `BAIDU_OCR_TIMEOUT_SECONDS`
remains a compatibility alias for the read timeout. Retries are off by default:
a timeout may mean the provider received the image but the response was lost.
If a controlled retry is needed, set `BAIDU_OCR_MAX_RETRIES=1`; the trace records
the actual provider request count. Non-JSON responses record only HTTP status,
content type, byte count, and a body digest, never the raw body.

The latency probe reports the shared-reader initialization separately from
steady-state calls and should use a small repeat count:

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python scripts/benchmark_ocr_profiles.py --repeats 1 \
  /path/to/image.jpg
```

The configured OCR backend is used only for visible-text observations:

- detected text strings;
- quadrilateral and axis-aligned image coordinates;
- recognition confidence and a simple language label.

It does not establish the truth of the text's implied real-world claim. Positive
OCR observations may anchor a visible fact; missing or low-confidence text remains
inconclusive.

Use `scripts/server/run_gpu13.sh` for project commands. It always sources the
required proxy, credential-file path, and threading environment and fails if the
OMP value is not `1`.

## Server Data Root

All datasets and generated runtime artifacts live on the gpu-13 data filesystem:

```text
/gsdata/home/wza/image-factual-verifier-v2-data/
  datasets/       downloaded archives and extracted datasets
  artifacts/      web pages, images, RIS/SERP snapshots, and source material
  benchmarks/     benchmark manifests and benchmark-owned assets
  cache/          Hugging Face, Torch, and tool caches
  runs/
    _logs/        background evaluation stdout and stderr logs
    traces/       standalone JSON Agent trajectories
    eval/         named evaluation runs with predictions, summaries, and JSON traces
  generated/      later synthetic training and diagnostic data
```

`/gs/home/wza/gsdata` resolves to `/gsdata/home/wza`; the latter had about 399 TB
available during the deployment audit. `gpu13_env.sh` exports `IFV_DATA_ROOT` and
routes Hugging Face, Torch, and tool caches into this tree. Runtime and data
scripts also derive their default output paths from `IFV_DATA_ROOT`.

Do not download datasets, write evaluation traces, or generate images inside the Git
checkout. The repository contains code, schemas, documentation, and small reviewed
manifests only.

## First Deployment

Push the desired branch from the local repository first:

```powershell
git push -u origin codex/gpu13-canary-20260804-plan-relaxation-01
```

Then execute the following on gpu-13 through Jupyter or an approved terminal. The
clone is downloaded by gpu-13 from GitHub, not copied over the SSH endpoint.

```bash
export http_proxy=http://100.10.1.210:47899
export https_proxy=http://100.10.1.210:47899
export OMP_NUM_THREADS=1

mkdir -p /gs/home/wza/projects/image-factual-verifier-v2-worktrees
checkout=/gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01
git clone --branch codex/gpu13-canary-20260804-plan-relaxation-01 \
  https://github.com/wangzian828/image-factual-verifier-v2.git \
  "$checkout"
cd "$checkout"
bash scripts/server/bootstrap_gpu13.sh
```

The active canary checkout verified on 2026-08-05 is:

```text
/gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01
```

The former `/gs/home/wza/projects/image-factual-verifier-v2` and
`visual-fact-search-agent` server checkouts were removed on 2026-08-16. They are
not valid runtime paths and must not be recreated for normal work. The
`run_gpu13.sh` wrapper now rejects any non-canonical path or branch before starting
a project process.
The currently verified canary branch is
`codex/gpu13-canary-20260804-plan-relaxation-01`; its exact HEAD is intentionally
not recorded here because it changes after every deployment. The server source tree
must not be reset or reused when it is dirty or on a different branch than the
committed local work.

### Gemini-only separated visual mode

The runtime supports two explicit image-access modes:

- `direct_multimodal`: historical behavior; the main Gemini LLM receives the
  image at image-aware planning and final judgment.
- `separate_vlm`: the Gemini VLM performs perception and one final visual audit;
  the main Gemini LLM receives only structured perception/audit and never the
  original image at planning, route-local replan, or final judgment.

Select the separated mode explicitly with:

```bash
--image-access-mode separate_vlm
```

The final VLM audit is recorded as `image_only_final_visual_audit`. It is visual
context for judgment, not a policy action, Finding, Evidence record, or
investigation-budget action.

The deployment uses the isolated `ifv-agent` environment with Python 3.11. The
bootstrap script is idempotent and stores `OMP_NUM_THREADS=1` in that Conda
environment as an additional guard. It also installs the `ifv-agent` Jupyter
kernelspec, whose wrapper sources `gpu13_env.sh` before launching the kernel. This
ensures browser notebooks and REST/WebSocket-launched project commands retain the
same runtime environment and places the `ifv-agent` binary directory first in
`PATH`, so shell cells also invoke the project interpreter rather than the Jupyter
server's base Conda Python.

## Update From GitHub

After each local commit and push:

```bash
cd /gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01
bash scripts/server/update_gpu13_checkout.sh codex/gpu13-canary-20260804-plan-relaxation-01
bash scripts/server/bootstrap_gpu13.sh
```

The updater allows only a fast-forward from `origin/<branch>` and refuses a dirty
server worktree. Do not bypass that guard by editing or resetting server files.

## Test And Run

Run the full deterministic suite without credentials first:

```bash
cd /gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q
```

Current traces report:

```text
total_tool_calls       policy-level tool actions
total_tool_subcalls    real provider requests inside those actions
tool_subcalls_by_kind  search, fetch, extract, OCR, upload, and comparison counts
```

The active runtime enforces one text query, up to three independently visited pages,
or one reverse-image branch per policy action. `crop_and_search` and `count_objects`
are not exposed to the Agent loop; general VLM anomaly checks remain diagnostic and
cannot create verdict Evidence.

These are contract and scripted-state checks. For real Gemini transport, create an
untracked `.env` on gpu-13 using a secure interactive method, then run:

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python scripts/probe_gemini_interactions.py --model gemini-3.7-flash
```

Keep benchmark datasets and caches under `IFV_DATA_ROOT`, outside the Git checkout.

For repeated rollouts, deterministic perception caching is enabled by default
for `perceive_scene` and `ocr_with_position`. Set
`PERCEPTION_CACHE_ENABLED=0` to disable it. Web-result caching remains opt-in
through `TOOL_CACHE_ENABLED=1`; do not enable that for freshness-sensitive
production runs without an explicit cache namespace and TTL.

Reference-image comparison now reuses downloaded reference data within a process,
with an LRU/TTL cache. Configure it with
`REFERENCE_IMAGE_CACHE_TTL_SECONDS` and `REFERENCE_IMAGE_CACHE_MAX_BYTES`.
Unchanged local image serializations reuse their file-metadata-keyed base64/JPEG
cache. Visual reverse search reuses an unexpired upload URL for the same local
image SHA-256; configure its lifetime with
`VISUAL_SEARCH_UPLOAD_CACHE_TTL_SECONDS`. These caches avoid repeated download,
encoding, and upload work only; they do not cache a new factual judgment.

The context ledger records per-request component sizes, an exact logical request
snapshot hash, provider token usage, and native Gemini retry metadata. Compare
`provider_input_tokens` against the component sizes in the request manifest when
evaluating context compaction; shrinking an archive file alone is not evidence
that the provider request became smaller.

Tool-internal Gemini vision requests record per-subcall durations for diagnosis.
Their established 8192-token output budgets remain the default: a controlled
same-input A/B on gpu-13 did not show that lower ceilings reduce latency.
Override either budget only after reproducing a provider-specific truncation or
latency improvement.

The active process shares one structured VLM client across the visual tools.
Gemini vision requests from those synchronous tool adapters run on one persistent
Python 3.11 async transport, so the HTTP connection can be reused across
perception, semantic image search, and visual reinspection calls. This changes
transport reuse only; it does not change prompts, output budgets, tool boundaries,
or evidence semantics.

Jina page extraction follows the same transport rule. Each process shares one
Jina Reader client between `visit` and `crop_and_search`; its Gemini evidence
extractor keeps one persistent async transport while independent page requests
remain concurrent. The page-content cache and HTTP sessions are shared as well,
so revisiting a candidate from another tool does not repeat the page fetch; an
identical claim/goal pair also reuses the completed extraction. This is an
implementation optimization only and does not turn a page preview into Evidence.

Dataset acquisition, review, and release finalization now belong to the separate
`image-factual-verifier-data-pipeline` project. Acquire a finalized release from
gpu-13 itself through Hugging Face, Google Drive, approved object storage, or a
mounted data path. Do not transfer dataset bytes through `47.104.232.153`, and do not
run construction code from this runtime checkout.

### Active automatic-diverse-20 v4 data

The server-side v4 release is the only active automatic-diverse-20 input:

```text
producer checkout:
/gs/home/wza/projects/image-factual-verifier-data-pipeline

construction:
/gsdata/home/wza/image-factual-verifier-v2-data/benchmarks/construction/
automatic-diverse-20-v4-evidence-chain-consolidated-20260715

development pilot:
/gsdata/home/wza/image-factual-verifier-v2-data/benchmarks/development/
automatic-diverse-20-development-pilot-v4-20260715

human review:
/gsdata/home/wza/image-factual-verifier-v2-data/benchmarks/reviews/
automatic-diverse-20-development-pilot-v4-20260715

runtime release:
/gsdata/home/wza/image-factual-verifier-v2-data/releases/
automatic-diverse-20-development-preview-v4-20260715

release audit:
/gsdata/home/wza/image-factual-verifier-v2-data/runs/release_audits/
automatic-diverse-20-development-preview-v4-20260715

construction run:
/gsdata/home/wza/image-factual-verifier-v2-data/runs/benchmark_pipeline/
automatic-diverse-20-v4-evidence-chain-gpu13-final-20260715
```

Validated on 2026-07-15:

```text
20 runtime cases
20 private gold rows
10 supported / 10 refuted / 0 unverifiable
three-field runtime rows only
all image hashes valid
all 27 release SHA-256 entries valid
current v3 release consumer accepts every case
source-access policy active with 3 excluded provenance URLs
```

The release is a `development_subset`, not a formal three-class benchmark. The
data-owned release audit's perfect predictions verify packaging/scorer consistency;
they are not Agent results.

### Frozen historical snapshot replays

`scripts/replay_snapshot_discrepancy.py` is a mechanism validator, not a
free-running evaluator. It restores a historical investigation snapshot, injects
only explicitly selected public source Evidence, runs the bounded Decision →
focused-pixel → Decision sequence, and writes a separate replay artifact.

Some archived reviewed-52 traces predate the active automatic-diverse-20 release.
For those snapshots, use the scoring release named by the historical run manifest,
not the current 20-case runtime release. The reviewed-52 replay verified on
2026-08-02 used:

```text
/gsdata/home/wza/image-factual-verifier-data-pipeline-data/benchmarks/releases/
ifv-scoring-gold-v1-reviewed-52-20260723/runtime_input/cases.jsonl
```

Write replay output under the data root (for example,
`/gsdata/home/wza/image-factual-verifier-v4/runs/replays/<run-id>/`) and never
overwrite the historical evaluation run or write artifacts into a Git checkout.

For `qwen_local` snapshot replays, the harness fills the local OpenAI-compatible
base URL when `--llm-base-url` / `--vlm-base-url` are omitted. For Gemini
Interactions replays, leave those base URL flags unset so the Gemini adapter uses
`GEMINI_INTERACTIONS_URL` or its official default; do not point Gemini at the local
Qwen `/v1` endpoint.

The frozen `process_reference_protocol.json` still names the older strict
acceptable-evidence and citation metrics. Current runtime scoring intentionally uses
chain-only recovery instead. Do not edit the release in place; correct this metadata
in the next producer release.

Local H-drive paths ending in
`automatic-diverse-20-development-pilot-v3-20260715` or
`automatic-diverse-20-development-preview-v3-20260715` are superseded. Final v4 has
not been synchronized to H and must not be run from those v3 directories. The active
local producer checkout remains:

```text
D:\image-factual-verifier-data-pipeline
```

Foreground canary:

```bash
cd /gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01
release="$IFV_DATA_ROOT/releases/automatic-diverse-20-development-preview-v4-20260715"
run_id="automatic-diverse-20-v4-canary-$(date -u +%Y%m%dT%H%M%SZ)"
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python scripts/run_real_canary.py \
  --benchmark "$release/runtime_input/cases.jsonl" \
  --source-access-policy "$release/evaluator_private/source_access_policy.json" \
  --output-dir "$IFV_DATA_ROOT/runs/eval/$run_id" \
  --limit 2
```

Only after the canary passes should the larger teacher rollout be launched with
the autopilot. The current Agent is `unified-react-v1`: one continuous ReAct
loop followed by one Judgment request. It does not issue standalone Planning,
Query Replan, Route Replan, Reflection, or Discrepancy Decision requests.

The runtime allows at most 24 accepted tool actions. All mature visual,
search, page, comparison, and inspection tools are ordinary ReAct tools from
the first turn; Gemini chooses which one to call and may choose visual tools
again later. Every turn is one thought plus one native tool call. The runtime
owns state, IDs, deduplication, budgets, failures, and termination.

In `direct_multimodal` mode, the root ReAct request receives one temporary
controlled image attachment. Later ReAct requests reuse that image through the
same Gemini Interaction history and add only the newest function result and
current observation; they do not upload the original image again. The default is
a JPEG with longest edge 1280 and quality 88; the image is not copied into text
history or persisted as base64. The independent final Judgment request receives
its own controlled image attachment. The context ledger stores only the
externalized media artifact and image metadata. In `separate_vlm` mode, visual
tools receive the image and the policy model receives structured observations
only. Both modes use the same dynamic tool schema, reducer, state delta, and
trace contract. Mature visual/search/browse tool implementations remain
unchanged.

Trace snapshots store a `runtime_image` reference, not image base64. A tool action
uses a short native `function_call -> function_result` round trip. ReAct actions
share one provider-side InteractionSession: the completed function result is
submitted to the next request exactly once, while earlier history remains
available through the provider session rather than being manually duplicated in
the new local context.
Gemini main-chain requests use a 90-second per-attempt HTTP timeout, twelve retries,
and a 900-second outer stage deadline by default. Retry delays grow exponentially,
include jitter, and honor provider `Retry-After` or `google.rpc.RetryInfo` hints up
to the configured maximum delay.
Configure them with `AGENT_LLM_REQUEST_TIMEOUT_SECONDS`,
`AGENT_LLM_REQUEST_MAX_RETRIES`, `AGENT_STAGE_REQUEST_TIMEOUT_SECONDS`,
`GEMINI_RETRY_BASE_DELAY_SECONDS`, `GEMINI_RETRY_JITTER_SECONDS`, and
`GEMINI_RETRY_MAX_DELAY_SECONDS`. Tool-internal vision calls keep their separate
`VLM_TOOL_REQUEST_TIMEOUT_SECONDS`, `VLM_TOOL_REQUEST_MAX_RETRIES`, and
`GEMINI_VISION_TIMEOUT_SECONDS` limits. This remains inside the 1,800-second
per-image budget: a transient 429 continues the same stored Interaction instead of
discarding completed Agent actions, while a provider outage that exceeds the
bounded retry window still produces a diagnostic error trace.
Coverage runs after every accepted action and stops when the target fact and its
required evidence gaps resolve, meaningful routes are exhausted, or the 24-action
budget is reached. The four active policy stages default to an 8,192-token output
budget and low Gemini thinking. Qwen policy runs can enable visible thinking for
`unified_react`, `unified_reflection`, `unified_discrepancy_decision`, and
`unified_judgment`; the exported ReAct target remains `<think>` plus one native
tool call. Interactions failures that exhaust the
bounded retry window remain hard failures, and the error trace retains completed
calls and retry diagnostics. Evaluation also rejects queries that
explicitly target policy-excluded fact-check domains before Serper.

Run a foreground no-mock canary first. It validates provider configuration, launches
the real evaluator, requires successful search/visit/visual tool classes, and runs the
strict trace audit:

```bash
cd /gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01
run_id="runtime-canary-$(date -u +%Y%m%dT%H%M%SZ)"
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python scripts/run_real_canary.py \
  --benchmark "$IFV_DATA_ROOT/releases/<release-id>/runtime_input/cases.jsonl" \
  --output-dir "$IFV_DATA_ROOT/runs/eval/$run_id" \
  --limit 2
```

Only after this command passes should a larger formal evaluation be launched in the
background with `scripts/server/start_eval_gpu13.sh`.

During any long canary, follow the durable event stream instead of waiting for the
whole-case evaluator summary:

```bash
python scripts/monitor_runtime_events.py \
  --run-dir "$IFV_DATA_ROOT/runs/eval/<run-id>" \
  --case-id <case-id> \
  --follow
```

The monitor reports each model request's stage, estimated input, output cap, actual
provider token counts, tool result, and terminal engineering-error/final snapshot.

For background evaluation launched with `start_eval_gpu13.sh`, poll by run ID only:

```bash
scripts/server/poll_eval_gpu13.sh <run-id>
```

The launcher requires every background output directory to be exactly
`$IFV_DATA_ROOT/runs/eval/<run-id>`, prints the resolved `run_id`, and registers
the log path under `$IFV_DATA_ROOT/runs/_jobs/`. The polling helper resolves the
same canonical data-root path itself and refuses checkout paths, full output
paths, unsafe IDs, non-canonical branches, and non-canonical checkouts. Do not
copy a historical absolute path from an old run command.

The helper also works for runs created before this registration change: it reports
the durable run files and detects a live evaluator from its `--output-dir`
argument, but there may be no registered log path.

`run_real_canary.py` refuses a dirty checkout and rejects any `GIT_COMMIT` value that
does not match the actual HEAD. The child evaluator receives the verified HEAD, so a
run directory name or inherited environment variable cannot falsify manifest
provenance.

Historical baseline acceptance:

```text
local commit:
fd305d712626a1c189433fe37a1e286f30238365

local run:
D:\image-factual-verifier-runs\group-001-v3-canary-20260715-19

gpu-13 commit:
abb7db553cd4d3e8046faed3c43dac3dce67e328

gpu-13 checkout (historical; removed on 2026-08-16):
<removed historical checkout; artifacts remain under the run directory>

gpu-13 run:
/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/
group-001-v3-gpu13-20260715-01
```

Both runs produced the expected `fake` and `real` classifications. The gpu-13 run
passed 165 repository tests, strict two-trace audit, the data-owned classification
scorer, and strict policy-dataset audit.

Historical trajectory-quality acceptance from 2026-07-15:

```text
runtime commit:
c20948d8dc4230c45e4c2f25707e0c52fc31bd80

gpu-13 run:
/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/
group-001-v3-quality-20260715-07

server tests:
175 passed

classification:
2/2 correct

strict trace audit:
2 passed
0 scheduler rejections
0 protocol rejections
0 route-control rejections

policy dataset:
2 eligible episodes
11 examples
0 excluded episodes
0 strict audit errors
```

The accepted fake trajectory used 7 actions and the accepted real trajectory used 3.
Both had complete visual binding, aligned minimal verdict bases, no semantic duplicate
execution, no post-determination actions, and no low-value actions.

Reference-chain recovery was reduced to chain-only metrics and replayed on the same
accepted traces at commit `65a5507`:

```bash
run=/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/group-001-v3-quality-20260715-07
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m src.eval.score_reference_chain --run-dir "$run"
```

The server suite passed `179` tests. Both cases scored `1.0` for fact recovery,
evidence recovery, complete chain recovery, and basis reference precision. Artemis
matched a same-source NASA page version; NOAA matched the official same-capture image
asset. URL, page-snapshot, and SHA identity are not runtime quality metrics. The
optional LLM fallback made zero calls because deterministic matching resolved both
edges.

The older `/gs/home/wza/projects/image-factual-verifier-v2` checkout contained staged
server-side changes and was deliberately left untouched. A fresh GitHub clone was
used instead. The group-001 release was downloaded by gpu-13 through temporary object
storage, verified against the archive hash and all release SHA-256 entries, and the
temporary transfer object was then deleted. No dataset bytes passed through
`47.104.232.153`.

The background launcher prints `pid`, `pid_file`, and `log_file`. Evaluation stdout and stderr
always go to the printed path under `IFV_DATA_ROOT/runs/_logs`, never into the named
run directory. PID files live under `/tmp/image-factual-verifier-v3` and are removed
automatically when their worker exits. The run directory contains durable evaluation
artifacts such as `run_manifest.json`, `predictions.jsonl`, `summary.json`, and JSON
traces. Passing `--output-dir` is mandatory; every other argument is forwarded unchanged
to `python -m src.eval.run_eval` in the `ifv-agent` Conda environment. A non-empty
output directory is rejected so artifacts from separate runs cannot be mixed. Evaluation logs
older than `IFV_LOG_RETENTION_DAYS` are removed when a new run starts; the default is
30 days.

HTML is a derived diagnostic, not a formal evaluation artifact. Generate it only when
needed from one JSON trace or the run's trace directory:

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m src.render_trace_html \
  "$IFV_DATA_ROOT/runs/eval/$run_id/traces" \
  --output-dir "$IFV_DATA_ROOT/runs/eval/$run_id/trace_html"
```

## Troubleshooting

- GitHub/Hugging Face/GDrive connect to `100.10.1.210:47894`: an old shell proxy was
  inherited. Run through `run_gpu13.sh` or source `gpu13_env.sh`.
- Local `8333` accepts TCP but HTTP hangs: verify the SSH process and recreate the
  tunnel with `-F NUL`; then test both the jump-host `127.0.0.1:8333` and local route.
- Jupyter REST login times out intermittently: use the committed client, whose request
  timeout defaults to 30 seconds, and avoid parallel kernel creation.
- A deployment script reports a dirty checkout: stop. Inspect why the server differs
  from Git, but do not continue installing from or editing that checkout.
- OpenMP or native library instability: verify both the wrapper and Conda environment
  report `OMP_NUM_THREADS=1` before running the Agent.

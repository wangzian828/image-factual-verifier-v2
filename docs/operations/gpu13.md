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

OCR is selected explicitly with `OCR_BACKEND`; the default is `easyocr`.
EasyOCR uses one process-shared CPU reader and serializes calls because a
reader instance is not assumed to be thread-safe. Baidu uses the general OCR
API with an in-memory access-token cache. There is no silent backend fallback:
a missing package, missing credential, failed request, malformed result, or
runtime exception is an explicit `ocr_with_position` tool failure.

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

Only after the canary passes should all 20 cases be launched with
`scripts/server/start_eval_gpu13.sh`.

The v3 runtime allows at most 24 real tool actions. Initial target Planning chooses
the evidence target, and ReAct selects the first tool route; there is no fixed initial
reverse-image call. Structured Reflection runs after accepted actions 4, 8, 12, 16,
20, and 24.

Target Planning uploads the original image once as the root of the stored Gemini
Interactions main chain. ReAct, Evidence Decision, Reflection, and Judgment continue
through `previous_interaction_id`; they do not upload the same image again.
Query Concept Extraction, Query Replan, OCR, webpage extraction, and tool-internal
Gemini calls remain separate. If a tool action ends at a deterministic segment
boundary, its pending `function_result` is submitted with the next main-chain
`user_input` step. Trace snapshots store a `runtime_image` reference, not image
base64.
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
Coverage runs after every accepted action and stops immediately when the one core fact
and its required evidence gaps resolve. It also stops as `information_saturated` when
no executable core-gap route remains or when two consecutive action checkpoints make
no qualified core progress. Open-ended ReAct turns use a 16,384-token output budget;
Reflection and Judgment use 8,192. Image Account Planning defaults to high thinking;
its thought tokens are recorded but are not Evidence. Investigation, extraction,
visual-tool, Decision, and Judgment calls remain low thinking. Interactions failures that exhaust the
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

# gpu-13 Operations Guide

This document records the verified deployment path for the `wza` account on
`gpu-13`. It intentionally contains no Jupyter password, API key, private key, or
other credential.

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

The server's `.bashrc` was observed to contain the obsolete proxy port `47894`.
Do not rely on it for this project. The committed gpu-13 wrappers override it with:

```bash
export http_proxy=http://100.10.1.210:47899
export https_proxy=http://100.10.1.210:47899
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export OMP_NUM_THREADS=1
export IFV_DATA_ROOT=/gsdata/home/wza/image-factual-verifier-v2-data
```

The following were verified through that proxy:

- GitHub HTTPS and repository `git ls-remote`;
- Hugging Face HTTPS;
- Google Drive HTTPS.

Use `scripts/server/run_gpu13.sh` for project commands. It always sources the
required proxy and threading environment and fails if the OMP value is not `1`.

## Server Data Root

All datasets and generated runtime artifacts live on the gpu-13 data filesystem:

```text
/gsdata/home/wza/image-factual-verifier-v2-data/
  datasets/       downloaded archives and extracted datasets
  artifacts/      web pages, images, RIS/SERP snapshots, and source material
  benchmarks/     benchmark manifests and benchmark-owned assets
  cache/          Hugging Face, Torch, EasyOCR, and tool caches
  runs/
    _logs/        background evaluation stdout and stderr logs
    traces/       standalone JSON Agent trajectories
    eval/         named evaluation runs with predictions, summaries, and JSON traces
  generated/      later synthetic training and diagnostic data
```

`/gs/home/wza/gsdata` resolves to `/gsdata/home/wza`; the latter had about 399 TB
available during the deployment audit. `gpu13_env.sh` exports `IFV_DATA_ROOT` and
routes Hugging Face, Torch, EasyOCR, and tool caches into this tree. Runtime and data
scripts also derive their default output paths from `IFV_DATA_ROOT`.

Do not download datasets, write evaluation traces, or generate images inside the Git
checkout. The repository contains code, schemas, documentation, and small reviewed
manifests only.

## First Deployment

Push the desired branch from the local repository first:

```powershell
git push -u origin codex/image-factual-verifier-v3
```

Then execute the following on gpu-13 through Jupyter or an approved terminal. The
clone is downloaded by gpu-13 from GitHub, not copied over the SSH endpoint.

```bash
export http_proxy=http://100.10.1.210:47899
export https_proxy=http://100.10.1.210:47899
export OMP_NUM_THREADS=1

mkdir -p /gs/home/wza/projects
cd /gs/home/wza/projects
git clone --branch codex/image-factual-verifier-v3 \
  https://github.com/wangzian828/image-factual-verifier-v2.git
cd image-factual-verifier-v2
bash scripts/server/bootstrap_gpu13.sh
```

No prior Image Factual Verifier checkout or clearly reusable Agent environment was
found under `/gs/home/wza` during the bounded-depth audit. Existing Conda environments
serve unrelated vision/inference projects. The deployment therefore uses the isolated
`ifv-agent` environment with Python 3.11. The bootstrap script is idempotent and stores
`OMP_NUM_THREADS=1` in that Conda environment as an additional guard. It also
installs the `ifv-agent` Jupyter kernelspec, whose wrapper sources
`gpu13_env.sh` before launching the kernel. This ensures browser notebooks and
REST/WebSocket-launched project commands retain the same runtime environment and
places the `ifv-agent` binary directory first in `PATH`, so shell cells also invoke
the project interpreter rather than the Jupyter server's base Conda Python.

## Update From GitHub

After each local commit and push:

```bash
cd /gs/home/wza/projects/image-factual-verifier-v2
bash scripts/server/update_gpu13_checkout.sh codex/image-factual-verifier-v3
bash scripts/server/bootstrap_gpu13.sh
```

The updater allows only a fast-forward from `origin/<branch>` and refuses a dirty
server worktree. Do not bypass that guard by editing or resetting server files.

## Test And Run

Run focused contract tests without credentials first:

```bash
cd /gs/home/wza/projects/image-factual-verifier-v2
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q
```

These are contract and scripted-state checks. For real Gemini transport, create an
untracked `.env` on gpu-13 using a secure interactive method, then run:

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python scripts/probe_gemini_interactions.py --model gemini-3-flash-preview
```

Keep benchmark datasets and caches under `IFV_DATA_ROOT`, outside the Git checkout.

Dataset acquisition, review, and release finalization now belong to the separate
`image-factual-verifier-data-pipeline` project. Copy or mount only a finalized release
under `IFV_DATA_ROOT`; do not run construction code from this runtime checkout.

The v3 runtime allows at most 24 real tool actions. The initial reverse-image search
counts as action 1; structured Reflection runs after actions 4, 8, 12, 16, 20, and 24.
Coverage can stop earlier when all decisive facts resolve or after two consecutive
low-gain Reflection intervals with no unattempted priority-1 task. Open-ended ReAct
turns use a 16,384-token output budget; Reflection and Judgment use 8,192. All active
Gemini stages require minimal thinking. Interactions failures remain hard failures,
and the error trace retains completed calls. Evaluation also rejects queries that
explicitly target policy-excluded fact-check domains before Serper.

Run a foreground no-mock canary first. It validates provider configuration, launches
the real evaluator, requires successful search/visit/visual tool classes, and runs the
strict trace audit:

```bash
cd /gs/home/wza/projects/image-factual-verifier-v3
run_id="runtime-canary-$(date -u +%Y%m%dT%H%M%SZ)"
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python scripts/run_real_canary.py \
  --benchmark "$IFV_DATA_ROOT/releases/<release-id>/runtime_input/cases.jsonl" \
  --output-dir "$IFV_DATA_ROOT/runs/eval/$run_id" \
  --limit 2
```

Only after this command passes should a larger formal evaluation be launched in the
background with `scripts/server/start_eval_gpu13.sh`.

The committed runtime was accepted both locally and on gpu-13:

```text
local commit:
fd305d712626a1c189433fe37a1e286f30238365

local run:
D:\image-factual-verifier-runs\group-001-v3-canary-20260715-19

gpu-13 commit:
abb7db553cd4d3e8046faed3c43dac3dce67e328

gpu-13 checkout:
/gs/home/wza/projects/image-factual-verifier-v3

gpu-13 run:
/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/
group-001-v3-gpu13-20260715-01
```

Both runs produced the expected `fake` and `real` classifications. The gpu-13 run
passed 165 repository tests, strict two-trace audit, the data-owned classification
scorer, and strict policy-dataset audit.

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

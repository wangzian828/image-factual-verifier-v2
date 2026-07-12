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
password, exposed the `python3` kernelspec, and successfully executed a temporary
kernel on `gpu-13`. The kernel observed Python 3.12.7 and eight NVIDIA A100-SXM4
40 GB GPUs. The temporary kernel was deleted after the smoke test.

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
python scripts/server/jupyter_remote.py --shell "hostname; id -un"
```

The client prompts for the password without echo. For unattended automation, inject
`JUPYTER_REMOTE_PASSWORD` from a secret manager for that process only; do not persist
it in a profile or script.

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
  runs/traces/    standalone JSON and HTML Agent trajectories
  runs/eval/      benchmark predictions, summaries, and per-case traces
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
git push -u origin codex/gemini-interactions-agent
```

Then execute the following on gpu-13 through Jupyter or an approved terminal. The
clone is downloaded by gpu-13 from GitHub, not copied over the SSH endpoint.

```bash
export http_proxy=http://100.10.1.210:47899
export https_proxy=http://100.10.1.210:47899
export OMP_NUM_THREADS=1

mkdir -p /gs/home/wza/projects
cd /gs/home/wza/projects
git clone --branch codex/gemini-interactions-agent \
  https://github.com/wangzian828/image-factual-verifier-v2.git
cd image-factual-verifier-v2
bash scripts/server/bootstrap_gpu13.sh
```

No prior Image Factual Verifier checkout or clearly reusable Agent environment was
found under `/gs/home/wza` during the bounded-depth audit. Existing Conda environments
serve unrelated vision/inference projects. The deployment therefore uses the isolated
`ifv-agent` environment with Python 3.11. The bootstrap script is idempotent and stores
`OMP_NUM_THREADS=1` in that Conda environment as an additional guard.

## Update From GitHub

After each local commit and push:

```bash
cd /gs/home/wza/projects/image-factual-verifier-v2
bash scripts/server/update_gpu13_checkout.sh codex/gemini-interactions-agent
bash scripts/server/bootstrap_gpu13.sh
```

The updater allows only a fast-forward from `origin/<branch>` and refuses a dirty
server worktree. Do not bypass that guard by editing or resetting server files.

## Test And Run

Run focused contract tests without credentials first:

```bash
cd /gs/home/wza/projects/image-factual-verifier-v2
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m pytest -q \
  test_unit.py \
  test_evidence_grounding.py \
  test_failure_contracts.py \
  test_provider_failure_propagation.py \
  test_native_interactions.py \
  test_gemini_interactions_contract.py \
  test_gemini_vlm_interactions.py \
  test_full_native_agent_trace.py \
  test_trace_viewer.py
```

For real Gemini and search tests, create an untracked `.env` on gpu-13 using a secure
interactive method, then run only through the wrapper:

```bash
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python scripts/probe_gemini_interactions.py --model gemini-3-flash-preview
```

Keep benchmark datasets and caches under `IFV_DATA_ROOT`, outside the Git checkout.

Acquire the first real-world candidate pool directly from gpu-13:

```bash
cd /gs/home/wza/projects/image-factual-verifier-v2
scripts/server/run_gpu13.sh conda run --no-capture-output -n ifv-agent \
  python -m scripts.benchmark.acquire_averimatec
```

The command downloads AVerImaTeC into `datasets/averimatec`, extracts the images into
a content-addressed directory, and writes an audit manifest under
`benchmarks/candidates/real_seed_v0/averimatec`. It also writes
`source_access_policy.json` and an evaluator-private `evaluation.jsonl`, both derived
from benchmark provenance. `src.eval.run_eval` automatically derives the policy when
the benchmark-wide sibling policy exists, or accepts that file through
`--source-access-policy`. It never derives an AVerImaTeC policy from only the selected
subset because another benchmark case's fact-check page can leak the same answer. It
refuses AVerImaTeC runs without the full policy. The policy stays outside
the model-visible case and trace. It does not freeze any core case:
every item remains pending until a reviewer confirms that the claim is recoverable
from pixels and has a public evidence path. Person-identity claims are allowed, but
must be investigated through RIS, source captions, reporting, and event context rather
than a dedicated biometric model.

Real evaluation defaults to four verification/replanning iterations, twelve native
Interactions turns per iteration, and a 30-minute per-image timeout. Use
`--max-verification-iterations` and `--max-rounds-verification` only for an explicit
experiment; do not reduce them merely to make a run finish. The adaptive audit can stop
after two consecutive low-information-gain iterations when no P2 service or ReInspect
work remains, and records whether it stopped on coverage, saturation, or the hard cap.

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

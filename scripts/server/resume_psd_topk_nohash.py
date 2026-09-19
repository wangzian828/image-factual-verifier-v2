"""Take over completed teacher shards without hashing large PSD artifacts.

The v76 owner is paused while its four GPU children finish.  This controller
then restores the exact serving replicas, writes one gzip cache directly, and
materializes/attests the bank with stat-only v2 stage bindings.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import urllib.request


ROOT = Path("/volume/ybo/wza")
SERVICE = ROOT / "inference/psd-sft3084-20260916"
ROUND = ROOT / "runs/psd-production-round1-20260917-v1"
FINAL = ROOT / "runs/psd-stopped-tail-finalization-20260919-v1"
OUT = FINAL / "teacher-topk-preservation-dedup-v74"
DEDUP = FINAL / "preservation-dedup-v74/bank"
TARGETS = DEDUP / "targets/targets.jsonl"
TARGET_MANIFEST = DEDUP / "targets/manifest.json"
SNAPSHOT = ROOT / "training-artifacts/psd-contract-audit-20260916-v10/snapshot"
ROLLOUT_GATE = ROUND / "rollout-gate.json"
DEPLOY = ROOT / "training-artifacts/psd-lightweight-recovery-20260920-v89"
CODE = DEPLOY / "code"


def owner_module():
    path = ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py"
    spec = importlib.util.spec_from_file_location("psd_owner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def process_state(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(pid: int) -> bool:
    return process_state(pid) is not None


def wait_for_shards(owner) -> None:
    processes = owner.load(OUT / "score-processes.json")["processes"]
    while True:
        states = {str(row["pid"]): process_state(int(row["pid"])) for row in processes}
        if not any(state not in (None, "Z") for state in states.values()):
            break
        sizes = []
        for index in range(4):
            path = OUT / f"gpu-{index}/teacher_topk_cache.jsonl"
            sizes.append(path.stat().st_size if path.is_file() else 0)
        atomic_json(DEPLOY / "progress.json", {"phase": "waiting_teacher_shards",
            "process_states": states, "bytes": sizes, "large_payload_reads": 0})
        time.sleep(15)
    for index in range(4):
        manifest = owner.load(OUT / f"gpu-{index}/manifest.json")
        if manifest.get("status") != "ready_for_materialization" or manifest["counts"]["remaining"]:
            raise RuntimeError(f"teacher shard {index} did not finish cleanly")


def restore_serving(owner, parent_pid: int) -> None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:19025/health", timeout=5) as response:
            if len(json.loads(response.read()).get("replicas", [])) == 4:
                return
    except Exception:
        pass
    if alive(parent_pid):
        os.kill(parent_pid, signal.SIGKILL)
    backends = [owner.load(SERVICE / f"replica-{index}.json") for index in range(4)]
    runtime = ROOT / "envs/h20-qwen35-vllm-0181"
    environments = []
    for index in range(4):
        env = dict(os.environ)
        env.update(PATH=str(runtime / "bin") + ":" + env["PATH"],
            LD_LIBRARY_PATH=":".join((str(runtime / "lib"),
                str(ROOT / "envs/h20-qwen35-128k/lib"), env.get("LD_LIBRARY_PATH", ""))),
            CUDA_VISIBLE_DEVICES=str(index), PYTHONPATH=str(CODE),
            PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(ROOT / "tmp"),
            HF_HOME=str(ROOT / "cache/huggingface"),
            HUGGINGFACE_HUB_CACHE=str(ROOT / "cache/huggingface/hub"),
            TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1",
            TORCH_HOME=str(ROOT / "cache/torch"), NCCL_CUMEM_ENABLE="0",
            NCCL_CUMEM_HOST_ENABLE="0", VLLM_USE_FLASHINFER_SAMPLER="0")
        environments.append(env)
    guard = owner.load(SERVICE / "guard.json")
    if not alive(int(guard["pid"])):
        raise RuntimeError("serving guard is not alive")
    os.kill(guard["pid"], signal.SIGSTOP)
    launched = []
    try:
        errors = []
        for index in range(4):
            try:
                receipt = owner.spawn(backends[index]["command"], environments[index],
                    DEPLOY / f"restored-backend-{index}.log")
                launched.append(receipt)
                owner.save(SERVICE / f"replica-{index}.json", receipt)
            except Exception as error:
                errors.append({"gpu": index, "error_type": type(error).__name__})
        if errors:
            atomic_json(DEPLOY / "restore-errors.json", errors)
            raise RuntimeError("serving restore failed")
        # Four concurrent cold starts can take longer than four minutes while
        # each worker compiles its first multimodal graph.  Keep the bounded
        # wait generous, but fail immediately if a worker actually exits.
        for _ in range(300):
            if any(not alive(int(receipt["pid"])) for receipt in launched):
                raise RuntimeError("restored serving worker exited during startup")
            try:
                with urllib.request.urlopen("http://127.0.0.1:19025/health", timeout=5) as response:
                    if len(json.loads(response.read())["replicas"]) == 4:
                        return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("restored serving did not become healthy")
    except Exception:
        for receipt in launched:
            try:
                if alive(int(receipt["pid"])):
                    owner.stop(receipt)
            except (AssertionError, FileNotFoundError, ProcessLookupError):
                pass
        raise
    finally:
        if alive(int(guard["pid"])):
            os.kill(guard["pid"], signal.SIGCONT)


def merge_gzip(owner) -> Path:
    destination = OUT / "topk-cache/teacher_topk_cache.jsonl.gz"
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    seen = set()
    with temporary.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1, mtime=0) as target:
            for index in range(4):
                with (OUT / f"gpu-{index}/teacher_topk_cache.jsonl").open("rb") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        target_id = str(json.loads(line).get("target_id") or "")
                        if not target_id or target_id in seen:
                            raise ValueError("teacher cache has a missing or duplicate target_id")
                        seen.add(target_id)
                        target.write(line)
        raw.flush()
        os.fsync(raw.fileno())
    expected = owner.load(TARGET_MANIFEST)["counts"]["targets"]
    if len(seen) != expected:
        raise ValueError("teacher cache coverage is incomplete")
    os.replace(temporary, destination)
    atomic_json(destination.parent / "receipt.json", {"schema_version": "ifv-psd-topk-stat-v1",
        "targets": len(seen), "cache": str(destination), "large_payload_hashing": False})
    return destination


def current_snapshot(owner) -> Path:
    """Bind the unchanged checkpoint metadata to the live four-replica gateway."""
    snapshot = DEPLOY / "snapshot-current"
    profile_path = snapshot / "serving-profile.json"
    checkpoint_path = snapshot / "checkpoint-manifest.json"
    if profile_path.is_file() and checkpoint_path.is_file():
        return snapshot
    snapshot.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SNAPSHOT / "checkpoint-manifest.json", checkpoint_path)
    profile = owner.load(SNAPSHOT / "serving-profile.json")
    profile["base_url"] = "http://127.0.0.1:19025/v1"
    atomic_json(profile_path, profile)
    return snapshot


def materialize(owner, cache: Path) -> str:
    sys.path[:0] = [str(CODE), str(CODE / "training")]
    from ifv_training.psd import materialize_psd_topk_cache
    from ifv_training.psd_datums import build_sparse_topk_package
    from ifv_training.psd_materialization import completed_package
    from ifv_training.psd_preflight import verify_psd_training_input
    from scripts.run_psd_round import attest, load_ready
    from types import SimpleNamespace

    # v86 retained an exact attested resolved bank. Reuse it, but rebuild the
    # trainer JSON once with a stable mixed-media struct schema.
    previous_resolved = OUT / "resolved-targets-stat-v86"
    reuse_previous = all(path.is_file() for path in (
        previous_resolved / "targets.jsonl", previous_resolved / "manifest.json",
        OUT / "datums-stat-v86/datums.jsonl", OUT / "datums-stat-v86/manifest.json"))
    resolved = previous_resolved if reuse_previous else OUT / "resolved-targets-stat-v89"
    datums = OUT / "datums-stat-v89"
    if reuse_previous:
        previous = load_ready(OUT / "attested-stat-v86/ready.json")
        if Path(previous["initialization"]["targets_binding"]["path"]).resolve() != (resolved / "targets.jsonl").resolve():
            raise RuntimeError("retained v86 target binding changed")
    else:
        result = completed_package(output_dir=resolved, input_files=[TARGETS, cache],
            build=lambda destination: materialize_psd_topk_cache(targets_path=TARGETS,
                cache_path=cache, output_dir=destination, topk=20))
        if result["status"] != "ready_for_training":
            raise RuntimeError("teacher cache materialization failed")
    result = completed_package(output_dir=datums, input_files=[resolved / "targets.jsonl"],
        build=lambda destination: build_sparse_topk_package(targets_path=resolved / "targets.jsonl",
            output_dir=destination, topk=20, max_sequence_length=131072, require_both_kinds=True))
    if result["status"] != "ready_for_trainer":
        raise RuntimeError("PSD datum materialization failed")
    gate = verify_psd_training_input(datums_path=datums / "datums.jsonl",
        manifest_path=datums / "manifest.json", expected_topk=20, max_context=131072)
    if not gate["passed"]:
        raise RuntimeError("PSD datum preflight failed")
    ready = attest(SimpleNamespace(output=OUT / "attested-stat-v89", rollout_gate=ROLLOUT_GATE,
        datums=datums / "datums.jsonl", datum_manifest=datums / "manifest.json",
        snapshot=current_snapshot(owner)))
    atomic_json(OUT / "result-stat-v89.json", {"status": "ready_for_training", "topk": 20,
        "targets": owner.load(TARGET_MANIFEST)["counts"]["targets"], "ready": ready["ready"],
        "large_payload_hashing": False, "new_agent_or_provider_calls": False})
    # The attestation binds the resolved targets by stat identity, so that file
    # must remain beside the ready package.  Only redundant scoring shards are
    # removable after the canonical gzip cache is durable.
    for path in [OUT / "target-shards", *(OUT / f"gpu-{index}" for index in range(4))]:
        if path.exists():
            shutil.rmtree(path)
    return ready["ready"]


def execute(parent_pid: int) -> None:
    owner = owner_module()
    try:
        cache = OUT / "topk-cache/teacher_topk_cache.jsonl.gz"
        if not cache.is_file():
            wait_for_shards(owner)
        restore_serving(owner, parent_pid)
        atomic_json(DEPLOY / "state.json", {"phase": "teacher_complete_serving_restored"})
        cache = merge_gzip(owner)
        ready = materialize(owner, cache)
        atomic_json(OUT / "state-stat-v89.json", {"phase": "ready_for_training", "ready": ready,
            "formal_training": False, "large_payload_hashing": False})
        atomic_json(DEPLOY / "state.json", {"phase": "ready_for_training", "ready": ready})
    except Exception as error:
        atomic_json(DEPLOY / "state.json", {"phase": "failed_requires_fix",
            "error_type": type(error).__name__, "message": str(error)[:1000]})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "execute"))
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == "execute":
        execute(args.parent_pid)
        return
    DEPLOY.mkdir(parents=True, exist_ok=True)
    if (DEPLOY / "state.json").exists() or (DEPLOY / "process.json").exists():
        raise FileExistsError("lightweight top-k handoff already launched")
    owner = owner_module()
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "execute",
        "--parent-pid", str(args.parent_pid)]
    receipt = owner.spawn(command, os.environ.copy(), DEPLOY / "run.log")
    owner.save(DEPLOY / "process.json", receipt)
    atomic_json(DEPLOY / "state.json", {"phase": "launched", "pid": receipt["pid"]})
    print(json.dumps({"pid": receipt["pid"], "output": str(DEPLOY)}))


if __name__ == "__main__":
    main()

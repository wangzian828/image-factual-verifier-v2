"""Resume the formal PSD round with bounded, resumable local postprocessing."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/volume/ybo/wza")
RUN = ROOT / "runs/psd-production400x8-20260917-v6"
ROUND = ROOT / "runs/psd-production-round1-20260917-v1"
CONTROL = ROOT / "runs/psd-formal-prepare-controller-20260917-v10"
DEPLOY = ROOT / "training-artifacts/psd-streaming-materialization-20260917-v54"
CODE = DEPLOY / "code"
PREFETCH = RUN / "source-review-prefetch-v2-auto-retry"
SERVICE = ROOT / "inference/psd-sft3084-20260916"
PRIVATE_ENV = ROOT / "private/runtime.env"
INTERRUPTION_RECEIPT = DEPLOY / "controlled-interruptions-v2.json"
EXTERNAL_ENV_KEYS = {
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_WIRE_API",
    "GEMINI_MODEL", "GEMINI_VISION_MODEL", "SERPER_API_KEY",
    "SERPER_KEY_ID", "JINA_API_KEY", "JINA_API_KEYS",
    "BAIDU_OCR_API_KEY", "BAIDU_OCR_SECRET_KEY", "BAIDU_OCR_ACCESS_TOKEN",
    "OCR_BACKEND", "OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET",
    "OSS_ENDPOINT", "OSS_BUCKET_NAME", "OSS_KEY_PREFIX",
    "OSS_USE_SIGNED_URL", "OSS_SIGNED_URL_EXPIRY_SECONDS",
    "IMAGE_UPLOAD_PROVIDER", "VISUAL_SEARCH_PROVIDER", "BROWSE_FETCH_PROVIDER",
    "BROWSE_EXTRACT_PROVIDER", "BROWSE_EXTRACT_MODEL", "IFV_SERVER_PROXY",
    "NO_PROXY", "PADDLEOCR_API_TOKEN", "TOOL_CACHE_ENABLED",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def external_environment() -> tuple[dict[str, str], dict[str, bool]]:
    from dotenv import dotenv_values
    mode = PRIVATE_ENV.stat().st_mode & 0o777
    if mode & 0o077:
        raise RuntimeError("private runtime environment must not be group/world accessible")
    parsed = {key: str(value) for key, value in dotenv_values(PRIVATE_ENV).items()
              if key in EXTERNAL_ENV_KEYS and value is not None and str(value).strip()}
    present = lambda *keys: any(parsed.get(key, "").strip() for key in keys)
    checks = {
        "gemini": present("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "serper": present("SERPER_API_KEY", "SERPER_KEY_ID"),
        "jina": present("JINA_API_KEY", "JINA_API_KEYS"),
        "baidu_ocr": (present("BAIDU_OCR_ACCESS_TOKEN")
                       or (present("BAIDU_OCR_API_KEY")
                           and present("BAIDU_OCR_SECRET_KEY"))),
        "oss": all(present(key) for key in ("OSS_ACCESS_KEY_ID",
            "OSS_ACCESS_KEY_SECRET", "OSS_ENDPOINT", "OSS_BUCKET_NAME")),
    }
    if not all(checks.values()):
        raise RuntimeError("PSD external credential preflight is incomplete")
    parsed["GEMINI_MAX_INFLIGHT_REQUESTS"] = "4"
    return parsed, checks


def reconcile_controlled_interruptions() -> dict:
    """Close attempts left running by an attested whole-controller restart.

    This preserves every partial attempt and charges it to the normal bounded
    infrastructure budget.  It never inspects judge/model outcomes.
    """
    if not INTERRUPTION_RECEIPT.is_file():
        return {"status": "not_requested", "reconciled": 0}
    receipt = load(INTERRUPTION_RECEIPT)
    if (receipt.get("schema_version") != "ifv-psd-controlled-interruptions-v1"
            or Path(str(receipt.get("round"))).resolve() != ROUND.resolve()):
        raise RuntimeError("controlled interruption receipt is invalid")
    pids = receipt.get("terminated_prepare_pids")
    if (not isinstance(pids, list) or not pids
            or any(type(pid) is not int or pid < 2 for pid in pids)):
        raise RuntimeError("controlled interruption receipt has invalid PIDs")
    if any(Path("/proc", str(pid)).exists() for pid in pids):
        raise RuntimeError("controlled prepare process is still live")
    for process in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = process.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "run_psd_round.py prepare" in command and str(ROUND) in command:
            raise RuntimeError("cannot reconcile while formal prepare is live")

    sys.path[:0] = [str(CODE), str(CODE / "training")]
    from ifv_training.psd_repair_storage import load_bound, save_bound
    receipt_hash = hashlib.sha256(INTERRUPTION_RECEIPT.read_bytes()).hexdigest()
    reconciled = []
    pattern = "search/repairs/*/slate-rounds/*/infrastructure-attempts/retry-state.json"
    for marker in sorted(ROUND.glob(pattern)):
        saved = load(marker)
        identity = saved.get("identity")
        if not isinstance(identity, dict):
            raise RuntimeError("retry ledger identity is invalid")
        state = load_bound(marker, identity=identity)
        attempts = state.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            continue
        latest = attempts[-1]
        if latest.get("status") != "running":
            continue
        if (marker.parent / "result.json").exists():
            raise RuntimeError("running retry ledger already has a committed result")
        latest.update(
            status="infrastructure_failed",
            reason="controlled_prepare_restart",
            interruption_receipt_sha256=receipt_hash,
        )
        save_bound(marker, identity=identity, payload=state)
        reconciled.append(str(marker.relative_to(ROUND)))
    report = {
        "schema_version": "ifv-psd-interruption-reconciliation-v1",
        "receipt": str(INTERRUPTION_RECEIPT),
        "receipt_sha256": receipt_hash,
        "reconciled": len(reconciled),
        "ledgers": reconciled,
        "time": time.time(),
    }
    save(CONTROL / "interruption-reconciliation.json", report)
    return report


def prepare_command() -> list[str]:
    binding = load(RUN / "binding.json")
    return [sys.executable, "-u", str(CODE / "scripts/run_psd_round.py"), "prepare",
        "--run-dir", str(RUN / "episodes"), "--benchmark", binding["benchmark"],
        "--train-cases", binding["train_cases"], "--private-gold", binding["private_gold"],
        "--source-access-policy", binding["source_access_policy"], "--snapshot", binding["snapshot"],
        "--output", str(ROUND), "--round-index", "1", "--attempts", "6",
        "--case-concurrency", "40", "--source-review-concurrency", "4",
        "--postprocess-workers", "16", "--candidate-workers", "16",
        "--reuse-completed-source-reviews",
        "--source-review-prefetch", str(PREFETCH), "--expected-rollouts-per-case", "8",
        "--task-source-selection", "longest_failed", "--repair-mode", "slate",
        "--judge-model", "gemini-3.1-pro-preview", "--teacher-device", "cuda:0", "--defer-topk"]


def current_status() -> str:
    if (ROUND / "ready.json").exists():
        return "ready_for_training"
    progress = ROUND / "search/progress.json"
    if progress.exists():
        return str(load(progress).get("status") or "")
    postprocess = RUN / "episodes/psd-postprocess-progress.json"
    if postprocess.exists():
        return "postprocess_" + str(load(postprocess).get("status") or "")
    return ""


def worker() -> None:
    reconcile_controlled_interruptions()
    retryable = {"", "postprocess_running", "paused_search_requires_resume"}
    for attempt in range(1, 33):
        save(CONTROL / "state.json", {"phase": "formal_prepare", "attempt": attempt,
            "status_before": current_status(), "time": time.time()})
        with (CONTROL / f"prepare-attempt-{attempt}.log").open("xb") as log:
            process = subprocess.run(prepare_command(), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
        status = current_status()
        save(CONTROL / "state.json", {"phase": "formal_prepare_result", "attempt": attempt,
            "returncode": process.returncode, "status": status, "time": time.time()})
        if process.returncode != 0:
            raise RuntimeError("lightweight formal PSD prepare exited nonzero; inspect bound log")
        if status in {"requires_frozen_teacher_topk", "ready_for_training"}:
            save(CONTROL / "state.json", {"phase": status, "attempt": attempt,
                "training_started": False, "time": time.time()})
            return
        if status not in retryable:
            raise RuntimeError(f"formal PSD prepare stopped in non-retryable status {status!r}")
        time.sleep(min(900, 60 * attempt))
    raise RuntimeError("formal PSD prepare exhausted controller retry budget")


def launch() -> None:
    if CONTROL.exists():
        raise RuntimeError("lightweight PSD controller already exists")
    state = load(DEPLOY / "stage-state.json")
    if (state.get("deployment_ready_not_live") is not True
            or state.get("commit") != "a8ee56c"):
        raise RuntimeError("lightweight PSD deployment is not validated")
    old_owners = []
    for process in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = process.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "run_psd_round.py prepare" in command or "handoff_psd_review_resume_perf.py worker" in command:
            old_owners.append(command)
    if old_owners:
        raise RuntimeError("another formal PSD prepare owner is still live")
    CONTROL.mkdir(parents=True, exist_ok=False)
    owner = module("psd_epoch3_owner",
        ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py")
    env = owner.checked(load(SERVICE / "gateway.json"))
    external, checks = external_environment()
    env.update(external)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker"]
    receipt = owner.spawn(command, env, CONTROL / "controller.log")
    save(CONTROL / "process.json", receipt)
    save(CONTROL / "state.json", {"phase": "launched", "external_checks": checks,
        "gemini_max_inflight_requests": 4, "time": time.time()})
    print(json.dumps({"pid": receipt["pid"], "mode": "lightweight_resume"}))


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "worker"))
    args = parser.parse_args()
    {"launch": launch, "worker": worker}[args.mode]()

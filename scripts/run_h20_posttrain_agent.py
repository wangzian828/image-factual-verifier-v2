"""Run fixed post-SFT Agent cases, engineering-only retries, then unified judge.

Does not modify the Agent or select cases using outcomes. Existing successful
episodes and completed judgments are retained. PSD never starts here.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.finish_agent_eval_judge import digest, rows, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "manifest", "gold", "output", "export-record", "runtime-env"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    export = json.loads(args.export_record.read_text())
    if not export["passed"] or export["global_step"] != 982:
        raise ValueError("post-training export is not accepted")
    identity = {"benchmark_sha256": digest(args.benchmark), "manifest_sha256": digest(args.manifest),
        "gold_sha256": digest(args.gold), "export_sha256": digest(args.export_record),
        "concurrency": args.concurrency, "canary": args.canary,
        "judge_model": "gemini-3.7-flash", "thinking_level": "low"}
    binding = output / "inputs.json"
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError("post-training controller inputs changed")
    save(binding, identity)
    from dotenv import dotenv_values
    env = {**os.environ, **{k: v for k, v in dotenv_values(args.runtime_env).items() if v is not None}}
    env.update(QWEN35_LOCAL_BASE_URL="http://127.0.0.1:8901/v1",
        QWEN35_LOCAL_MODEL="ifv-qwen3.5-9b-sft-982", OMP_NUM_THREADS="1",
        PYTHONPATH=str(ROOT), TMPDIR="/volume/ybo/wza/tmp")
    for port in (8902, 8903, 8904):
        deadline = time.monotonic() + 600
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as response:
                    cards = json.load(response)["data"]
                if not any(card["id"] == "ifv-qwen3.5-9b-sft-982" and
                    Path(card["root"]).resolve() == Path(export["model_path"]).resolve() and
                    card.get("max_model_len", 0) >= 131072 for card in cards):
                    raise ValueError("wrong SFT model/128K service")
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("SFT service did not become ready")
                time.sleep(10)
    expected = [row["case_id"] for row in rows(args.benchmark)]
    if args.canary:
        expected = expected[:1]
    selected, attempts = {}, []
    for attempt in range(3):
        pending = [case for case in expected if selected.get(case, {}).get("status") != "success"]
        if not pending:
            break
        directory = output / f"attempt-{attempt}"
        attempts.append(directory)
        if directory.exists() and not (directory / "summary.json").is_file():
            raise RuntimeError("interrupted rollout: preserve traces and explicitly recover missing/error cases")
        if not directory.exists():
            case_list = output / f"attempt-{attempt}-cases.txt"
            case_list.write_text("".join(case + "\n" for case in pending))
            command = [sys.executable, "-m", "src.eval.run_cases", "--benchmark", str(args.benchmark),
                "--profile", "student-qwen3.5-local", "--model", "ifv-qwen3.5-9b-sft-982",
                "--vlm-model", "ifv-qwen3.5-9b-sft-982", "--output-dir", str(directory),
                "--case-list", str(case_list), "--concurrency", str(min(args.concurrency, len(pending))),
                "--base-sampling-seed", "1729", "--timeout", "1800"]
            if attempt:
                command += ["--resume-from", str(attempts[attempt - 1])]
            save(output / "progress.json", {"phase": "agent_rollout", "attempt": attempt,
                "pending": len(pending), "expected": len(expected), "judge_complete": False})
            with (output / f"attempt-{attempt}.log").open("ab") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        for row in rows(directory / "run_results.jsonl"):
            if row["case_id"] not in pending:
                raise ValueError("attempt resampled a successful/unknown case")
            selected[row["case_id"]] = row
    failed = [case for case in expected if selected.get(case, {}).get("status") != "success"]
    if failed:
        save(output / "progress.json", {"phase": "engineering_retry_budget_exhausted", "remaining": len(failed)})
        raise RuntimeError("Agent engineering failures require inspection")
    if args.canary:
        save(output / "progress.json", {"phase": "canary_passed", "completed": len(selected), "judge_complete": False})
        return
    save(output / "progress.json", {"phase": "judging", "completed": len(selected), "judge_complete": False})
    command = [sys.executable, str(ROOT / "scripts/finish_agent_eval_judge.py"),
        "--run-dir", str(attempts[0]), "--benchmark", str(args.benchmark), "--manifest", str(args.manifest),
        "--gold", str(args.gold), "--output", str(output / "judged"), "--expected-count", str(len(expected)),
        "--judge-model", "gemini-3.7-flash", "--thinking-level", "low", "--concurrency", "64"]
    for directory in attempts[1:]:
        command += ["--retry-dir", str(directory)]
    with (output / "judge-controller.log").open("ab") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    progress = json.loads((output / "judged/progress.json").read_text())
    save(output / "progress.json", {"phase": progress["stage"], "completed": progress["completed"],
        "judge_complete": progress["stage"] == "judge_complete", "psd_started": False})


if __name__ == "__main__":
    main()

"""Stage and test immutable PSD review fixes; never launch production here."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path("/volume/ybo/wza")
OUT = ROOT / "training-artifacts/psd-abstention-quote-repair-20260917-v45"
BASE = ROOT / "training-artifacts/psd-literal-json-20260917-v44/code"
LOCAL_COMMIT = "440621f"
FILES = {
    "scripts": ["postprocess_psd_training.py", "review_psd_sources.py"],
    "training/ifv_training": ["psd_candidates.py", "psd_source_review.py"],
    "training/tests": ["test_psd_candidates.py", "test_psd_source_review.py",
                       "test_psd_training_postprocess.py"],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    os.umask(0o077)
    code = OUT / "code"
    if code.exists():
        raise RuntimeError("Do not modify an existing immutable candidate")
    for parent, names in FILES.items():
        for name in names:
            if not (OUT / parent / name).is_file():
                raise FileNotFoundError(OUT / parent / name)
    shutil.copytree(BASE, code, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for parent, names in FILES.items():
        (code / parent).mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(OUT / parent / name, code / parent / name)
    for path in (BASE / "src").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            if digest(path) != digest(code / path.relative_to(BASE)):
                raise ValueError("Frozen Agent changed")
    binding = {str(path.relative_to(code)): digest(path) for path in code.rglob("*")
               if path.is_file() and "__pycache__" not in path.parts}
    (OUT / "code-binding.json").write_text(json.dumps(binding, indent=2), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": f"{code}:{code / 'training'}",
           "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
           "TMPDIR": str(ROOT / "tmp")}
    with (OUT / "tests.log").open("x", encoding="utf-8") as log:
        result = subprocess.call([sys.executable, "-m", "pytest", "-q", "training/tests",
                                  "-k", "psd", "--tb=short"],
                                 cwd=code, env=env, stdout=log, stderr=log)
    status = {"tests_returncode": result, "local_commit": LOCAL_COMMIT,
              "frozen_agent_unchanged": True,
              "deployment_ready_not_live": result == 0,
              "formal_training": False}
    (OUT / "stage-state.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

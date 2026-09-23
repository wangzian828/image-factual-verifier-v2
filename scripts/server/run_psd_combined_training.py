"""Launch the verified two-source PSD bank from frozen SFT3 step3084.

Uses the previously successful DP4 five-epoch trainer and its native restore
gate.  The bank and gate must both be complete before a GPU lease is taken.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path("/volume/ybo/wza")
CODE = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/psd-combined-smallbank-old1000-sft3-20260923-v1/attested-v7"
READY = RUN / "ready.json"
GATE = RUN / "dp4-resume-gate-combined-v1/result.json"
LAUNCH = RUN / "formal-training-launch-v1"
OUTPUT = ROOT / "training-artifacts/psd-combined-formal-training-20260923-v1"
EXPERIMENT = "psd-combined-smallbank-old1000-sft3-dp4-5epoch-20260923-v1"
SNAPSHOT = ROOT / "training-artifacts/psd-lightweight-recovery-20260920-v89/snapshot-current"


def preflight() -> None:
    sys.path[:0] = [str(CODE), str(CODE / "training")]
    from scripts.run_psd_round import load_ready
    ready = load_ready(READY)
    gate = json.loads(GATE.read_text())
    if (ready.get("schema_version") != "ifv-psd-combined-sft3-ready-v1"
            or Path(ready.get("native_resume_result", "")).resolve() != GATE
            or ready.get("target_counts") != {"repair": 1598, "preserve": 1598}
            or ready.get("adapter") is not None
            or Path(ready["model"]).resolve() !=
                (ROOT / "exports/h20-sft-merged4872-3epoch-step3084-20260915/model").resolve()
            or Path(ready["serving_profile"]).parent != SNAPSHOT
            or gate.get("passed") is not True
            or gate.get("adapter_bitwise_equal") is not True
            or gate.get("formal_training") is not False):
        raise ValueError("combined bank/native DP4 recovery gate has not passed")
    if OUTPUT.exists():
        raise FileExistsError("formal training already has an output; inspect owner before relaunch")


def execute() -> None:
    preflight()
    spec = importlib.util.spec_from_file_location("smallbank_owner",
        CODE / "scripts/server/run_psd_smallbank_training.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.CODE = CODE
    module.DEPLOY = CODE.parent
    module.SNAPSHOT = SNAPSHOT
    module.EXPERIMENT = EXPERIMENT
    module.execute(READY, GATE, OUTPUT)


def launch() -> None:
    preflight()
    if LAUNCH.exists():
        raise FileExistsError("combined formal owner already launched; inspect before retry")
    LAUNCH.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("trusted_owner",
        ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py")
    owner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(owner)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "execute"]
    receipt = owner.spawn(command, os.environ.copy(), LAUNCH / "owner.log")
    owner.save(LAUNCH / "process.json", receipt)
    owner.save(LAUNCH / "state.json", {"phase": "formal_training_launched",
        "ready": str(READY), "gate": str(GATE), "experiment": EXPERIMENT})
    print(json.dumps({"pid": receipt["pid"], "output": str(OUTPUT)}))


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "execute"))
    args = parser.parse_args()
    (launch if args.mode == "launch" else execute)()

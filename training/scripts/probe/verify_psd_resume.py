import argparse
import json
from pathlib import Path
from ifv_training.psd_resume import bind_resume

parser = argparse.ArgumentParser(description="Verify same-round PSD optimizer resume binding")
for name in ("output-root", "datum-manifest-path", "initialization-gate-path", "profile-gate-path"):
    parser.add_argument("--" + name, type=Path, required=True)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-grad-norm", type=float, default=1.0)
print(json.dumps(bind_resume(**vars(parser.parse_args()))))

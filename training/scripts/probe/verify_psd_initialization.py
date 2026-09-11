"""Verify student weight initialization before a PSD Swift launch."""
import argparse
import json
from pathlib import Path

from ifv_training.io import write_json
from ifv_training.psd_initialization import verify_initialization

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--datum-manifest", type=Path, required=True)
parser.add_argument("--serving-profile", type=Path, required=True)
parser.add_argument("--checkpoint-manifest", type=Path, required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--adapter")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
result = verify_initialization(datum_manifest_path=args.datum_manifest,
    serving_profile_path=args.serving_profile, checkpoint_manifest_path=args.checkpoint_manifest,
    model_path=args.model, adapter_path=args.adapter)
write_json(args.output, result)
print(json.dumps(result))

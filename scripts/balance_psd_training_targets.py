"""Build an immutable count-balanced PSD target subset before teacher scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from ifv_training.psd_target_balance import balance_psd_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preserve-target-cap", type=int,
                        help="Default: number of repair targets (1:1 target counts)")
    parser.add_argument("--seed", default="psd-target-balance-v1")
    args = parser.parse_args()
    result = balance_psd_targets(source=args.source, output_dir=args.output_dir,
                                 preserve_target_cap=args.preserve_target_cap,
                                 seed=args.seed)
    print(json.dumps(result["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

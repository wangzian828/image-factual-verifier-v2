from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_smoke_dataset_is_multimodal_and_split(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/server/build_smoke_dataset.py"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    row = json.loads((output / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert manifest["synthetic"] is True
    assert manifest["splits"] == {"train": 2, "validation": 1}
    assert Path(row["images"][0]).is_file()
    assert "<image>" in row["messages"][1]["content"]
    assert set(row["messages"][-1]) == {"role", "content"}

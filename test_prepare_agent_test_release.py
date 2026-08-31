import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.prepare_agent_test_release import prepare_agent_test_release
from src.eval.public_release import load_public_release
from src.eval.release_adapter import RUNTIME_CASE_KEYS


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(case_id: str, group: str, split: str = "test") -> dict:
    return {
        "unified_case_id": case_id,
        "unified_image_path": f"images/{case_id}.jpg",
        "split": split,
        "construction_subroute": group,
        "record_id": f"record-{case_id}",
    }


def test_prepare_agent_test_release_balances_and_hides_private_fields(tmp_path: Path):
    dataset = tmp_path / "dataset"
    rows = [
        _row("case-a1", "a"),
        _row("case-a2", "a"),
        _row("case-a3", "a"),
        _row("case-b1", "b"),
        _row("case-b2", "b"),
        _row("case-b3", "b"),
    ]
    for row in rows:
        image = dataset / row["unified_image_path"]
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(row["unified_case_id"].encode("utf-8"))
    manifest = dataset / "test-manifest.jsonl"
    _write_jsonl(manifest, rows)
    gold = dataset / "evaluator_private" / "private-gold.jsonl"
    _write_jsonl(gold, [{"case_id": row["unified_case_id"]} for row in rows])
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("case-a3\ncase-b3\n", encoding="utf-8")
    policy = tmp_path / "source-policy.json"
    policy.write_text('{"excluded_domains":["example.test"]}\n', encoding="utf-8")

    output = tmp_path / "output"
    result = prepare_agent_test_release(
        dataset_root=dataset,
        test_manifest=manifest,
        private_gold_sidecar=gold,
        output_dir=output,
        limit=4,
        excluded_case_lists=[excluded],
        balanced_by=("construction_subroute",),
        selection_seed="test",
        source_access_policy=policy,
    )

    assert result["case_count"] == 4
    assert result["stratum_counts"] == {"a": 2, "b": 2}
    benchmark = output / "runtime-release" / "runtime_input" / "cases.jsonl"
    runtime_rows = [
        json.loads(line) for line in benchmark.read_text(encoding="utf-8").splitlines()
    ]
    assert len(runtime_rows) == 4
    assert all(set(row) == RUNTIME_CASE_KEYS for row in runtime_rows)
    assert {row["case_id"] for row in runtime_rows}.isdisjoint({"case-a3", "case-b3"})
    assert load_public_release(benchmark).release_stage == "development_subset"

    manifest_payload = json.loads(
        (output / "runtime-release" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_payload["training_prohibited"] is True
    assert not (output / "runtime-release" / "evaluator_private" / "private-gold.jsonl").exists()


def test_prepare_agent_test_release_rejects_non_test_rows(tmp_path: Path):
    dataset = tmp_path / "dataset"
    row = _row("case-train", "a", split="train")
    image = dataset / row["unified_image_path"]
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image")
    manifest = dataset / "test-manifest.jsonl"
    _write_jsonl(manifest, [row])
    gold = dataset / "gold.jsonl"
    _write_jsonl(gold, [{"case_id": "case-train"}])

    with pytest.raises(ValueError, match="non-test row"):
        prepare_agent_test_release(
            dataset_root=dataset,
            test_manifest=manifest,
            private_gold_sidecar=gold,
            output_dir=tmp_path / "output",
            limit=1,
        )


def test_prepare_agent_test_release_cli_imports_project_package():
    script = Path(__file__).parent / "scripts" / "prepare_agent_test_release.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "Build a balanced evaluator-only test release" in completed.stdout

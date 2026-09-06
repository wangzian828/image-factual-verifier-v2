from __future__ import annotations

import json
from pathlib import Path

from scripts.build_direct_qa_portable_package import build_package
from scripts.prepare_direct_qa_inputs import prepare_inputs


def test_prepare_direct_qa_inputs_builds_manifest_and_private_gold(tmp_path: Path) -> None:
    package = tmp_path / "test-package"
    shard = package / "test-set-part-001"
    shard.mkdir(parents=True)
    row = {
        "case_id": "case:one",
        "factual_status": "refuted",
        "gold_verdict": "fake",
        "target_claim": "A false claim.",
        "primary_claim": "A false claim.",
        "construction_subroute": "no_prototype_fabrication",
    }
    (shard / "metadata.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    generated = tmp_path / "generated" / "part-001"
    generated.mkdir(parents=True)
    (generated / "case_one.png").write_bytes(b"image")
    (generated / "results-001.jsonl").write_text(
        json.dumps(
            {"case_id": "case:one", "status": "ok", "image": "case_one.png"}
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "inputs"
    summary = prepare_inputs(
        package_root=package,
        generated_root=tmp_path / "generated",
        output_dir=output,
    )

    assert summary["manifest_count"] == 1
    manifest = json.loads((output / "manifest.jsonl").read_text(encoding="utf-8"))
    assert manifest["unified_case_id"] == "case:one"
    assert manifest["factual_status"] == "refuted"
    gold = json.loads((output / "private-gold.jsonl").read_text(encoding="utf-8"))
    assert gold["target_claim"] == "A false claim."
    assert (output / "images" / "case_one.png").is_file()


def test_build_direct_qa_portable_package_copies_runtime_and_shards(tmp_path: Path) -> None:
    test_package = tmp_path / "test-package"
    test_package.mkdir()
    (test_package / "README.md").write_text("test\n", encoding="utf-8")
    (test_package / "SHA256SUMS.txt").write_text("hash\n", encoding="utf-8")
    (test_package / "test-set-part-001.tar.gz").write_bytes(b"tar")

    output = tmp_path / "portable"
    result = build_package(
        source_repo=Path(__file__).resolve().parent,
        test_package=test_package,
        output_dir=output,
    )

    assert result["schema_version"] == "ifv-direct-qa-portable-package-v1"
    assert (output / "runtime" / "scripts" / "run_direct_qa_baseline.py").is_file()
    assert (output / "runtime" / "scripts" / "audit_direct_qa_baseline.py").is_file()
    assert (output / "test-set" / "test-set-part-001.tar.gz").read_bytes() == b"tar"
    assert "--prompt-file" in (output / "run_direct_qa.sh").read_text(encoding="utf-8")
    assert "prompt.txt" in (output / "run_direct_qa.ps1").read_text(encoding="utf-8")
    assert (output / "prompt.txt").read_text(encoding="utf-8").startswith(
        "You are an Image Factual Verifier."
    )

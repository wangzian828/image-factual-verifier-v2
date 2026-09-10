from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from ifv_training.io import sha256_file


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe"
    / "verify_sft_data_contract.py"
)
SPEC = importlib.util.spec_from_file_location("verify_sft_data_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


CONTRACT = {
    "max_length": 131072,
    "truncation_strategy": "raise",
    "max_pixels": 262144,
    "padding_free": True,
    "sequence_parallel_size": 8,
    "loss_scale": "ignore_empty_think",
    "enable_thinking": False,
    "add_non_thinking_prefix": False,
    "image_max_token_num": 1024,
}


def _dataset(root: Path) -> tuple[Path, Path]:
    row = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<think>x</think><answer>y</answer>"},
        ],
        "images": [],
    }
    train = root / "train.jsonl"
    validation = root / "validation.jsonl"
    train.write_text(json.dumps(row) + "\n", encoding="utf-8")
    validation.write_text(json.dumps(row) + "\n", encoding="utf-8")
    artifacts = {}
    for name, path in (("train", train), ("validation", validation)):
        artifacts[name] = {
            "path": path.name,
            "rows": 1,
            "sha256": sha256_file(path),
        }
    (root / "manifest.json").write_text(
        json.dumps({"artifacts": artifacts}),
        encoding="utf-8",
    )
    return train, validation


def _processor_report(
    path: Path,
    *,
    train: Path,
    validation: Path,
    model: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "ifv-ms-swift-agent-processor-verification-v2",
                "passed": True,
                "model": str(model.resolve()),
                "template_contract": CONTRACT,
                "dataset_files": [
                    MODULE._file_record(train),
                    MODULE._file_record(validation),
                ],
            }
        ),
        encoding="utf-8",
    )


def test_raw_data_gate_binds_manifest_processor_and_profile(tmp_path: Path) -> None:
    train, validation = _dataset(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    processor_report = tmp_path / "processor.json"
    _processor_report(
        processor_report,
        train=train,
        validation=validation,
        model=model,
    )

    result = MODULE.verify_sft_data_contract(
        train_jsonl=train,
        validation_jsonl=validation,
        dataset_dir=tmp_path,
        processor_report_path=processor_report,
        model=str(model),
        expected_template_contract=CONTRACT,
    )

    assert result["passed"] is True
    assert result["dataset_audit"]["passed"] is True


def test_raw_data_gate_rejects_data_drift_after_processor_audit(
    tmp_path: Path,
) -> None:
    train, validation = _dataset(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    processor_report = tmp_path / "processor.json"
    _processor_report(
        processor_report,
        train=train,
        validation=validation,
        model=model,
    )
    train.write_text(train.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    result = MODULE.verify_sft_data_contract(
        train_jsonl=train,
        validation_jsonl=validation,
        dataset_dir=tmp_path,
        processor_report_path=processor_report,
        model=str(model),
        expected_template_contract=CONTRACT,
    )

    assert result["passed"] is False
    assert any("train SHA-256" in error for error in result["errors"])
    assert "strict dataset manifest audit did not pass" in result["errors"]

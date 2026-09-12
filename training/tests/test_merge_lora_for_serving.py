from pathlib import Path
import json
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "h20"
sys.path.insert(0, str(SCRIPT_DIR))

from merge_lora_for_serving import _existing_export, _source_identity  # noqa: E402


def test_merged_export_identity_is_content_bound(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    output = tmp_path / "merged"
    base.mkdir()
    adapter.mkdir()
    output.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (output / "model.safetensors").write_bytes(b"merged")
    source = _source_identity(base.resolve(), adapter.resolve())
    import merge_lora_for_serving

    artifact = {
        "bytes": 6,
        "sha256": merge_lora_for_serving.sha256_file(output / "model.safetensors"),
    }
    record = {
        "passed": True,
        "source": source,
        "model_artifacts": {"model.safetensors": artifact},
    }
    (output / "merge-export.json").write_text(json.dumps(record), encoding="utf-8")
    assert _existing_export(output, source) == record
    (output / "model.safetensors").write_bytes(b"changed")
    assert _existing_export(output, source) is None

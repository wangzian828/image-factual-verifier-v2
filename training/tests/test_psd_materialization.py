import json
from pathlib import Path

import pytest

from ifv_training.psd_materialization import completed_package


def test_interrupted_local_stage_rebuilds_without_repeating_prior_stages(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("{}")
    output = tmp_path / "datums"
    calls = []

    def build(directory):
        calls.append(1)
        directory.mkdir()
        (directory / "data.jsonl").write_text("{}\n")
        if len(calls) == 1:
            raise RuntimeError("simulated process interruption")
        (directory / "manifest.json").write_text('{"status":"ready"}')
        return {"status": "ready"}

    with pytest.raises(RuntimeError, match="interruption"):
        completed_package(output_dir=output, input_files=[source], build=build)
    assert completed_package(output_dir=output, input_files=[source], build=build)["status"] == "ready"
    assert completed_package(output_dir=output, input_files=[source], build=build)["status"] == "ready"
    assert len(calls) == 2
    assert len(list((tmp_path / ".datums-stage").glob("interrupted-*"))) == 1
    (output / "data.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="files changed"):
        completed_package(output_dir=output, input_files=[source], build=build)


def test_stage_refuses_unbound_existing_directory(tmp_path):
    source = tmp_path / "source"
    source.write_text("input")
    output = tmp_path / "package"
    output.mkdir()
    (output / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="unbound"):
        completed_package(output_dir=output, input_files=[source], build=lambda _: {})
    assert (output / "manifest.json").read_text() == "{}"


def test_stage_binds_inputs_and_all_media_not_just_manifest(tmp_path):
    source = tmp_path / "source"
    source.write_text("input")
    output = tmp_path / "package"

    def build(directory):
        directory.mkdir()
        (directory / "manifest.json").write_text("{}")
        (directory / "pixels.pt").write_bytes(b"pixels")
        return {"status": "ready"}

    completed_package(output_dir=output, input_files=[source], build=build)
    source.write_text("different input")
    with pytest.raises(ValueError, match="binding changed"):
        completed_package(output_dir=output, input_files=[source], build=build)
    source.write_text("input")
    (output / "pixels.pt").unlink()
    with pytest.raises(ValueError, match="files changed"):
        completed_package(output_dir=output, input_files=[source], build=build)


def test_interruption_retention_is_bounded(tmp_path):
    source = tmp_path / "source"
    source.write_text("input")

    def fail(directory):
        directory.mkdir()
        (directory / "partial").write_text("partial")
        raise RuntimeError("interrupted")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            completed_package(output_dir=tmp_path / "package", input_files=[source], build=fail)
    with pytest.raises(ValueError, match="review storage"):
        completed_package(output_dir=tmp_path / "package", input_files=[source], build=fail)

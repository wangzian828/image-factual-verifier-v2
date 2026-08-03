from __future__ import annotations

import json
from pathlib import Path

import torch

from ifv_training.encode_cache import (
    METRICS_SCHEMA_VERSION,
    CachedEncodeFunction,
    EncodeCacheConfig,
    aggregate_encode_cache_metrics,
)


class _Config:
    def to_dict(self) -> dict[str, object]:
        return {"size": 8, "mode": "fixture"}


class _Processor:
    def to_dict(self) -> dict[str, object]:
        return {"processor": "fixture"}


class _TemplateMeta:
    template_type = "fixture"


class _Template:
    _version = "v-test"
    max_length = 32
    mode = "train"
    processor = _Processor()
    template_meta = _TemplateMeta()
    loss_scale = _Config()

    def __init__(self) -> None:
        self.calls = 0

    def encode(
        self,
        row: dict[str, object],
        *,
        return_length: bool,
    ) -> dict[str, object]:
        self.calls += 1
        image_path = Path(str(row["images"][0]["path"]))  # type: ignore[index]
        value = image_path.read_bytes()[0]
        return {
            "input_ids": [1, 2, value],
            "labels": [-100, 2, value],
            "pixel_values": torch.tensor(
                [[float(value), 2.0]],
                dtype=torch.float32,
            ),
            "lengths": [3] if return_length else [],
        }


def test_content_addressed_encode_cache_preserves_payload_and_invalidates_image(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"fixture"}')
    image = tmp_path / "image.jpg"
    image.write_bytes(b"A-image")
    monkeypatch.setenv("IFV_MODEL_ID", str(model))  # type: ignore[attr-defined]
    monkeypatch.setenv("IFV_MAX_LENGTH", "32")  # type: ignore[attr-defined]
    monkeypatch.setenv("IFV_IMAGE_MAX_TOKEN_NUM", "8")  # type: ignore[attr-defined]
    monkeypatch.setenv(  # type: ignore[attr-defined]
        "IFV_ADD_NON_THINKING_PREFIX",
        "true",
    )

    template = _Template()
    cached = CachedEncodeFunction(
        template.encode,
        EncodeCacheConfig(
            root=tmp_path / "cache",
            metrics_dir=tmp_path / "metrics",
        ),
    )
    row = {
        "messages": [{"role": "user", "content": "<image>"}],
        "images": [{"bytes": None, "path": str(image)}],
    }

    first = cached(row, return_length=True)
    second = cached(row, return_length=True)

    assert template.calls == 1
    assert first["input_ids"] == second["input_ids"]
    assert torch.equal(first["pixel_values"], second["pixel_values"])
    assert first["pixel_values"].dtype == second["pixel_values"].dtype

    image.write_bytes(b"B-image")
    third = cached(row, return_length=True)

    assert template.calls == 2
    assert third["input_ids"] != first["input_ids"]


def test_encode_cache_report_aggregates_rank_metrics(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    rows = [
        {
            "schema_version": METRICS_SCHEMA_VERSION,
            "contract_digest": "a" * 64,
            "requests": 3,
            "hits": 2,
            "misses": 1,
            "writes": 1,
            "write_races": 0,
            "bytes_loaded": 10,
            "bytes_written": 20,
            "load_seconds": 0.1,
            "encode_seconds": 0.2,
            "write_seconds": 0.3,
            "errors": [],
        },
        {
            "schema_version": METRICS_SCHEMA_VERSION,
            "contract_digest": "a" * 64,
            "requests": 2,
            "hits": 2,
            "misses": 0,
            "writes": 0,
            "write_races": 0,
            "bytes_loaded": 8,
            "bytes_written": 0,
            "load_seconds": 0.05,
            "encode_seconds": 0.0,
            "write_seconds": 0.0,
            "errors": [],
        },
    ]
    for index, row in enumerate(rows):
        (metrics / f"rank-{index}.json").write_text(
            json.dumps(row),
            encoding="utf-8",
        )

    report = aggregate_encode_cache_metrics(metrics)

    assert report["passed"] is True
    assert report["totals"]["requests"] == 5
    assert report["totals"]["hits"] == 4
    assert report["totals"]["misses"] == 1
    assert report["hit_rate"] == 0.8

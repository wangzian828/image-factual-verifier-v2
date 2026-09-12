#!/usr/bin/env python3
"""CPU-only forward/backward smoke for the IFV ms-swift PSD plugin."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    from swift.loss.mapping import loss_map
    from swift.template.register import TEMPLATE_MAPPING
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer

    runpy.run_path(str(args.plugin))
    from ifv_training.psd_ms_swift import (
        PSD_MS_SWIFT_LOSS,
        PSD_MS_SWIFT_TEMPLATE,
        sparse_topk_cross_entropy,
    )

    if PSD_MS_SWIFT_LOSS not in loss_map:
        raise RuntimeError("PSD loss was not registered by external plugin")
    if PSD_MS_SWIFT_TEMPLATE not in TEMPLATE_MAPPING:
        raise RuntimeError("PSD template was not registered by external plugin")
    if not getattr(Seq2SeqTrainer, "_ifv_psd_topk_bridge_installed", False):
        raise RuntimeError("PSD Seq2SeqTrainer bridge was not installed")

    template_cls = TEMPLATE_MAPPING[PSD_MS_SWIFT_TEMPLATE].template_cls
    template = template_cls.__new__(template_cls)
    template.padding_free = False
    template.sequence_parallel_size = 1
    template.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0)
    )
    raw_rows = [
        {
            "input_ids": [1, 2, 3],
            "target_tokens": [[0, 0], [3, 4], [5, 6]],
            "weights": [[0.0, 0.0], [0.75, 0.25], [0.75, 0.25]],
            "topk": 2,
        },
        {
            "input_ids": [4, 5],
            "target_tokens": [[0, 0], [6, 7]],
            "weights": [[0.0, 0.0], [0.5, 0.5]],
            "topk": 2,
        },
    ]
    encoded = [template.encode(row) for row in raw_rows]
    from ifv_training.psd_datums import compact_datum
    for raw, expected in zip(raw_rows, encoded, strict=True):
        compact = compact_datum({**raw, "loss_positions": [index for index, weights in enumerate(raw["weights"])
                                                           if sum(weights) > 0]})
        if template.encode(compact) != expected:
            raise RuntimeError("compact PSD target expansion changes the native template input")
    batch = template.data_collator(encoded)

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16, bias=False)
            self.model_info = SimpleNamespace(is_moe_model=False)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

        def _get_name(self) -> str:
            return "IfvPsdToyModel"

    model = ToyModel()

    class FakeTemplate:
        sequence_parallel_size = 1

        @staticmethod
        def compute_sft_loss(
            model: Any,
            inputs: dict[str, Any],
            **_: Any,
        ) -> Any:
            return model(**inputs)

    class FakeAccelerator:
        @staticmethod
        def unwrap_model(model: Any) -> Any:
            return model

    fake_trainer = SimpleNamespace(
        template=FakeTemplate(),
        model=model,
        args=SimpleNamespace(
            use_liger_kernel=False,
            enable_dft_loss=False,
            enable_channel_loss=False,
            router_aux_loss_coef=None,
            # Skip unrelated token-accuracy bookkeeping in the tiny mock.
            tuner_backend="unsloth",
        ),
        accelerator=FakeAccelerator(),
        label_smoother=None,
        custom_metrics={"train": {}, "eval": {}},
    )

    batch["compute_loss_func"] = (
        lambda outputs, labels, **kwargs: sparse_topk_cross_entropy(
            outputs,
            psd_target_tokens=kwargs["psd_target_tokens"],
            psd_weights=kwargs["psd_weights"],
        )
    )
    loss = Seq2SeqTrainer.compute_loss(fake_trainer, model, batch)
    if not torch.isfinite(loss):
        raise RuntimeError("PSD smoke loss is non-finite")
    loss.backward()
    gradient_l1 = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if gradient_l1 <= 0:
        raise RuntimeError("PSD smoke produced no gradient")

    result = {
        "schema_version": "ifv-psd-ms-swift-plugin-smoke-v1",
        "passed": True,
        "loss": float(loss.detach().cpu()),
        "gradient_l1": gradient_l1,
        "registered_loss": PSD_MS_SWIFT_LOSS,
        "registered_template": PSD_MS_SWIFT_TEMPLATE,
        "compact_and_legacy_template_equivalent": True,
        "batch_shapes": {
            name: list(value.shape)
            for name, value in batch.items()
            if hasattr(value, "shape")
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

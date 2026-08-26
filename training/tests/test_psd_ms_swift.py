from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch

from ifv_training.psd_ms_swift import (
    PSD_MS_SWIFT_LOSS,
    PSD_MS_SWIFT_TEMPLATE,
    sparse_topk_cross_entropy,
)


def test_sparse_topk_cross_entropy_has_expected_value_and_gradient() -> None:
    logits = torch.tensor(
        [[[0.0, 1.0, 2.0], [1.0, 0.0, -1.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target_tokens = torch.tensor([[[2, 1], [0, 1]]])
    weights = torch.tensor([[[0.75, 0.25], [0.0, 0.0]]])

    loss = sparse_topk_cross_entropy(
        SimpleNamespace(logits=logits),
        psd_target_tokens=target_tokens,
        psd_weights=weights,
    )
    expected = -(
        0.75 * torch.log_softmax(logits[0, 0], dim=-1)[2]
        + 0.25 * torch.log_softmax(logits[0, 0], dim=-1)[1]
    )
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0, 0]) > 0
    assert torch.count_nonzero(logits.grad[0, 1]) == 0


def test_ms_swift_plugin_contract_constants_are_stable() -> None:
    assert PSD_MS_SWIFT_TEMPLATE == "ifv_psd_topk"
    assert PSD_MS_SWIFT_LOSS == "ifv_psd_topk"


def test_ms_swift_plugin_smoke_exercises_template_collator_and_trainer_bridge() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "probe"
        / "psd_ms_swift_plugin_smoke.py"
    ).read_text(encoding="utf-8")

    assert "runpy.run_path" in source
    assert "template.data_collator" in source
    assert "Seq2SeqTrainer.compute_loss" in source
    assert "loss.backward()" in source

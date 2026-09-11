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


def test_sparse_topk_cross_entropy_preserves_row_weight_scale() -> None:
    logits = torch.tensor([[[0.0, 1.0, 2.0]]], dtype=torch.float32)
    target_tokens = torch.tensor([[[2, 1]]])
    full = torch.tensor([[[0.75, 0.25]]])
    quarter = full * 0.25

    full_loss = sparse_topk_cross_entropy(
        SimpleNamespace(logits=logits),
        psd_target_tokens=target_tokens,
        psd_weights=full,
    )
    quarter_loss = sparse_topk_cross_entropy(
        SimpleNamespace(logits=logits),
        psd_target_tokens=target_tokens,
        psd_weights=quarter,
    )

    assert torch.allclose(quarter_loss, full_loss * 0.25)


def test_sparse_topk_cross_entropy_sums_tokens_then_averages_batch() -> None:
    logits = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [1.0, 0.0, -1.0]],
            [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    target_tokens = torch.tensor(
        [
            [[2, 1], [0, 1]],
            [[0, 1], [2, 1]],
        ]
    )
    weights = torch.tensor(
        [
            [[0.75, 0.25], [0.0, 0.0]],
            [[0.375, 0.125], [0.375, 0.125]],
        ]
    )

    loss = sparse_topk_cross_entropy(
        SimpleNamespace(logits=logits),
        psd_target_tokens=target_tokens,
        psd_weights=weights,
    )
    selected = torch.gather(
        torch.log_softmax(logits, dim=-1),
        -1,
        target_tokens,
    )
    expected = -(selected * weights).sum() / 2
    assert torch.allclose(loss, expected)


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

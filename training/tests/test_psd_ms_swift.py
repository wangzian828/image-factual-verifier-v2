from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch
import pytest

from ifv_training.psd_ms_swift import (
    PSD_MS_SWIFT_LOSS,
    PSD_MS_SWIFT_TEMPLATE,
    data_parallel_sum_loss,
    prepare_psd_logits_to_keep,
    sparse_topk_cross_entropy,
    split_psd_targets_for_sequence_parallel,
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


def test_data_parallel_sum_loss_compensates_averaged_gradients() -> None:
    parameters = [
        torch.tensor(2.0, requires_grad=True),
        torch.tensor(2.0, requires_grad=True),
    ]
    local_losses = [parameters[0] * 3.0, parameters[1] * 5.0]
    local_gradients = []
    for parameter, loss in zip(parameters, local_losses):
        data_parallel_sum_loss(loss, data_parallel_size=2).backward()
        local_gradients.append(parameter.grad.detach().clone())

    averaged_distributed_gradient = torch.stack(local_gradients).mean()
    assert averaged_distributed_gradient.item() == 8.0


def test_psd_logits_to_keep_filters_only_supervised_positions() -> None:
    inputs = {
        "labels": torch.full((1, 5), -100),
        "psd_target_tokens": torch.tensor(
            [[[0, 0], [4, 7], [0, 0], [2, 9], [0, 0]]]
        ),
        "psd_weights": torch.tensor(
            [[[0.0, 0.0], [0.75, 0.25], [0.0, 0.0], [0.6, 0.4], [0.0, 0.0]]]
        ),
    }

    kept = prepare_psd_logits_to_keep(inputs)

    assert kept == 2
    assert torch.equal(
        inputs["logits_to_keep"],
        torch.tensor([False, True, False, True, False]),
    )
    assert inputs["labels"].shape == (1, 2)
    assert torch.equal(
        inputs["psd_target_tokens"],
        torch.tensor([[[4, 7], [2, 9]]]),
    )
    assert torch.equal(
        inputs["psd_weights"],
        torch.tensor([[[0.75, 0.25], [0.6, 0.4]]]),
    )


def test_psd_logits_to_keep_rejects_non_sparse_label_supervision() -> None:
    inputs = {
        "labels": torch.tensor([[-100, 7]]),
        "psd_target_tokens": torch.tensor([[[0, 0], [2, 9]]]),
        "psd_weights": torch.tensor([[[0.0, 0.0], [0.6, 0.4]]]),
    }
    with pytest.raises(ValueError, match="supervision only from sparse targets"):
        prepare_psd_logits_to_keep(inputs)


def test_psd_logits_to_keep_uses_a_common_batched_suffix() -> None:
    inputs = {
        "labels": torch.full((2, 6), -100),
        "psd_target_tokens": torch.tensor(
            [
                [[0, 0], [0, 0], [3, 8], [4, 9], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [2, 7], [5, 6]],
            ]
        ),
        "psd_weights": torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.7, 0.3], [0.6, 0.4], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.8, 0.2], [0.9, 0.1]],
            ]
        ),
    }

    kept = prepare_psd_logits_to_keep(inputs)

    assert kept == 4
    assert inputs["logits_to_keep"] == 4
    assert inputs["labels"].shape == (2, 4)
    assert inputs["psd_target_tokens"].shape == (2, 4, 2)
    assert torch.equal(inputs["psd_target_tokens"][0, 0], torch.tensor([3, 8]))
    assert torch.equal(inputs["psd_target_tokens"][1, -1], torch.tensor([5, 6]))


def test_sparse_topk_cross_entropy_sums_tokens_and_datums_like_tinker() -> None:
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
    expected = -(selected * weights).sum()
    assert torch.allclose(loss, expected)


def test_sparse_topk_cross_entropy_allows_an_empty_local_sp_shard() -> None:
    logits = torch.tensor([[[0.0, 1.0, 2.0]]], requires_grad=True)
    target_tokens = torch.zeros((1, 1, 2), dtype=torch.long)
    weights = torch.zeros((1, 1, 2), dtype=torch.float32)

    loss = sparse_topk_cross_entropy(
        SimpleNamespace(logits=logits),
        psd_target_tokens=target_tokens,
        psd_weights=weights,
        require_active_per_datum=False,
    )

    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_psd_targets_use_ms_swift_sequence_parallel_split_order() -> None:
    target_tokens = torch.arange(24).reshape(1, 6, 4)
    weights = target_tokens.float()

    class FakeSequenceParallel:
        real_position_ids = torch.arange(6).unsqueeze(0)

        def pad_and_split_inputs(self, *args, **kwargs):
            assert kwargs["real_position_ids"] is self.real_position_ids
            values = kwargs["extra_split_values"]
            assert [(pad, dim) for _, pad, dim in values] == [(0, 1), (0.0, 1)]
            extras = [tensor[:, 2:5] for tensor, _, _ in values]
            return None, None, None, None, None, None, extras

    split_targets, split_weights = split_psd_targets_for_sequence_parallel(
        target_tokens,
        weights,
        sequence_parallel_instance=FakeSequenceParallel(),
    )

    assert torch.equal(split_targets, target_tokens[:, 2:5])
    assert torch.equal(split_weights, weights[:, 2:5])


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
    assert "return_length_contract" in source
    assert "Seq2SeqTrainer.compute_loss" in source
    assert "loss.backward()" in source

    distributed_source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "probe"
        / "psd_sequence_parallel_smoke.py"
    ).read_text(encoding="utf-8")
    assert "GatherLoss" in distributed_source
    assert "split_psd_targets_for_sequence_parallel" in distributed_source
    assert "gathers_full_vocab_logits" in distributed_source

    data_parallel_source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "probe"
        / "psd_data_parallel_smoke.py"
    ).read_text(encoding="utf-8")
    assert "DistributedDataParallel" in data_parallel_source
    assert "Seq2SeqTrainer.compute_loss" in data_parallel_source
    assert "global_datum_sum" in data_parallel_source

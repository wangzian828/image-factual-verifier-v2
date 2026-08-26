"""ms-swift 4.4.2 adapter for IFV sparse top-K PSD training.

The adapter is intentionally loaded through ms-swift's ``--external_plugins``.
It registers a pre-tokenized template and a custom loss while preserving the
framework's normal model, accelerator, FSDP, DeepSpeed, checkpoint, and logging
paths.  It does not train standard SFT labels.
"""

from __future__ import annotations

from typing import Any, Mapping


PSD_MS_SWIFT_TEMPLATE = "ifv_psd_topk"
PSD_MS_SWIFT_LOSS = "ifv_psd_topk"


def sparse_topk_cross_entropy(
    outputs: Any,
    *,
    psd_target_tokens: Any,
    psd_weights: Any,
) -> Any:
    """Return truncated teacher top-K cross entropy over PSD action positions."""

    import torch

    logits = outputs["logits"] if isinstance(outputs, Mapping) else outputs.logits
    if logits.ndim != 3:
        raise ValueError("PSD logits must have shape [batch, sequence, vocab]")
    target_tokens = psd_target_tokens.to(
        device=logits.device,
        dtype=torch.long,
        non_blocking=True,
    )
    weights = psd_weights.to(
        device=logits.device,
        dtype=torch.float32,
        non_blocking=True,
    )
    expected = (*logits.shape[:2], target_tokens.shape[-1])
    if target_tokens.ndim != 3 or tuple(target_tokens.shape) != expected:
        raise ValueError(
            "psd_target_tokens must have shape [batch, sequence, topk]"
        )
    if tuple(weights.shape) != tuple(target_tokens.shape):
        raise ValueError("psd_weights shape must match psd_target_tokens")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("psd_weights must be finite and non-negative")
    active = weights > 0
    if not active.any():
        raise ValueError("PSD batch has no active top-k target")
    active_tokens = target_tokens[active]
    if active_tokens.min() < 0 or active_tokens.max() >= logits.shape[-1]:
        raise ValueError("PSD target token ID is outside model vocabulary")

    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    selected = torch.gather(log_probabilities, -1, target_tokens)
    denominator = weights.sum()
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("PSD weight mass must be finite and positive")
    loss = -(selected * weights).sum() / denominator
    if not torch.isfinite(loss):
        raise ValueError("PSD loss is non-finite")
    return loss


def install_ms_swift_psd_plugin() -> None:
    """Register the IFV template/loss and one narrowly scoped trainer bridge."""

    import torch
    from swift.loss.base import BaseLoss
    from swift.loss.mapping import loss_map
    from swift.template import Template, TemplateMeta, register_template
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer

    class IfvPsdTopKLoss(BaseLoss):
        def __call__(
            self,
            outputs: Any,
            labels: Any,
            *,
            psd_target_tokens: Any = None,
            psd_weights: Any = None,
            **_: Any,
        ) -> Any:
            if psd_target_tokens is None or psd_weights is None:
                raise ValueError(
                    "ifv_psd_topk requires psd_target_tokens and psd_weights"
                )
            return sparse_topk_cross_entropy(
                outputs,
                psd_target_tokens=psd_target_tokens,
                psd_weights=psd_weights,
            )

    class IfvPsdTopKTemplate(Template):
        """Pass repository-produced token IDs through without re-tokenization."""

        def encode(
            self,
            inputs: Mapping[str, Any],
            return_template_inputs: bool = False,
            return_length: bool = False,
        ) -> dict[str, Any]:
            if not isinstance(inputs, Mapping):
                raise ValueError("IFV PSD template requires a mapping input")
            input_ids = inputs.get("input_ids")
            target_tokens = inputs.get("target_tokens")
            weights = inputs.get("weights")
            topk = inputs.get("topk")
            if (
                not isinstance(input_ids, list)
                or not input_ids
                or not isinstance(target_tokens, list)
                or not isinstance(weights, list)
            ):
                raise ValueError("IFV PSD datum is missing token arrays")
            if len(target_tokens) != len(input_ids) or len(weights) != len(input_ids):
                raise ValueError("IFV PSD datum sequence dimensions do not match")
            if not isinstance(topk, int) or topk < 1:
                raise ValueError("IFV PSD datum has invalid topk")
            for position, (tokens, values) in enumerate(
                zip(target_tokens, weights, strict=True)
            ):
                if (
                    not isinstance(tokens, list)
                    or len(tokens) != topk
                    or not isinstance(values, list)
                    or len(values) != topk
                ):
                    raise ValueError(
                        f"IFV PSD datum position {position} has invalid top-k shape"
                    )
            encoded = {
                "input_ids": [int(token) for token in input_ids],
                # The custom loss ignores labels. They make Seq2SeqTrainer take
                # its supported custom-loss route and never reach model.forward.
                "labels": [-100] * len(input_ids),
                "psd_target_tokens": [
                    [int(token) for token in tokens]
                    for tokens in target_tokens
                ],
                "psd_weights": [
                    [float(weight) for weight in values] for values in weights
                ],
            }
            if return_length:
                encoded["length"] = len(input_ids)
            if return_template_inputs:
                raise ValueError(
                    "IFV PSD pre-tokenized template has no renderable messages"
                )
            return encoded

        def data_collator(
            self,
            batch: list[dict[str, Any]],
            *,
            padding_to: int | None = None,
        ) -> dict[str, Any]:
            if not batch:
                raise ValueError("IFV PSD data collator received an empty batch")
            if self.padding_free or self.sequence_parallel_size > 1:
                raise ValueError(
                    "IFV PSD top-k training requires padding_free=false and "
                    "sequence_parallel_size=1"
                )
            topk = len(batch[0]["psd_target_tokens"][0])
            sequence_length = max(len(row["input_ids"]) for row in batch)
            if padding_to is not None:
                if padding_to < sequence_length:
                    raise ValueError("padding_to is smaller than an IFV PSD datum")
                sequence_length = padding_to
            for index, row in enumerate(batch):
                if (
                    len(row["input_ids"]) != len(row["psd_target_tokens"])
                    or len(row["input_ids"]) != len(row["psd_weights"])
                    or any(
                        len(tokens) != topk
                        for tokens in row["psd_target_tokens"]
                    )
                    or any(len(values) != topk for values in row["psd_weights"])
                ):
                    raise ValueError(f"IFV PSD datum {index} has incompatible shape")

            batch_size = len(batch)
            input_ids = torch.full(
                (batch_size, sequence_length),
                int(self.tokenizer.pad_token_id),
                dtype=torch.long,
            )
            attention_mask = torch.zeros(
                (batch_size, sequence_length),
                dtype=torch.long,
            )
            labels = torch.full(
                (batch_size, sequence_length),
                -100,
                dtype=torch.long,
            )
            target_tokens = torch.zeros(
                (batch_size, sequence_length, topk),
                dtype=torch.long,
            )
            weights = torch.zeros(
                (batch_size, sequence_length, topk),
                dtype=torch.float32,
            )
            for index, row in enumerate(batch):
                length = len(row["input_ids"])
                input_ids[index, :length] = torch.tensor(
                    row["input_ids"],
                    dtype=torch.long,
                )
                attention_mask[index, :length] = 1
                target_tokens[index, :length] = torch.tensor(
                    row["psd_target_tokens"],
                    dtype=torch.long,
                )
                weights[index, :length] = torch.tensor(
                    row["psd_weights"],
                    dtype=torch.float32,
                )
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "psd_target_tokens": target_tokens,
                "psd_weights": weights,
            }

    loss_map[PSD_MS_SWIFT_LOSS] = IfvPsdTopKLoss
    register_template(
        TemplateMeta(
            PSD_MS_SWIFT_TEMPLATE,
            prefix=[],
            prompt=["{{QUERY}}"],
            chat_sep=None,
            template_cls=IfvPsdTopKTemplate,
        ),
        exist_ok=True,
    )

    if getattr(Seq2SeqTrainer, "_ifv_psd_topk_bridge_installed", False):
        return
    original_compute_loss = Seq2SeqTrainer.compute_loss

    def compute_loss_with_psd_topk(
        self: Any,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> Any:
        target_tokens = inputs.pop("psd_target_tokens", None)
        weights = inputs.pop("psd_weights", None)
        if target_tokens is None and weights is None:
            return original_compute_loss(
                self,
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        if target_tokens is None or weights is None:
            raise ValueError(
                "IFV PSD batch must contain both psd_target_tokens and psd_weights"
            )
        base_loss_func = inputs.get("compute_loss_func")
        if base_loss_func is None:
            raise ValueError(
                "IFV PSD batch requires --loss_type ifv_psd_topk"
            )

        def bound_loss(
            outputs: Any,
            labels: Any,
            **kwargs: Any,
        ) -> Any:
            return base_loss_func(
                outputs,
                labels,
                psd_target_tokens=target_tokens,
                psd_weights=weights,
                **kwargs,
            )

        inputs["compute_loss_func"] = bound_loss
        return original_compute_loss(
            self,
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    Seq2SeqTrainer.compute_loss = compute_loss_with_psd_topk
    Seq2SeqTrainer._ifv_psd_topk_bridge_installed = True

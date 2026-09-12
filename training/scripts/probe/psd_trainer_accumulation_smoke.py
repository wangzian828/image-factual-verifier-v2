"""CPU acceptance: real HF Trainer loop + installed Swift PSD compute_loss.

Compare one batched update with accumulated microbatches, including a short
final window. SGD exposes gradient scaling that Adam's first update can hide.
This is a trainer integration test, not a real-model/GPU quality experiment.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
from ifv_training.psd_ms_swift import install_ms_swift_psd_plugin, sparse_topk_cross_entropy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    install_ms_swift_psd_plugin()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.head = torch.nn.Linear(4, 16, bias=False)
            self.model_info = SimpleNamespace(is_moe_model=False)

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    class Template:
        sequence_parallel_size = 1
        padding_free = False

        @staticmethod
        def compute_sft_loss(model, inputs, **kwargs):
            return model(**inputs)

    def loss_function(outputs, labels, **kwargs):
        return sparse_topk_cross_entropy(outputs,
            psd_target_tokens=kwargs["psd_target_tokens"], psd_weights=kwargs["psd_weights"])

    class IntegrationTrainer(Trainer):
        compute_loss = Seq2SeqTrainer.compute_loss
        _get_num_items_in_batch = Seq2SeqTrainer._get_num_items_in_batch

        def _get_train_sampler(self, *args, **kwargs):
            return torch.utils.data.SequentialSampler(self.train_dataset)

        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            inputs["compute_loss_func"] = self.compute_loss_func
            return inputs

    class Gradients(TrainerCallback):
        def __init__(self):
            self.values = []

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            self.values.append(torch.cat([p.grad.flatten().clone() for p in model.parameters()]))

    rows = []
    for i in range(5):
        weights = torch.zeros(6, 2)
        weights[-(i % 3 + 1):] = torch.tensor([0.75, 0.25]) * (i + 1) / 5
        rows.append({"input_ids": torch.tensor([1, 2, 3, 4, 5, i + 6]),
            "labels": torch.full((6,), -100), "psd_weights": weights,
            "psd_target_tokens": torch.tensor([[3, 4]] * 6)})
    collate = lambda batch: {key: torch.stack([row[key] for row in batch]) for key in batch[0]}
    torch.manual_seed(37)
    initial = Model().state_dict()
    results = []
    with tempfile.TemporaryDirectory(prefix="ifv-psd-trainer-") as temporary:
        for batch_size, accumulation in ((2, 1), (1, 2)):
            model = Model()
            model.load_state_dict(copy.deepcopy(initial))
            training_args = TrainingArguments(output_dir=str(Path(temporary) / str(batch_size)),
                use_cpu=True, per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=accumulation, num_train_epochs=1,
                max_grad_norm=0, save_strategy="no", logging_strategy="no",
                report_to=[], remove_unused_columns=False, dataloader_pin_memory=False,
                disable_tqdm=True)
            # These Swift options only control unrelated accuracy/auxiliary loss.
            for key, value in {"use_liger_kernel": False, "enable_dft_loss": False,
                "enable_channel_loss": False, "router_aux_loss_coef": None,
                "tuner_backend": "unsloth", "max_epochs": None}.items():
                setattr(training_args, key, value)
            gradients = Gradients()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            trainer = IntegrationTrainer(model=model, args=training_args, train_dataset=rows,
                data_collator=collate, compute_loss_func=loss_function, callbacks=[gradients],
                optimizers=(optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)))
            trainer.template = Template()
            trainer.custom_metrics = {"train": {}, "eval": {}}
            trainer.model_accepts_loss_kwargs = True
            assert trainer._get_num_items_in_batch([collate(rows[:1])], torch.device("cpu")) is None
            trainer.train()
            results.append((torch.cat([p.detach().flatten() for p in model.parameters()]), gradients.values))
        # Independent full-sum reference, not another call to our custom CE.
        reference = Model()
        reference.load_state_dict(initial)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
        reference_gradients = []
        for start in range(0, len(rows), 2):
            batch = collate(rows[start:start + 2])
            logits = reference(batch["input_ids"]).logits
            selected = logits.log_softmax(-1).gather(-1, batch["psd_target_tokens"])
            loss = -(selected * batch["psd_weights"]).sum()
            loss.backward()
            reference_gradients.append(torch.cat([p.grad.flatten().clone() for p in reference.parameters()]))
            optimizer.step()
            optimizer.zero_grad()
        expected = torch.cat([p.detach().flatten() for p in reference.parameters()])
    errors = [float((values - expected).abs().max()) for values, _ in results]
    gradient_errors = [max(float((a - b).abs().max()) for a, b in zip(grads, reference_gradients, strict=True))
                       for _, grads in results]
    passed = all(error < 2e-6 for error in errors + gradient_errors)
    report = {"schema_version": "ifv-psd-trainer-accumulation-smoke-v1", "passed": passed,
        "optimizer_steps": len(reference_gradients), "datums": 5, "short_final_window": True,
        "backward_reduction": "weighted_sum", "max_parameter_errors": errors,
        "max_gradient_errors": gradient_errors, "uses_actual_trainer_train_loop": True,
        "uses_actual_swift_compute_loss": True, "real_model_or_gpu_acceptance": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("PSD accumulated gradients differ from direct Tinker-sum reference")


if __name__ == "__main__":
    main()

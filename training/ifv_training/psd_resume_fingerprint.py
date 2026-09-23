"""Bounded, aggregate-only diagnostics at the native resume boundary.

The probe never writes raw tokens, pixels, targets, or parameter tensors. It
records small summaries for the first four microbatches of optimizer step 2.
"""

import logging
import os


logger = logging.getLogger(__name__)


def _summary(value):
    import torch

    if not isinstance(value, torch.Tensor):
        return None
    local = value.detach()
    if hasattr(local, "to_local"):
        local = local.to_local()
    flat = local.reshape(-1)
    if not flat.numel():
        return {"shape": list(local.shape), "dtype": str(local.dtype), "size": 0}
    # A bounded strided sample catches gross input/state mismatches without
    # copying or hashing the full tensor (especially image and model tensors).
    positions = torch.linspace(0, flat.numel() - 1, min(16, flat.numel()),
                               device=flat.device).long()
    sampled = flat[positions].float()
    return {"shape": list(local.shape), "dtype": str(local.dtype),
            "size": flat.numel(), "sample_sum": float(sampled.sum().item()),
            "sample_sqsum": float(sampled.square().sum().item())}


def install_resume_fingerprint() -> None:
    if os.environ.get("IFV_PSD_RESUME_FINGERPRINT") != "1":
        return
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer

    if getattr(Seq2SeqTrainer, "_ifv_psd_resume_fingerprint_installed", False):
        return
    original = Seq2SeqTrainer.compute_loss
    seen = 0

    def traced(self, model, inputs, *args, **kwargs):
        nonlocal seen
        capture = getattr(self.state, "global_step", None) == 1 and seen < 4
        if capture:
            seen += 1
            fields = {name: _summary(inputs.get(name)) for name in
                      ("input_ids", "pixel_values", "image_grid_thw",
                       "psd_target_tokens", "psd_weights")}
            weights = {}
            for name, parameter in model.named_parameters():
                if "lora_" in name and len(weights) < 4:
                    weights[name] = _summary(parameter)
            logger.warning("IFV PSD resume fingerprint rank=%s micro=%d inputs=%s lora=%s",
                           os.environ.get("RANK", "?"), seen, fields, weights)
        result = original(self, model, inputs, *args, **kwargs)
        if capture:
            loss = result[0] if isinstance(result, tuple) else result
            logger.warning("IFV PSD resume loss rank=%s micro=%d value=%s",
                           os.environ.get("RANK", "?"), seen, float(loss.detach().item()))
        return result

    Seq2SeqTrainer.compute_loss = traced
    Seq2SeqTrainer._ifv_psd_resume_fingerprint_installed = True

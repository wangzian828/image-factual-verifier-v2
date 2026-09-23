"""Bounded, aggregate-only diagnostics at the native resume boundary.

The probe never writes raw tokens, pixels, targets, or parameter tensors. It
records small summaries for the first four microbatches of optimizer step 2.
"""

import logging
import os
import hashlib


logger = logging.getLogger(__name__)


def _sample_positions(size: int, count: int = 16) -> list[int]:
    if size < 1:
        return []
    n = min(count, size)
    return [0] if n == 1 else [i * (size - 1) // (n - 1) for i in range(n)]


def _summary(value, *, exact_small=False, image_moments=False):
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
    # torch.linspace uses fp32 by default; above 2**24 it can round the
    # endpoint *up* to size, turning a diagnostic into a CUDA device assert.
    positions = torch.tensor(_sample_positions(flat.numel()),
                             device=flat.device, dtype=torch.long)
    sampled = flat[positions].float()
    summary = {"shape": list(local.shape), "dtype": str(local.dtype),
            "size": flat.numel(), "sample_sum": float(sampled.sum().item()),
            "sample_sqsum": float(sampled.square().sum().item())}
    if exact_small and flat.numel() <= 200_000:
        # Current in-memory batch tensors only, not model weights, original
        # trajectories, or images. Record a digest, never the raw contents.
        summary["batch_digest"] = hashlib.sha256(
            local.contiguous().cpu().numpy().tobytes()).hexdigest()
    if image_moments:
        # Pixels can be hundreds of MB. Avoid copying or hashing them: a small
        # vector of per-chunk moments detects different processing/order.
        cuts = [i * flat.numel() // 16 for i in range(17)]
        summary["chunk_moments"] = [
            (float(flat[cuts[i]:cuts[i + 1]].sum().item()),
             float(flat[cuts[i]:cuts[i + 1]].square().sum().item()))
            for i in range(16)]
    return summary


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
        capture = getattr(getattr(self, "state", None), "global_step", None) == 1 and seen < 4
        if capture:
            seen += 1
            fields = {name: _summary(inputs.get(name),
                                    exact_small=name != "pixel_values",
                                    image_moments=name == "pixel_values" and seen == 3) for name in
                      ("input_ids", "pixel_values", "image_grid_thw",
                       "psd_target_tokens", "psd_weights")}
            lora = [(name, parameter) for name, parameter in model.named_parameters()
                    if "lora_" in name]
            text_lora = [(name, parameter) for name, parameter in lora
                         if ".layers." in name]
            selected = dict(lora[:4] + text_lora[:4] + lora[-4:])
            weights = {name: _summary(parameter) for name, parameter in selected.items()}
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

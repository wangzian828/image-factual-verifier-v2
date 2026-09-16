"""Observe prefill layer finiteness; synchronize only after compute_logits.

Opt-in diagnostics may affect timing. They never clamp tensors or accept/retry
an invalid response and are not part of the formal serving recipe.
"""
import json
import os
from pathlib import Path
import time
import torch


def install_prefill_observer(worker, output, *, allowed_root=Path('/volume/ybo/wza/inference')):
    output = Path(output).resolve()
    output.relative_to(allowed_root)
    output.mkdir(parents=True, exist_ok=True)
    log = output / f'prefill-{os.getpid()}.jsonl'
    runner = worker.model_runner
    model = runner.get_model()
    original = model.compute_logits
    pending, seen = [], set()
    classes = {'Qwen3_5DecoderLayer', 'Qwen3NextDecoderLayer',
        'Qwen3NextGatedDeltaNet', 'Qwen3_5GatedDeltaNet'}

    def collect(name, value):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and value.numel():
                # Enqueue reductions on the same stream; no .item()/.cpu() here.
                flags = torch.stack((torch.isnan(value).sum(), torch.isinf(value).sum()))
                pending.append((name, list(value.shape), str(value.dtype), flags))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value): collect(f'{name}:{i}', item)

    def observing():
        batch = runner.input_batch
        ids = tuple(batch.req_ids)
        return ids and ids not in seen and len(seen) < 128

    for name, module in model.named_modules():
        if type(module).__name__ not in classes and name != 'visual': continue
        def hook(layer, args, kwargs, result, label=name):
            if observing():
                collect(label, result)
                if 'output' in kwargs: collect(label + ':out_buffer', kwargs['output'])
        module.register_forward_hook(hook, with_kwargs=True)

    def compute_logits(*args, **kwargs):
        logits = original(*args, **kwargs)
        # A later decode can fail even when prefill was finite. Diagnostic only:
        # this introduces a synchronization on each step, never repairs values.
        invalid = bool((~torch.isfinite(logits)).any().item())
        if observing() or invalid:
            collect('raw_model_logits', logits)
            batch = runner.input_batch
            ids = tuple(batch.req_ids); seen.add(ids)
            if pending:
                counts = torch.stack([r[3] for r in pending]).cpu().tolist()
                rows = [{'stage': r[0], 'shape': r[1], 'dtype': r[2], 'nan': c[0], 'inf': c[1]}
                        for r, c in zip(pending, counts)]
                record = {'time': time.time(), 'requests': ids, 'layers': rows,
                    'num_computed_tokens': batch.num_computed_tokens_cpu[:len(ids)].tolist(),
                    'diagnostic_changes_tensor_values': False, 'timing_may_change': True}
                with log.open('a') as f: f.write(json.dumps(record) + '\n')
                last = rows[-1]
                if last['stage'] == 'raw_model_logits' and (last['nan'] or last['inf']):
                    pending.clear()
                    raise RuntimeError('PSD diagnostic captured nonfinite raw model logits; aborted without sampling')
        pending.clear()
        return logits
    model.compute_logits = compute_logits

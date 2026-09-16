"""Opt-in eager-only numerical diagnostic worker; never repairs/clamps tensors.

Hooks synchronize CUDA, so clean runs under observation cannot prove a fix.
Only tensor shapes, finite counts and request IDs are recorded, not text/images.
"""
import json
import os
from pathlib import Path
import time

import torch
from vllm.v1.worker.gpu_worker import Worker


class DiagnosticWorker(Worker):
    def load_model(self):
        super().load_model()
        assert self.model_config.enforce_eager, 'Numerical hooks require isolated eager execution'
        output = Path(os.environ['PSD_NUMERICAL_DIAGNOSTIC_DIR']).resolve()
        output.relative_to(Path('/volume/ybo/wza/inference').resolve())
        output.mkdir(parents=True, exist_ok=True)
        self._diagnostic_file = output / f'worker-{os.getpid()}.jsonl'
        self._diagnostic_seen = set()
        self._diagnostic_count = 0
        self._diagnostic_hooks = []
        model = self.model_runner.get_model()
        classes = {'Qwen3_5DecoderLayer', 'Qwen3NextDecoderLayer',
                   'Qwen3NextGatedDeltaNet', 'Qwen3_5GatedDeltaNet',
                   'Qwen3NextAttention', 'ChunkGatedDeltaRule', 'RMSNormGated'}
        for name, module in model.named_modules():
            if type(module).__name__ not in classes:
                continue
            def hook(layer, inputs, kwargs, result, label=name):
                self._check(label + ':input', (inputs, kwargs))
                self._check(label + ':output', result)
                if 'output' in kwargs:
                    self._check(label + ':out_buffer', kwargs['output'])
            self._diagnostic_hooks.append(module.register_forward_hook(hook, with_kwargs=True))
        assert len(self._diagnostic_hooks) >= 32, 'Expected decoder/GDN diagnostic hooks'
        self._record({'event': 'diagnostic_hooks_installed', 'hooks': len(self._diagnostic_hooks),
                      'eager_only': True, 'changes_tensor_values': False})

    def _record(self, value):
        if self._diagnostic_count >= 256:
            return
        self._diagnostic_count += 1
        with self._diagnostic_file.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'time': time.time(), **value}) + '\n')

    def _check(self, stage, value):
        batch = getattr(self.model_runner, 'input_batch', None)
        requests = tuple(getattr(batch, 'req_ids', ()) or ())
        if not requests or len(self._diagnostic_seen) >= 256:
            return
        def tensors(item):
            if isinstance(item, torch.Tensor):
                yield item
            elif isinstance(item, dict):
                for sub in item.values():
                    yield from tensors(sub)
            elif isinstance(item, (list, tuple)):
                for sub in item:
                    yield from tensors(sub)
        for index, tensor in enumerate(tensors(value)):
            if not tensor.is_floating_point() or not tensor.numel():
                continue
            key = (requests, stage, index)
            if key in self._diagnostic_seen:
                continue
            if not torch.isfinite(tensor).all().item():
                self._diagnostic_seen.add(key)
                self._record({'event': 'nonfinite_tensor', 'requests': requests,
                    'stage': stage, 'index': index, 'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype), 'nan': torch.isnan(tensor).sum().item(),
                    'inf': torch.isinf(tensor).sum().item()})

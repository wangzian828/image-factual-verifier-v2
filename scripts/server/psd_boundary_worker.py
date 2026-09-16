"""Read-only probes before/after grammar masking, not per-layer sync hooks.

Finite raw network logits and an all-masked grammar row are distinct failures.
This worker never clamps logits, retries a request, or alters a grammar.
"""
import json
import os
from pathlib import Path
import time

import torch
from vllm.v1.worker.gpu_worker import Worker


class BoundaryWorker(Worker):
    def load_model(self):
        super().load_model()
        import vllm.v1.worker.gpu_model_runner as runner
        self._out=Path(os.environ['PSD_NUMERICAL_DIAGNOSTIC_DIR']).resolve()
        self._out.relative_to(Path('/volume/ybo/wza/inference'))
        self._out.mkdir(parents=True,exist_ok=True)
        self._file=self._out/f'worker-{os.getpid()}.jsonl'
        self._seen=set();self._count=0
        original=runner.apply_grammar_bitmask
        def apply(*args,**kwargs):
            logits=args[3] if len(args)>=4 else kwargs['logits']
            before=self._flags(logits)
            result=original(*args,**kwargs)
            self._inspect('grammar_boundary',before,self._flags(logits))
            return result
        runner.apply_grammar_bitmask=apply
        def pre(module,args,kwargs):
            logits=args[0] if args else kwargs['logits']
            self._inspect('sampler_input',self._flags(logits),None)
        self._hook=self.model_runner.sampler.register_forward_pre_hook(pre,with_kwargs=True)
        self._record({'event':'boundary_hooks_installed','tensor_mutation':False,
            'sync_after_grammar_may_affect_timing':True})

    @staticmethod
    def _flags(logits):
        return torch.stack([torch.isfinite(logits).sum(-1),torch.isnan(logits).sum(-1),
                            torch.isposinf(logits).sum(-1)],dim=-1)

    def _record(self,record):
        if self._count>=4096:return
        self._count+=1
        with self._file.open('a') as f:f.write(json.dumps({'time':time.time(),**record})+'\n')

    def _inspect(self,stage,before,after):
        batch=self.model_runner.input_batch
        requests=list(batch.req_ids)
        if not requests:return
        # Synchronize only after the actual mask application, never before it.
        flags=torch.stack([before,after],dim=0).cpu().tolist() if after is not None else [before.cpu().tolist()]
        bad=any(v[0]==0 or v[1]>0 or v[2]>0 for group in flags for v in group)
        key=(stage,tuple(requests))
        if not bad and key in self._seen:return
        self._seen.add(key)
        self._record({'event':'invalid_boundary' if bad else 'first_boundary',
            'stage':stage,'requests':requests,'flags_order':['finite','nan','positive_infinity'],
            'flags':flags,'num_computed_tokens':batch.num_computed_tokens_cpu[:len(requests)].tolist()})

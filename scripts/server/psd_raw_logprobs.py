"""Expose pre-grammar teacher logprobs without changing constrained sampling.

vLLM 0.18.1 applies grammar masks before Sampler.forward, so its nominal
raw_logprobs are already constrained. PSD needs the frozen model distribution.
This isolated hook copies logits before masking, then replaces only the
returned logprob tensors after the original sampler has selected its token.
"""
from inspect import signature
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

SEMANTICS = 'pre_grammar_unprocessed_v1'


def validate_raw_worker_receipts(receipts, urls, files):
    """Fail closed before labeling a response as pre-grammar probabilities."""
    if not receipts or not files or len(receipts) != len(urls):
        raise ValueError('Raw logprob capture requires all live worker receipts and code hashes')
    for name, expected in files.items():
        path = Path(name).resolve()
        path.relative_to(Path('/volume/ybo/wza'))
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Raw worker source binding changed')
    ports = []
    for name in receipts:
        path = Path(name).resolve()
        path.relative_to(Path('/volume/ybo/wza'))
        receipt = json.loads(path.read_text())
        command = receipt['command']
        actual = Path(f"/proc/{int(receipt['pid'])}/cmdline").read_bytes().rstrip(b'\0').decode().split('\0')
        if actual != command or '--enforce-eager' not in command:
            raise ValueError('Raw teacher worker is no longer the attested live process')
        if command[command.index('--worker-cls') + 1] != 'scripts.server.psd_raw_teacher_worker.RawTeacherWorker':
            raise ValueError('Replica has not installed pre-grammar teacher capture')
        ports.append(int(command[command.index('--port') + 1]))
    if sorted(ports) != sorted(urlparse(url).port for url in urls):
        raise ValueError('Raw teacher receipts do not match all routed replicas')


def install_raw_policy_logprobs(runner_module, sampler):
    if getattr(sampler, '_ifv_raw_policy_installed', False):
        raise ValueError('Raw policy capture already installed')
    original_apply = runner_module.apply_grammar_bitmask
    original_forward = sampler.forward
    apply_signature = signature(original_apply)
    pending = []

    def apply(*args, **kwargs):
        if pending:
            raise RuntimeError('Grammar logits were not consumed by their sampler')
        logits = apply_signature.bind(*args, **kwargs).arguments['logits']
        pending.append(logits.clone())
        try:
            return original_apply(*args, **kwargs)
        except BaseException:
            pending.clear()
            raise

    def forward(logits, sampling_metadata, *args, **kwargs):
        raw = pending.pop() if pending else None
        count = sampling_metadata.max_num_logprobs
        override = kwargs.get('logprobs_mode_override') or (args[1] if len(args) > 1 else None)
        mode = override or sampler.logprobs_mode
        if count is not None and (mode != 'raw_logprobs' or not 0 < count <= 100):
            raise ValueError('PSD raw teacher capture requires 1..100 raw logprobs')
        # No grammar means stock vLLM already computes raw logprobs correctly.
        if raw is not None and raw.shape != logits.shape:
            raise RuntimeError('Grammar and sampler batch shapes differ')
        result = original_forward(logits, sampling_metadata, *args, **kwargs)
        if raw is not None and count is not None:
            logprobs = sampler.compute_logprobs(raw)
            result.logprobs_tensors = sampler.gather_logprobs(
                logprobs, count, token_ids=result.sampled_token_ids.flatten().long())
        return result

    runner_module.apply_grammar_bitmask = apply
    sampler.forward = forward
    sampler._ifv_raw_policy_installed = True
    return {'semantics': SEMANTICS, 'sampling_changes': False}

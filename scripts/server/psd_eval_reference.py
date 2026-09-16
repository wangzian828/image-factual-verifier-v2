"""Reconstruct the successful epoch-3 evaluation settings for isolated A/B.

Only for diagnosis: the old server did not implement the requested thinking
budget, and its stock grammar logprobs are not raw PSD teacher probabilities.
"""
import copy
from pathlib import Path


def stop_if_present(owner, receipt, *, proc_root=Path('/proc')):
    """An already-exited candidate must not prevent restoration of its owner."""
    try:
        raw = (proc_root/str(receipt['pid'])/'cmdline').read_bytes()
    except FileNotFoundError:
        return
    if not raw:
        return  # Reaped/zombie process; no live command to terminate.
    if raw.rstrip(b'\0').decode().split('\0') != receipt['command']:
        raise RuntimeError('Candidate PID was reused; refusing to terminate another process')
    owner.stop(receipt)


def reference_command(previous):
    command = list(previous)
    if command[2] != 'serve' or '--served-model-name' not in command:
        raise ValueError('Expected an owned vLLM serving command')
    for flag in ('--worker-cls', '--logits-processors', '--tool-parser-plugin',
                 '--mm-processor-cache-gb', '--mamba-cache-mode', '--compilation-config'):
        if flag in command:
            index = command.index(flag); del command[index:index+2]
    for flag in ('--enforce-eager', '--no-async-scheduling', '--async-scheduling', '--no-enable-prefix-caching'):
        if flag in command: command.remove(flag)
    command[command.index('--tool-call-parser')+1] = 'qwen3_coder'
    command[command.index('--max-num-seqs')+1] = '8'
    # Other flags match the archived evaluation startup, including model,
    # context131072, multimodal32, bf16, memory.94, batch32768 and GDN Triton.
    return command


def reference_body(original, mode):
    if mode not in ('plain', 'token_ids', 'top20'):
        raise ValueError('Unknown diagnostic capture mode')
    body = copy.deepcopy(original)
    for key in ('vllm_xargs', 'cache_salt', 'return_token_ids', 'return_tokens_as_token_ids', 'logprobs', 'top_logprobs'):
        body.pop(key, None)
    # Reproduce the original API envelope; explicitly document that the stock
    # vLLM version ignored this extra field rather than enforced the budget.
    body['thinking_token_budget'] = 8192
    if mode != 'plain': body['return_token_ids'] = True
    if mode == 'top20':
        body.update(return_tokens_as_token_ids=True, logprobs=True, top_logprobs=20)
    assert body['max_tokens'] == 32768
    return body


def deferred_teacher_command(previous, code):
    """Stock worker plus public plugins; teacher scoring is a separate forward."""
    command = reference_command(previous)
    command[command.index('--tool-call-parser') + 1] = 'ifv_psd_qwen3_single'
    command += ['--tool-parser-plugin', str(Path(code) / 'scripts/server/psd_single_tool_parser.py'),
                '--logits-processors', 'scripts.server.psd_qwen_thinking:PSDThinkingBudget',
                '--mm-processor-cache-gb', '0', '--no-enable-prefix-caching',
                '--mamba-cache-mode', 'none']
    return command

"""Bounded exact-wire required/auto diagnostics; never execute returned actions."""
from __future__ import annotations

import asyncio
from collections import Counter
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/volume/ybo/wza')
SERVICE = ROOT/'inference/psd-sft2056-safety-20260916'
CAPTURE = SERVICE/'wire-diagnostic-v1/wire'
RUN = ROOT/'runs/psd-wire-diagnostic4x2-20260916'
OUT = SERVICE/'exact-wire-tool-choice-v1'
HELPERS = ROOT/'training-artifacts/psd-serving-safety-20260916'
TICKETS = ['1789522929291883598-bc9dabcf3ae7447faa1ab3e32c7f4112',
           '1789523051897689894-c6ce847909674467b7995076eb5c74e6']
CONCURRENT = False
BACKEND_INDEX = 2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def variants(original):
    if (original.get('tool_choice') != 'required' or original.get('max_tokens') != 32768
            or original.get('vllm_xargs', {}).get('ifv_thinking_budget') != 8192
            or original.get('temperature') != .7):
        raise ValueError('Unexpected original wire protocol')
    result = []
    for choice in ['required', 'auto']:
        body = copy.deepcopy(original)
        body['return_token_ids'] = True
        body['tool_choice'] = choice
        result.append((choice, body))
    return result


def check_backend():
    receipt = json.loads((SERVICE/f'replica-{BACKEND_INDEX}.json').read_text())
    command = [s.decode() for s in Path(f'/proc/{receipt["pid"]}/cmdline').read_bytes().split(b'\0') if s]
    assert command == receipt['command']
    assert '--no-enable-prefix-caching' in command
    assert command[command.index('--mamba-cache-mode')+1] == 'none'
    return {'pid':receipt['pid'],'command':command}


def probe_plan(concurrent):
    return [(index, repeat, choice) for repeat in range(2 if concurrent else 1)
            for index in range(2) for choice in ['required','auto']]


def prepare():
    assert not OUT.exists(), 'Never overwrite or repeat a diagnostic directory'
    assert json.loads((RUN/'state.json').read_text())['phase'] == 'completed_requires_raw_review'
    backend = check_backend()
    spec = importlib.util.spec_from_file_location('wire_audit', HELPERS/'audit_psd_wire_capture_v1.py')
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    inputs = []
    for ticket in TICKETS:
        body, request_meta = audit.read_bound(CAPTURE/ticket, 'request')
        response, response_meta = audit.read_bound(CAPTURE/ticket, 'response')
        choice = response['choices'][0]
        assert choice['finish_reason'] == 'length' and not choice['message'].get('tool_calls')
        variants(body)
        inputs.append({'ticket':ticket,'request_sha256':request_meta['sha256'],
                       'response_sha256':response_meta['sha256']})
    OUT.mkdir()
    save(OUT/'binding.json', {'backend':backend,'inputs':inputs,
        'requests':len(probe_plan(CONCURRENT)),'concurrency':8 if CONCURRENT else 1,
        'changes':['return_token_ids=true on every request', 'tool_choice=auto in diagnostic control only'],
        'unchanged':['messages','tools','images','seed','temperature','think8192','out32768'],
        'not_agent_or_training':True,'psd_gate_passed':False})


async def execute():
    import httpx
    from tokenizers import Tokenizer
    binding = json.loads((OUT/'binding.json').read_text())
    assert check_backend() == binding['backend']
    model = ROOT/'exports/h20-sft-merged4872-epoch2-step2056-20260915/model'
    tokenizer = Tokenizer.from_file(str(model/'tokenizer.json'))
    end_think = tokenizer.token_to_id('</think>')
    assert end_think is not None
    results = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(1230), trust_env=False,
                                transport=httpx.AsyncHTTPTransport(retries=0)) as client:
        async def one(index, repeat, variant):
            item = binding['inputs'][index]
            raw = gzip.decompress((CAPTURE/item['ticket']/'request.json.gz').read_bytes())
            assert hashlib.sha256(raw).hexdigest() == item['request_sha256']
            body = dict(variants(json.loads(raw)))[variant]
            label = f'{index}-{variant}-{repeat}'
            save(OUT/(label+'-request.json'),body)
            started = time.monotonic()
            try:
                response = await asyncio.wait_for(client.post(
                    f'http://127.0.0.1:{19002+BACKEND_INDEX}/v1/chat/completions',json=body),1200)
                with (OUT/(label+'-response.json')).open('xb') as stream:
                    stream.write(response.content)
                response.raise_for_status()
                payload = response.json(); choice = payload['choices'][0]
                ids = choice.get('token_ids') or []
                assert ids, 'Token observations required; do not silently accept missing IDs'
                message = choice['message']
                closures = [i for i,t in enumerate(ids) if t==end_think]
                start = closures[0]+1 if closures else 0
                after = ids[start:]
                result = {'label':label,'finish_reason':choice['finish_reason'],
                    'tokens':len(ids),'zero_tokens':ids.count(0),'endthink_indices':closures[:10],
                    'post_think_top_tokens':Counter(after).most_common(8),
                    'post_think_head':tokenizer.decode(after[:128],skip_special_tokens=False),
                    'post_think_tail':tokenizer.decode(after[-128:],skip_special_tokens=False),
                    'reasoning_chars':len(message.get('reasoning_content') or message.get('reasoning') or ''),
                    'content_chars':len(message.get('content') or ''),
                    'tool_calls':len(message.get('tool_calls') or []),'usage':payload.get('usage')}
            except Exception as error:
                result = {'label':label,'error_type':type(error).__name__,
                          'response_saved':(OUT/(label+'-response.json')).exists()}
            result['seconds'] = time.monotonic()-started
            save(OUT/(label+'-summary.json'),result); results.append(result)
            save(OUT/'progress.json',{'completed':len(results),'target':len(probe_plan(CONCURRENT)),
                'results':results,'psd_gate_passed':False})
            print(json.dumps(result,ensure_ascii=False),flush=True)
        if CONCURRENT:
            await asyncio.gather(*(one(*item) for item in probe_plan(True)))
        else:
            for item in probe_plan(False):
                await one(*item)
    save(OUT/'summary.json',{'results':results,'psd_gate_passed':False,'not_agent_or_training':True,
                             'concurrency':8 if CONCURRENT else 1})


if __name__ == '__main__':
    os.umask(0o077)
    if '--concurrent' in sys.argv:
        CONCURRENT = True
        OUT = SERVICE/'exact-wire-tool-choice-concurrent-v1'
    if '--launch' in sys.argv:
        prepare()
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--execute']
        if CONCURRENT:
            command.append('--concurrent')
        with (OUT/'run.log').open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        save(OUT/'process.json',{'pid':child.pid,'command':command,'script_sha256':sha(Path(__file__)),
                                'started_at':time.time()})
        print(json.dumps({'pid':child.pid,'requests':len(probe_plan(CONCURRENT)),'output':str(OUT)}))
    elif '--execute' in sys.argv:
        asyncio.run(execute())
    else:
        raise ValueError('Use --launch or --execute')

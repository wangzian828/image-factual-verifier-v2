"""Isolated BF16 serving A/B using frozen historical multimodal requests.

No tools are executed, no gold is loaded, no formal inference is released.
Historical JSON persistence can alter formatting; this is a shared benchmark
workload, not a claim of exact original-wire/token reconstruction.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path('/volume/ybo/wza')
REPO = ROOT / 'image-factual-verifier-v2'
WORK = ROOT / 'benchmarks/sft2056-serving-ab-20260915'
EXPORT = ROOT / 'exports/h20-sft-merged4872-epoch2-step2056-20260915'
SMOKE = ROOT / 'runs/eval/qwen35-sft3084-3epoch-agent-formal1527-20260915/smoke/traces'
ENGINE = ROOT / 'envs/h20-qwen35-vllm-0181'
ALIAS = 'ifv-sft2056-speed-probe'
PROFILES = [
    {'name':'baseline-s8-b32k','gpu':0,'port':18902,'seqs':8,'batch_tokens':32768,'apc':False},
    {'name':'s16-b32k','gpu':1,'port':18903,'seqs':16,'batch_tokens':32768,'apc':False},
    {'name':'s16-b8k','gpu':2,'port':18904,'seqs':16,'batch_tokens':8192,'apc':False},
    {'name':'apc-s8-b32k','gpu':3,'port':18905,'seqs':8,'batch_tokens':32768,'apc':True},
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    tmp.replace(path)


def prepare():
    sys.path.insert(0,str(REPO))
    from src.orchestrator.runtime_events import reconstruct_archived_request
    from src.orchestrator.llm_backend import APIBackend
    export = json.loads((EXPORT / 'export.json').read_text())
    assert export['passed'] and export['global_step']==2056 and export['epoch']==2
    path = WORK / 'workload.json'
    if path.exists():
        record = json.loads((WORK / 'prepared.json').read_text())
        assert record['workload_sha256']==digest(path) and record['export_sha256']==digest(EXPORT/'export.json')
        return json.loads(path.read_text())
    workload = []
    for trace_path in sorted(SMOKE.glob('*.json')):
        trace = json.loads(trace_path.read_text())
        steps = [s for s in trace['state']['all_steps'] if s.get('stage')=='unified_react'
                 and s.get('action_type')=='tool_call' and s.get('metadata',{}).get('context_request_id')]
        indexes = sorted({0,len(steps)//3,2*len(steps)//3,len(steps)-1})
        for index in indexes:
            step = steps[index]
            request_id = step['metadata']['context_request_id']
            runtime = Path(trace['state']['runtime_store']['runtime_path'])
            runtime.resolve().relative_to(ROOT)
            archived = reconstruct_archived_request(str(runtime),request_id)
            config = archived['generation_config']
            context = json.loads((runtime/'context'/f'{request_id}.json').read_text())
            body = {'model':ALIAS,'messages':archived['input_payload'],
                'tools':APIBackend._openai_tool_schemas(archived['tool_schema']),
                'tool_choice':'auto','parallel_tool_calls':False,
                'max_tokens':context['max_output_tokens'],
                'chat_template_kwargs':{'enable_thinking':config['enable_thinking']}}
            for key in ['temperature','top_p','top_k','min_p','presence_penalty',
                        'repetition_penalty','seed','thinking_token_budget']:
                if key in config:
                    body[key]=config[key]
            assert body['max_tokens']==32768 and body['thinking_token_budget']==8192
            workload.append({'id':f'{len(workload):02d}','source_trace':str(trace_path),
                'source_trace_sha256':digest(trace_path),'request_id':request_id,
                'source_images':context['image_count'],'body':body})
    assert len(workload)==16
    save(path,workload)
    save(WORK/'prepared.json',{'export_sha256':digest(EXPORT/'export.json'),
        'workload_sha256':digest(path),'requests':len(workload),
        'source':'4 frozen epoch-3 smoke histories, 4 decision points each',
        'profiles':PROFILES,'protocol':'Same epoch-2 weights and request payloads across A/B; no Agent tools or formal test run',
        'source_image_counts':[r['source_images'] for r in workload],
        'max_tokens':32768,'thinking_budget':8192,'context_length':131072,
        'note':'Repeated workload tests warm caching; it does not estimate complete Agent cases/hour.'})
    return workload


def serve():
    prepare()
    for profile in PROFILES:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',profile['port']))
        directory = WORK / profile['name']
        if (directory/'server.json').exists():
            raise ValueError('Profile already launched; inspect before restarting')
    for profile in PROFILES:
        directory = WORK/profile['name']
        directory.mkdir(parents=True,exist_ok=True)
        for folder in ['tmp','cache']:
            (directory/folder).mkdir(exist_ok=True)
        short_tmp = ROOT / f'tmp/ab2056-g{profile["gpu"]}'
        short_tmp.mkdir(exist_ok=True)
        env = {**os.environ,'CUDA_VISIBLE_DEVICES':str(profile['gpu']),
            'HF_HOME':str(ROOT/'cache/h20-huggingface'),'XDG_CACHE_HOME':str(directory/'cache'),
            'FLASHINFER_WORKSPACE_DIR':str(directory/'cache/flashinfer'),
            'TMPDIR':str(short_tmp),'VLLM_USE_FLASHINFER_SAMPLER':'0',
            'LD_LIBRARY_PATH':str(ROOT/'envs/h20-qwen35-128k/lib')}
        command = [str(ENGINE/'bin/vllm'),'serve',str(EXPORT/'model'),
            '--host','127.0.0.1','--port',str(profile['port']),'--served-model-name',ALIAS,
            '--dtype','bfloat16','--tensor-parallel-size','1','--max-model-len','131072',
            '--gpu-memory-utilization','0.94','--max-num-seqs',str(profile['seqs']),
            '--max-num-batched-tokens',str(profile['batch_tokens']),
            '--performance-mode','throughput','--gdn-prefill-backend','triton',
            '--reasoning-parser','qwen3','--enable-auto-tool-choice','--tool-call-parser','qwen3_coder',
            '--structured-outputs-config','{"backend":"xgrammar","reasoning_parser":"qwen3","disable_any_whitespace":true}',
            '--limit-mm-per-prompt','{"image":32,"video":0}',
            '--enable-tokenizer-info-endpoint','--max-log-len','4000','--disable-uvicorn-access-log']
        if profile['apc']:
            command += ['--enable-prefix-caching','--mamba-cache-mode','align']
        else:
            command += ['--no-enable-prefix-caching']
        with (directory/'server.log').open('xb') as log:
            process = subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(directory/'server.json',{'pid':process.pid,'profile':profile,'command':command,'time':time.time()})
        print('LAUNCHED',profile['name'],process.pid,flush=True)


def benchmark():
    import requests
    import jsonschema  # Preflight validation dependency before any paid GPU work.
    workload = prepare()

    def profile_run(profile):
        directory = WORK/profile['name']
        base = f'http://127.0.0.1:{profile["port"]}'
        deadline = time.monotonic()+900
        while True:
            try:
                response = requests.get(base+'/v1/models',timeout=5)
                response.raise_for_status()
                model = response.json()['data'][0]
                assert model['id']==ALIAS and model['root']==str(EXPORT/'model')
                break
            except requests.RequestException:
                if time.monotonic()>deadline:
                    save(directory/'benchmark.json',{'state':'server_unavailable'})
                    return
                time.sleep(10)

        def one(item):
            started = time.monotonic()
            result = {'id':item['id'],'input_images':item['source_images']}
            try:
                response = requests.post(base+'/v1/chat/completions',json=item['body'],timeout=900)
                result['http_status']=response.status_code
                if response.status_code!=200:
                    result['error_type']='HTTPError'
                else:
                    data=response.json()
                    result['usage']=data.get('usage')
                    save(directory/f'pass-{index}-responses'/f'{item["id"]}.json',data)
                    message=data['choices'][0]['message']
                    calls=message.get('tool_calls') or []
                    valid=len(calls)==1
                    if valid:
                        function=calls[0]['function']
                        tool=next((t['function'] for t in item['body']['tools'] if t['function']['name']==function['name']),None)
                        valid=tool is not None
                        if valid:
                            arguments=json.loads(function['arguments'])
                            jsonschema.validate(arguments,tool['parameters'])
                    result.update(finish_reason=data['choices'][0].get('finish_reason'),
                        usage=data.get('usage'),valid_tool_call=valid,
                        output_sha256=hashlib.sha256(json.dumps(message,sort_keys=True).encode()).hexdigest())
                    result['success']=valid and result['finish_reason']=='tool_calls'
            except Exception as error:
                result.update(success=False,error_type=type(error).__name__)
            result['seconds']=round(time.monotonic()-started,3)
            return result

        summaries=[]
        # First pass warms each service equally; retain it separately, never
        # hide its errors. Two steady passes quantify repeat variation.
        for index,concurrency in enumerate([8,16,16]):
            target=directory/f'pass-{index}.json'
            if target.exists():
                raise ValueError('Existing benchmark pass; no silent rerun')
            before=requests.get(base+'/metrics',timeout=10).text
            started=time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                results=list(pool.map(one,workload))
            wall=time.monotonic()-started
            after=requests.get(base+'/metrics',timeout=10).text
            save(directory/f'metrics-{index}.json',{'before':before,'after':after})
            values=[r['seconds'] for r in results]
            tokens=sum((r.get('usage') or {}).get('completion_tokens',0) for r in results)
            summary={'pass':index,'concurrency':concurrency,'cold_or_warm':'cold' if index==0 else 'warm',
                'wall_seconds':wall,'requests':len(results),'successful':sum(r.get('success',False) for r in results),
                'mean_seconds':statistics.mean(values),'p95_seconds':sorted(values)[-1],
                'output_tokens':tokens,'output_tokens_per_second':tokens/wall,'results':results}
            save(target,summary)
            summaries.append({k:v for k,v in summary.items() if k!='results'})
            save(directory/'benchmark.json',{'state':'running' if index<2 else 'completed','passes':summaries})
            print('PASS',profile['name'],json.dumps(summaries[-1]),flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(profile_run,PROFILES))
    save(WORK/'finished.json',{'time':time.time(),'formal_inference_started':False,
                             'note':'Review profile output and performance; no automatic promotion.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['prepare','serve','benchmark'],required=True)
    args=parser.parse_args()
    os.umask(0o077)
    WORK.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(WORK/(args.stage+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    {'prepare':prepare,'serve':serve,'benchmark':benchmark}[args.stage]()

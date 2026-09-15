"""Two fixed training cases through the unchanged Agent, not PSD collection."""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import time
import urllib.request

ROOT = Path('/volume/ybo/wza')
CODE = ROOT/'training-artifacts/psd-adjustments-20260915/code'
FROZEN = ROOT/'image-factual-verifier-v2'
PREP = ROOT/'runs/psd-pilot400-preparation-20260915-v1'
SERVICE = ROOT/'inference/psd-sft2056-safety-20260916'
RUN = ROOT/'runs/psd-real-runtime-gate-20260916-v2'
EXPORT = ROOT/'exports/h20-sft-merged4872-epoch2-step2056-20260915/export.json'
# The frozen runtime selects Qwen3.5 generation settings by the model name.
# The isolated gateway pins this public policy name to its verified backend alias.
POLICY_NAME = 'ifv-qwen3.5-9b-sft-2056'
BACKEND_ALIAS = 'ifv-psd-sft2056-safety'
GATEWAY = 'http://127.0.0.1:19001'
DISABLE_PERCEPTION_CACHE = False


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def environment():
    from dotenv import dotenv_values
    env = {**os.environ, **{k: v for k, v in dotenv_values(ROOT/'private/runtime.env').items() if v is not None}}
    present = lambda *keys: any(str(env.get(k, '')).strip() for k in keys)
    checks = {'serper': present('SERPER_API_KEY', 'SERPER_KEY_ID'),
              'jina': present('JINA_API_KEY', 'JINA_API_KEYS'),
              'gemini_tools': present('GEMINI_API_KEY', 'GOOGLE_API_KEY'),
              'baidu_ocr': present('BAIDU_OCR_API_KEY') and present('BAIDU_OCR_SECRET_KEY')}
    assert all(checks.values()), 'External credentials incomplete; values never logged'
    env.update(QWEN35_LOCAL_BASE_URL=GATEWAY+'/v1', QWEN35_LOCAL_MODEL=POLICY_NAME,
        QWEN_UNIFIED_REACT_MAX_OUTPUT_TOKENS='32768', QWEN_UNIFIED_REACT_THINKING_TOKEN_BUDGET='8192',
        QWEN_UNIFIED_JUDGMENT_MAX_OUTPUT_TOKENS='32768', QWEN_UNIFIED_JUDGMENT_THINKING_TOKEN_BUDGET='8192',
        AGENT_LLM_REQUEST_TIMEOUT_SECONDS='1230', AGENT_STAGE_REQUEST_TIMEOUT_SECONDS='1260',
        AGENT_LLM_REQUEST_MAX_RETRIES='0', TOOL_CACHE_ENABLED='0', OMP_NUM_THREADS='1',
        PYTHONPATH=str(CODE)+':'+str(CODE/'training'), TMPDIR=str(ROOT/'tmp'))
    if DISABLE_PERCEPTION_CACHE:
        env['PERCEPTION_CACHE_ENABLED'] = '0'
    return env, checks


def preflight():
    assert digest(EXPORT) == '55dfb77f56cb175573c5e966816c8a0a0c384385190ae5b62795963f681fbd28'
    assert json.loads((SERVICE/'cancellation-probes-v3/summary.json').read_text())['cancellation_gpu_gate_passed']
    with urllib.request.urlopen(GATEWAY+'/health', timeout=5) as response:
        health = json.load(response)
    assert health['prefix_cache_policy'] == 'unique_salt_per_request'
    assert health['timeout_contract'] == {'gateway': 1200, 'model_client': 1230, 'stage': 1260}
    if GATEWAY.endswith(':19012'):
        assert [row['url'] for row in health['replicas']] == ['http://127.0.0.1:19004']
        assert health['replicas'][0]['healthy']
    for port in range(19002, 19006):
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=5) as response:
            model = json.load(response)['data'][0]
        assert model['id'] == BACKEND_ALIAS and model['root'] == str(EXPORT.parent/'model')
        assert model['max_model_len'] == 131072
    hashes = {}
    # Compare normalized source, because the isolated snapshot has CRLF endings.
    # This gate must not silently substitute newer Agent prompts or tool code.
    for source in sorted((CODE/'src').rglob('*.py')):
        relative = source.relative_to(CODE)
        counterpart = FROZEN/relative
        assert counterpart.is_file() and source.read_text() == counterpart.read_text(), str(relative)
        hashes[str(relative)] = digest(source)
    canary = [json.loads(line) for line in (PREP/'training-canary-case-list.jsonl').read_text().splitlines()]
    assert len(canary) == 32 and all(row['split'] == 'train' for row in canary)
    selected = [row['case_id'] for row in canary[:2]]
    payload = json.loads((PREP/'prepared.json').read_text())['payload']
    public = [json.loads(line) for line in Path(payload['benchmark']).read_text().splitlines()]
    assert set(selected) <= {row['case_id'] for row in public}
    return {'case_ids': selected, 'benchmark': payload['benchmark'],
            'source_access_policy': payload['source_access_policy'],
            'source_sha256': hashes, 'source_normalized_equal_to_frozen': True,
            'export_sha256': digest(EXPORT), 'prepared_sha256': digest(PREP/'prepared.json'),
            'policy_model_name': POLICY_NAME, 'pinned_backend_alias': BACKEND_ALIAS,
            'gateway': GATEWAY,
            'perception_cache_disabled': DISABLE_PERCEPTION_CACHE,
            'temperature': 0.7, 'think_budget': 8192, 'output_budget': 32768,
            'concurrency': 2, 'rollouts_per_case': 1, 'base_sampling_seed': 0,
            'purpose': 'runtime protocol diagnostic only; not 400x8 source collection',
            'server_git_invoked': False, 'private_gold_loaded': False}


def execute():
    binding = json.loads((RUN/'binding.json').read_text())
    assert preflight() == binding
    sys.path[:0] = [str(CODE), str(CODE/'training')]
    from src.eval import run_cases
    from scripts.collect_psd_rollouts import PSDWorkflow
    args = ['runtime-gate', '--benchmark', binding['benchmark'],
            '--source-access-policy', binding['source_access_policy'],
            '--profile', 'student-qwen3.5-local',
            '--output-dir', str(RUN/'episodes'), '--concurrency', '2',
            '--rollouts-per-case', '1', '--base-sampling-seed', '0', '--timeout', '3000']
    for case in binding['case_ids']:
        args += ['--case-id', case]
    sys.argv = args
    parsed = run_cases._parse_args()
    run_cases.VerificationWorkflow = PSDWorkflow
    # No Git on the server. Exact source hashes are persisted instead; do not
    # claim an inferred or unrelated commit for this CRLF source snapshot.
    run_cases._git_commit = lambda: ''
    config = run_cases._workflow_config(parsed)
    workflow = PSDWorkflow(config)
    orchestrator = workflow._get_orchestrator(validate_startup=False)
    if DISABLE_PERCEPTION_CACHE:
        assert not orchestrator.tool_cache.enabled
        assert not orchestrator.cacheable_tools
        save(RUN/'effective-cache-config.json', {
            'tool_cache_enabled': False, 'cacheable_tools': [],
            'TOOL_CACHE_ENABLED': os.environ.get('TOOL_CACHE_ENABLED'),
            'PERCEPTION_CACHE_ENABLED': os.environ.get('PERCEPTION_CACHE_ENABLED'),
            'Agent_source_changed': False})
    configs = {stage: orchestrator._stage_generation_config(stage)
               for stage in ['UNIFIED_REACT', 'UNIFIED_JUDGMENT']}
    for value in configs.values():
        assert value['temperature'] == 0.7 and value['thinking_token_budget'] == 8192
    save(RUN/'effective-stage-config.json', configs)
    save(RUN/'state.json', {'phase': 'running', 'time': time.time(), 'psd_collection': False})
    summary = asyncio.run(run_cases._run_cases(parsed))
    save(RUN/'state.json', {'phase': 'completed_requires_trace_audit', 'summary': summary,
                            'time': time.time(), 'psd_collection': False, 'gate_passed': False})


def launch_no_apc_gateway():
    backend = json.loads((SERVICE/'replica-2.json').read_text())
    backend_args = [p.decode() for p in Path(f'/proc/{backend["pid"]}/cmdline').read_bytes().split(b'\0') if p]
    assert backend_args == backend['command']
    assert '--no-enable-prefix-caching' in backend_args
    assert backend_args[backend_args.index('--mamba-cache-mode')+1] == 'none'
    receipt = SERVICE/'gateway-no-apc.json'
    assert not receipt.exists(), 'Inspect existing isolated gateway; do not duplicate it'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 19012))
    original = json.loads((SERVICE/'gateway.json').read_text())
    args = [p.decode() for p in Path(f'/proc/{original["pid"]}/cmdline').read_bytes().split(b'\0') if p]
    assert args == original['command']
    env = dict(p.decode().split('=', 1) for p in Path(f'/proc/{original["pid"]}/environ').read_bytes().split(b'\0') if p)
    env['QWEN_REPLICA_BACKENDS'] = 'http://127.0.0.1:19004'
    args[args.index('--port')+1] = '19012'
    with (SERVICE/'gateway-no-apc.log').open('xb') as log:
        child = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(receipt, {'pid': child.pid, 'command': args, 'started_at': time.time(),
                   'backends': ['http://127.0.0.1:19004']})
    for _ in range(30):
        if child.poll() is not None:
            raise RuntimeError('Dedicated gateway exited; inspect gateway-no-apc.log')
        try:
            with urllib.request.urlopen(GATEWAY+'/health', timeout=1) as response:
                assert json.load(response)['status'] == 'ok'
            return
        except OSError:
            time.sleep(1)
    raise RuntimeError('Dedicated gateway readiness timeout')


def main():
    global RUN, GATEWAY, DISABLE_PERCEPTION_CACHE
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--single-no-apc', action='store_true')
    parser.add_argument('--cache-free', action='store_true',
                        help='Separate two-case diagnostic with both tool/perception caches disabled')
    args = parser.parse_args()
    if args.cache_free:
        assert not args.single_no_apc
        RUN = ROOT/'runs/psd-real-runtime-gate-no-perception-cache-20260916'
        GATEWAY = 'http://127.0.0.1:19012'
        DISABLE_PERCEPTION_CACHE = True
    if args.single_no_apc:
        RUN = ROOT/'runs/psd-real-runtime-gate-no-apc-20260916'
        GATEWAY = 'http://127.0.0.1:19012'
        if args.launch:
            assert not RUN.exists()
            launch_no_apc_gateway()
    if args.execute:
        try:
            execute()
        except BaseException as error:
            save(RUN/'state.json', {'phase': 'failed_requires_inspection', 'type': type(error).__name__,
                                    'time': time.time(), 'gate_passed': False})
            raise
        return
    binding = preflight()
    env, checks = environment()
    if not args.launch:
        print(json.dumps({'preflight': True, 'cases': binding['case_ids'], 'credentials_present': checks}), flush=True)
        return
    RUN.mkdir()
    save(RUN/'binding.json', binding)
    save(RUN/'credential-presence.json', checks)
    with (RUN/'run.log').open('xb') as log:
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--execute']
        if args.single_no_apc:
            command += ['--single-no-apc']
        if args.cache_free:
            command += ['--cache-free']
        child = subprocess.Popen(command,
            cwd=CODE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(RUN/'process.json', {'pid': child.pid, 'time': time.time(), 'script_sha256': digest(Path(__file__))})
    state = json.loads((SERVICE/'state.json').read_text())
    state.update(phase='real_runtime_gate_running', runtime_gate=str(RUN),
                 runtime_gateway=GATEWAY, gpu_verified=False, source_collection_started=False)
    save(SERVICE/'state.json', state)
    print('RUNTIME_GATE_LAUNCHED', child.pid, flush=True)


if __name__ == '__main__':
    main()

"""Audit completed real canary, then update only its idle loopback gateway."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/volume/ybo/wza')
CODE = ROOT/'training-artifacts/psd-serving-safety-20260916'
FROZEN = ROOT/'training-artifacts/psd-adjustments-20260915/code'
SERVICE = ROOT/'inference/psd-sft2056-safety-20260916'
RUN = ROOT/'runs/psd-real-runtime-gate-no-apc-20260916'
OUT = SERVICE/'tokenizer-gateway-validation-v1'
ALIAS = 'ifv-qwen3.5-9b-sft-2056'


def main():
    spec = importlib.util.spec_from_file_location('service_helpers', CODE/'probe_psd_no_apc_backend_v2.py')
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    assert not OUT.exists()
    sys.path[:0] = [str(FROZEN), str(FROZEN/'training')]
    from scripts.audit_real_trace import audit_trace, _json_summary
    from src.orchestrator.source_access import SourceAccessPolicy
    binding = json.loads((RUN/'binding.json').read_text())
    policy = SourceAccessPolicy.load(Path(binding['source_access_policy']))
    traces = sorted((RUN/'episodes/traces').glob('*.json'))
    assert len(traces) == 2
    audit = _json_summary([audit_trace(p, source_access_policy=policy) for p in traces], strict_scheduler=True)
    assert audit['passed'] and audit['warning_count'] == 0
    details = []
    for path in traces:
        trace = json.loads(path.read_text())
        reasons = [s.get('metadata', {}).get('finish_reason') for s in trace['state']['all_steps']]
        assert not any(r in {'length', 'abort', 'error'} for r in reasons)
        details.append({'case_id': trace['case_id'], 'seconds': trace['time_taken'],
                        'tool_calls': trace['total_tool_calls'], 'subcalls': trace['tool_subcalls_by_kind']})
    OUT.mkdir()
    f.save(OUT/'strict-runtime-audit.json', {**audit, 'case_details': details,
                                           'psd_training_gate_passed': False})
    health = f.get('http://127.0.0.1:19012/health')
    assert health['replicas'][0]['inflight'] == 0
    assert [r['url'] for r in health['replicas']] == ['http://127.0.0.1:19004']
    old, command, env = f.process(SERVICE/'gateway-no-apc.json')
    assert command[command.index('--port')+1] == '19012'
    f.save(OUT/'gateway-before.json', old)
    f.stop(old['pid'])
    command[command.index('--app-dir')+1] = str(CODE/'gateway-v3')
    env.update(PYTHONPATH=str(CODE/'gateway-v3'), PSD_PUBLIC_MODEL_ALIAS=ALIAS)
    child = f.spawn(command, env, SERVICE/'gateway-no-apc-v3.log')
    f.save(SERVICE/'gateway-no-apc.json', {'pid': child.pid, 'command': command,
        'started_at': time.time(), 'replaces': old['pid'], 'backends': ['http://127.0.0.1:19004'],
        'public_alias': ALIAS})
    for _ in range(30):
        if child.poll() is not None:
            raise RuntimeError('Tokenizer gateway exited; inspect log')
        try:
            health = f.get('http://127.0.0.1:19012/health')
            assert health['public_model_alias'] == ALIAS
            break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError('Tokenizer gateway readiness timeout')
    import httpx
    from ifv_training.psd_repair_search import validate_live_model
    from ifv_training.psd_slate import tokenizer_request
    from src.orchestrator.runtime_events import reconstruct_archived_request
    with httpx.Client(timeout=60, trust_env=False) as client:
        cards = client.get('http://127.0.0.1:19012/v1/models').json()
        profile = {'profile_id': ALIAS, 'model_path': str(ROOT/'exports/h20-sft-merged4872-epoch2-step2056-20260915/model'),
                   'context_length': 131072}
        live = validate_live_model(profile, cards)
        f.save(OUT/'live-model-check.json', live)
        # A real archived multi-image/tool context, identical payload in both
        # requests apart from the explicit public-to-backend alias mapping.
        archive = ROOT/'runs/psd-real-runtime-gate-20260916-v2/episodes/traces/runtime/route-aware-hrc-stage2-1-622974dc62/204418-8a5879'
        snapshot = reconstruct_archived_request(archive, 'req-000005')
        body = tokenizer_request(snapshot, model=ALIAS)
        f.save(OUT/'request.json', body)
        proxied = client.post('http://127.0.0.1:19012/tokenize', json=body)
        direct = client.post('http://127.0.0.1:19004/tokenize', json={**body, 'model': 'ifv-psd-sft2056-safety'})
        (OUT/'proxy-response.json').write_bytes(proxied.content)
        (OUT/'direct-response.json').write_bytes(direct.content)
        proxied.raise_for_status(); direct.raise_for_status()
        assert proxied.json()['tokens'] == direct.json()['tokens']
        assert proxied.json()['count'] == direct.json()['count'] > 0
        result = {'passed': True, 'token_count': proxied.json()['count'], 'alias_attested': True,
                  'comparison': 'same reconstructed payload via gateway versus direct backend',
                  'exact_live_capture_attested': False, 'generation_requests': 0,
                  'psd_training_gate_passed': False, 'time': time.time()}
        f.save(OUT/'summary.json', result)
    state = json.loads((SERVICE/'state.json').read_text())
    state.update(phase='single_card_runtime_and_tokenizer_checked', runtime_audit=str(OUT/'strict-runtime-audit.json'),
                 tokenizer_validation=str(OUT/'summary.json'), gpu_verified=False,
                 source_collection_started=False, next_phase='real_multisite_psd_canary')
    f.save(SERVICE/'state.json', state)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

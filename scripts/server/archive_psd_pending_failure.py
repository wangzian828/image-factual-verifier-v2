"""Reconstruct failed diagnostic requests from committed started-event artifacts.

Original runtime artifacts stay untouched. Bodies are reconstructed, not claimed
to be byte-identical wire copies; nonce/header identity is deliberately excluded.
"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path('/volume/ybo/wza')
CODE = ROOT / 'training-artifacts/psd-capture-callsite-20260916-v21/code'
RUN = ROOT / 'runs/psd-raw-runtime-gate-20260916-v1'
OUT = ROOT / 'inference/psd-sft3084-20260916/raw-multimage-failure-v1'


def main():
    os.umask(0o077); sys.path[:0] = [str(CODE), str(CODE / 'training')]
    from src.orchestrator.runtime_events import reconstruct_archived_request
    from src.orchestrator.llm_backend import APIBackend
    spec = importlib.util.spec_from_file_location('owner', ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    OUT.mkdir(exist_ok=False)
    class Captured(BaseException): pass
    details = []
    for events in sorted((RUN / 'episodes/traces/runtime').glob('*/*/events.jsonl')):
        rows = [json.loads(l) for l in events.read_text().splitlines() if l.strip()]
        manifest = next(r['payload'] for r in rows if r['event_type'] == 'context_request_started'
                        and r['payload']['request_id'] == 'req-000005')
        dest = OUT / events.parent.parent.name
        (dest / 'context').mkdir(parents=True)
        # Read-only artifact lookup; the new context manifest is outside the run.
        (dest / 'artifacts').symlink_to(events.parent / 'artifacts', target_is_directory=True)
        owner.save(dest / 'context/req-000005.json', manifest)
        archived = reconstruct_archived_request(dest, 'req-000005')
        saved = {}
        class CaptureClient:
            async def post(self, url, *, headers, json):
                saved.update(json); raise Captured()
        backend = APIBackend(provider='qwen_local', model_name=manifest['model'],
            api_key='none', base_url='http://127.0.0.1:19022/v1', max_retries=0, extra_body={})
        backend._get_shared_client = lambda: CaptureClient()
        try:
            asyncio.run(backend.get_response(archived['input_payload'],
                max_tokens=manifest['max_output_tokens'], tools=archived['tool_schema'],
                tool_choice='required', generation_config=archived['generation_config'],
                capture_policy_tokens=True, policy_topk=20))
        except Captured: pass
        assert saved['max_tokens'] == 32768 and saved['thinking_token_budget'] == 8192
        owner.save(dest / 'reconstructed-client-body.json', saved)
        translated = {**saved, 'model': 'ifv-psd-sft3084',
            'vllm_xargs': {'ifv_thinking_budget': saved['thinking_token_budget']},
            'cache_salt': 'isolated-diagnostic-' + uuid.uuid4().hex}
        translated.pop('thinking_token_budget')
        owner.save(dest / 'backend-body.json', translated)
        details.append({'case': events.parent.parent.name, 'request': manifest['request_id'],
            'images': manifest['image_count'], 'body': str(dest / 'backend-body.json'),
            'body_sha256': owner.sha(dest / 'backend-body.json'), 'source_events': str(events),
            'diagnostic_only': True, 'wire_equivalence': 'reconstructed; cache nonce differs'})
    owner.save(OUT / 'manifest.json', {'requests': details, 'formal_training': False})
    print(json.dumps(details))


if __name__ == '__main__': main()

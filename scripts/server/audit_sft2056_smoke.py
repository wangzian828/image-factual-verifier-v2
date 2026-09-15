"""Engineering acceptance for epoch-2 serving; no labels or judge decisions."""
from __future__ import annotations
import argparse
from collections import Counter
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys

from benchmark_sft2056_serving import ROOT, WORK, digest, save
import run_sft2056_eval  # Configure the shared frozen-protocol controller.
import run_sft3084_eval as controller

REPO = ROOT/'image-factual-verifier-v2'
POLICY = ROOT/'evaluation/factcheck-formal1527-available1526-20260912/runtime-release/evaluator_private/source_access_policy.json'


def images(value):
    if isinstance(value, list):
        for item in value:
            yield from images(item)
    elif isinstance(value, dict):
        if value.get('type') == 'image_url':
            yield value['image_url']['url']
        else:
            for item in value.values():
                yield from images(item)


def multiimage():
    import requests
    workload = json.loads((WORK/'workload.json').read_text())
    assert digest(WORK/'workload.json') == 'be40282d75e702bb1d019652648ab896d2fdef50cc086bbd205b282d3c03fc2e'
    urls = [next(images(workload[i]['body']['messages'])) for i in (0, 4)]
    assert len(set(urls)) == 2
    body = {'model': controller.ALIAS, 'messages': [{'role': 'user', 'content':
            [{'type': 'text', 'text': 'Briefly describe the two images separately.'}]
            + [{'type': 'image_url', 'image_url': {'url': u}} for u in urls]}],
            'temperature': 0, 'max_tokens': 256, 'chat_template_kwargs': {'enable_thinking': False}}
    def one(port):
        response = requests.post(f'http://127.0.0.1:{port}/v1/chat/completions', json=body, timeout=180)
        response.raise_for_status()
        data = response.json()
        choice = data['choices'][0]
        return {'port': port, 'passed': choice['finish_reason'] == 'stop' and bool(choice['message'].get('content')),
                'response': data}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(one, (8902, 8903, 8904, 8905)))
    record = {'passed': all(r['passed'] for r in records), 'distinct_images': 2,
              'export_sha256': digest(controller.EXPORT),
              'image_data_url_sha256': [hashlib.sha256(u.encode()).hexdigest() for u in urls],
              'diagnostic_only': True, 'results': records}
    save(controller.OUTPUT/'multiimage-service-probe.json', record)
    print(json.dumps({'multiimage_passed': record['passed'], 'replicas': len(records)}))
    if not record['passed']:
        raise RuntimeError('Two-image service probe did not pass')


def audit():
    sys.path.insert(0, str(REPO))
    from scripts.audit_real_trace import audit_trace, SourceAccessPolicy
    from src.orchestrator.runtime_events import ContentAddressedArtifactStore
    policy = SourceAccessPolicy.load(POLICY)
    smoke = controller.OUTPUT/'smoke'
    expected = set(controller.CANARY.read_text().splitlines())
    selected = controller.successful_traces([smoke], expected)
    assert len(expected) == 4 and set(selected) == expected
    probe = json.loads((controller.OUTPUT/'multiimage-service-probe.json').read_text())
    assert probe['passed'] and probe['export_sha256'] == digest(controller.EXPORT)
    finishes, subcalls = Counter(), Counter()
    records = []
    max_images = max_tokens = max_reasoning = 0
    for case, path in selected.items():
        trace = json.loads(path.read_text())
        strict = audit_trace(path, source_access_policy=policy).to_dict(strict_scheduler=True)
        runtime = Path(trace['state']['runtime_store']['runtime_path'])
        runtime.resolve().relative_to(ROOT)
        store = ContentAddressedArtifactStore(runtime/'artifacts/sha256')
        case_calls = Counter()
        budgets = []
        case_finishes = Counter()
        for step in trace['state']['all_steps']:
            metadata = step.get('metadata') or {}
            finish = metadata.get('finish_reason')
            if finish:
                finishes[finish] += 1
                case_finishes[finish] += 1
            for call in metadata.get('tool_subcalls') or []:
                key = (call.get('kind'), call.get('provider'), call.get('status'))
                subcalls[key] += 1
                case_calls[key] += 1
            max_reasoning = max(max_reasoning, metadata.get('response_reasoning_chars') or 0)
            request_id = metadata.get('context_request_id')
            if not request_id:
                continue
            context = json.loads((runtime/'context'/f'{request_id}.json').read_text())
            max_images = max(max_images, context.get('image_count') or 0)
            max_tokens = max(max_tokens, context.get('provider_output_tokens') or 0)
            if step.get('stage') not in ('unified_react', 'unified_judgment', 'judgment'):
                continue
            configs = [x for x in context['context_items'] if x['kind'] == 'generation_config']
            assert len(configs) == 1
            config = json.loads(store.read_bytes(configs[0]['artifact']))
            budget = {'output': context['max_output_tokens'], 'thinking': config.get('thinking_token_budget'),
                      'enabled': config.get('enable_thinking')}
            budgets.append(budget)
            assert budget == {'output': 32768, 'thinking': 8192, 'enabled': True}
        external = sum(count for (kind, provider, status), count in case_calls.items()
                       if status == 'success' and provider in ('serper', 'serper_lens', 'serper_image',
                                                            'jina_reader', 'jina_reranker', 'baidu_ocr', 'gemini'))
        records.append({'case_id': case, 'strict_audit': strict,
            'external_successful_subcalls': external, 'finish_reasons': dict(case_finishes),
            'main_request_budgets_checked': len(budgets),
            'tool_subcalls': {str(k):v for k,v in case_calls.items()}})
    passed = all(r['strict_audit']['passed'] and r['external_successful_subcalls'] > 0
                 and r['main_request_budgets_checked'] > 0 for r in records)
    passed = passed and not (finishes['length'] or finishes['abort'])
    report = {'passed': bool(passed), 'export_sha256': digest(controller.EXPORT),
        'trace_sha256': {case:digest(path) for case,path in selected.items()},
        'strict_audit_passed': all(r['strict_audit']['passed'] for r in records),
        'external_tools_verified': all(r['external_successful_subcalls'] > 0 for r in records),
        'multimodal_verified': probe['passed'], 'source_policy_sha256': digest(POLICY),
        'finish_reasons': dict(finishes), 'max_image_slots': max_images,
        'max_output_tokens_observed': max_tokens, 'max_reasoning_characters': max_reasoning,
        'subcall_status_counts': {str(k):v for k,v in subcalls.items()}, 'traces': records,
        'scope': 'Engineering acceptance only. No gold, judge selection or wall-time speed claim.'}
    save(controller.OUTPUT/'smoke-audit.json', report)
    if not passed:
        raise RuntimeError('Smoke engineering acceptance failed; inspect preserved audit')
    save(controller.OUTPUT/'smoke-acceptance.json', report)
    controller.verify_acceptance(controller.OUTPUT/'smoke-acceptance.json', smoke, expected, digest(controller.EXPORT))
    print(json.dumps({k:v for k,v in report.items() if k not in ('traces','trace_sha256')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['multiimage', 'audit'])
    args = parser.parse_args()
    multiimage() if args.stage == 'multiimage' else audit()

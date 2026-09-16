"""Read-only diagnosis of archived hinted requests; only calls /tokenize.

No generation, no external API, no source edits. Output contains structural
differences/counts, never text, images, credentials, or private checker material.
"""
from __future__ import annotations
import gzip
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path('/volume/ybo/wza')
CODE = ROOT / 'training-artifacts/psd-epoch3-repair-20260916-v6/code'
RUN = ROOT / 'runs/psd-sft3084-captured-canary4x8-20260916'
WIRE = ROOT / 'inference/psd-sft3084-20260916/captured-v2/wire'


def load(path):
    return json.loads(path.read_text())


def tokenize(body):
    request = urllib.request.Request('http://127.0.0.1:19003/tokenize',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)['tokens']


def compare(actual, expected):
    return {'equal': actual == expected, 'actual_count': len(actual), 'expected_count': len(expected),
        'first_mismatch': next((i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]), None)}


def structural_diff(left, right, path=''):
    if type(left) is not type(right):
        return [path + ':type']
    if isinstance(left, dict):
        return ([path + ':keys'] if set(left) != set(right) else []) + [
            item for key in left.keys() & right.keys()
            for item in structural_diff(left[key], right[key], path + '/' + key)]
    if isinstance(left, list):
        return ([path + ':length'] if len(left) != len(right) else []) + [
            item for i, (a, b) in enumerate(zip(left, right))
            for item in structural_diff(a, b, path + '/' + str(i))]
    return [] if left == right else [path + ':value']


def main():
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    from src.orchestrator.runtime_events import reconstruct_archived_request
    from ifv_training.psd_slate import hydrate_live_snapshot, tokenizer_request
    from src.orchestrator.stage_runner import StageRunner
    from src.orchestrator.llm_backend import APIBackend
    threshold = load(ROOT / 'training-artifacts/psd-epoch3-repair-20260916-v6/process.json')['time']
    bodies = []
    for directory in WIRE.iterdir():
        if load(directory / 'request-meta.json')['started_at'] < threshold:
            continue
        with gzip.open(directory / 'request.json.gz', 'rt') as file:
            body = json.load(file)
        bodies.append((directory, body))
    for repair in sorted((RUN / 'psd-round-v5/search/repairs').iterdir()):
        state = load(repair / 'slate-state.json')['payload']
        pending = state.get('pending_proposal', {})
        hints = [row['text'] for row in pending.get('hints', {}).values()]
        runtime = sorted(repair.glob('runtime/*/slate-*'))
        if len(runtime) != 1:
            print(json.dumps({'repair': repair.name, 'runtime_count': len(runtime)}))
            continue
        contexts = sorted((runtime[0] / 'context').glob('req-*.json'))
        latest = contexts[-1].stem
        report = {'repair': repair.name, 'last_request_id': latest, 'requests': len(contexts)}
        matched = [(d, b) for d, b in bodies if any(
            m.get('role') == 'user' and m.get('content') in hints
            for m in b.get('messages', []) if isinstance(m.get('content'), str))]
        report['hinted_wire_requests'] = len(matched)
        results = []
        for directory, body in matched:
            if not (directory / 'response.json.gz').exists():
                continue
            with gzip.open(directory / 'response.json.gz', 'rt') as file:
                response = json.load(file)
            expected = response.get('prompt_token_ids')
            if not expected:
                continue
            request = reconstruct_archived_request(str(runtime[0]), latest)
            raw = {'model': body['model'], 'messages': body['messages'],
                   'add_generation_prompt': True, 'add_special_tokens': False}
            for key in ('tools', 'chat_template_kwargs', 'mm_processor_kwargs', 'media_io_kwargs'):
                if key in body:
                    raw[key] = body[key]
            result = {'wire': directory.name, 'raw_tokenizer': compare(tokenize(raw), expected),
                      'finish_reason': response['choices'][0]['finish_reason']}
            try:
                # Recreate live snapshot's image placeholders; only archived bytes are restored.
                request['input_payload'] = hydrate_live_snapshot(
                    StageRunner._snapshot_input_payload(body['messages']), request['input_payload'])
                result['hydrate_live_snapshot'] = 'passed'
                archived_tools = APIBackend._openai_tool_schemas(request.get('tool_schema', []))
                result['archived_tools_equal'] = body.get('tools', []) == archived_tools
                result['tool_differences'] = structural_diff(body.get('tools', []), archived_tools)
                if result['archived_tools_equal']:
                    request['tool_schema'] = body.get('tools', [])
                result['target_tokenizer'] = compare(tokenize(tokenizer_request(request, model=body['model'])), expected)
            except Exception as error:
                result['error_type'] = type(error).__name__
                result['error'] = str(error)[:200]
            results.append(result)
        report['results'] = results
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

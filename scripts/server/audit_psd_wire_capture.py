"""Read-only audit of complete wire receipts, including otherwise lost failures."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time


def read_bound(directory, side):
    meta = json.loads((directory / (side + '-meta.json')).read_text())
    body = gzip.decompress((directory / (side + '.json.gz')).read_bytes())
    if len(body) != meta['bytes'] or hashlib.sha256(body).hexdigest() != meta['sha256']:
        raise ValueError('Wire body does not match its completed receipt')
    return json.loads(body), meta


def describe_choice(choice):
    message = choice.get('message') or {}
    content = message.get('content') or ''
    reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
    calls = message.get('tool_calls') or []
    malformed_arguments = 0
    for call in calls:
        try:
            arguments = call['function']['arguments']
            if not isinstance(arguments, str) or not isinstance(json.loads(arguments), dict):
                malformed_arguments += 1
        except (KeyError, ValueError, TypeError):
            malformed_arguments += 1
    return {'finish_reason': choice.get('finish_reason'),
            'content_chars': len(content), 'reasoning_chars': len(reasoning),
            'tool_calls': len(calls), 'malformed_argument_json': malformed_arguments,
            'unusable': not bool(content.strip() or calls),
            'reasoning_unique_chars': len(set(reasoning)),
            'reasoning_top_chars': Counter(reasoning).most_common(3)}


def audit(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError('Capture root does not exist')
    choices = Counter()
    pending, failures, anomalies = [], [], []
    completed = started = prompt_max = output_max = 0
    total_bytes = 0
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        if not (directory/'request-meta.json').exists():
            # The writer has not committed a request receipt yet.
            pending.append({'ticket': directory.name, 'phase': 'request_capture_in_progress'})
            continue
        request, request_meta = read_bound(directory, 'request')
        started += 1
        total_bytes += request_meta.get('gzip_bytes', 0)
        if not (directory/'response-meta.json').exists():
            if (directory/'error.json').exists():
                failures.append({'ticket': directory.name,
                                 **json.loads((directory/'error.json').read_text())})
            else:
                pending.append({'ticket': directory.name, 'phase': 'response_pending',
                                'age_seconds': round(time.time()-request_meta['started_at'])})
            continue
        response, response_meta = read_bound(directory, 'response')
        completed += 1
        total_bytes += response_meta.get('gzip_bytes', 0)
        usage = response.get('usage') or {}
        prompt_max = max(prompt_max, usage.get('prompt_tokens') or 0)
        output_max = max(output_max, usage.get('completion_tokens') or 0)
        rows = [describe_choice(c) for c in response.get('choices', [])]
        for row in rows:
            choices[row['finish_reason']] += 1
        if (response_meta['status_code'] != 200 or not rows or any(
                r['unusable'] or r['finish_reason'] == 'length' or r['malformed_argument_json'] for r in rows)):
            anomalies.append({'ticket': directory.name, 'http_status': response_meta['status_code'],
                'choices': rows, 'usage': usage,
                'request_budget': {k: request.get(k) for k in
                    ['max_tokens', 'max_completion_tokens', 'vllm_xargs', 'seed', 'tool_choice']}})
    return {'started': started, 'completed': completed, 'pending': pending,
            'capture_failures': failures, 'anomalies': anomalies,
            'finish_reasons': dict(choices), 'max_prompt_tokens': prompt_max,
            'max_completion_tokens': output_max, 'compressed_body_bytes': total_bytes,
            'psd_gate_passed': False,
            'scope': 'wire integrity and transport/syntax only; not tool schema, checker, or PSD validation'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), ensure_ascii=False, indent=2))

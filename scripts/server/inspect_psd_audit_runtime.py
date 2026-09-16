"""Read-only structural audit of fixed source traces and new runtime failures."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT / 'training-artifacts/psd-contract-audit-20260916-v10'
RUN = ROOT / 'runs/psd-sft3084-captured-canary4x8-20260916'
sys.path[:0] = [str(DEPLOY / 'code'), str(DEPLOY / 'code/training')]
from ifv_training.psd_gemini_judge import trace_steps


def main():
    source_counts = Counter()
    for path in sorted((RUN / 'episodes/traces').glob('*.json')):
        trace = json.loads(path.read_text())
        old, new = trace_steps(trace, include_transport_ids=False), trace_steps(trace)
        source_counts['traces'] += 1
        for original, before, after in zip(trace['state']['all_steps'], old, new, strict=True):
            call_id = original.get('metadata', {}).get('function_call_id')
            if call_id:
                source_counts['tool_ids_now_projected'] += after.get('function_call_id') == call_id
                source_counts['tool_ids_absent_from_old_projection'] += call_id not in json.dumps(before)
            capture = original.get('metadata', {}).get('policy_token_capture', {})
            if capture.get('status') == 'complete':
                source_counts['complete_native_captures'] += 1
    errors, contexts = [], Counter()
    for path in (RUN / 'psd-contract-audit-v10/repairs').glob('*/runtime/*/*/context/*.json'):
        row = json.loads(path.read_text())
        contexts[row.get('status', 'unknown')] += 1
        if row.get('error'):
            error = row['error']
            errors.append({'context': str(path), 'error_chars': len(error),
                'http_status': re.findall(r'(?:HTTP\s+|status(?:_code)?[ :=]+)([45]\d\d)', error, re.I),
                'known_markers': [s for s in ('maximum context', 'max_tokens', 'timeout', '429',
                    '503', '400', '500', 'schema', 'tool_choice', 'grammar', 'images',
                    'not found', 'prefix', 'thinking', 'cache', 'token_ids', 'invalid',
                    'tool_call_id', 'parser', 'unsupported', 'required') if s in error.lower()],
                'sha256': hashlib.sha256(error.encode()).hexdigest()})
    print(json.dumps({'source_counts': source_counts, 'runtime_context_statuses': contexts,
                      'runtime_errors': errors}, ensure_ascii=False))


if __name__ == '__main__':
    main()

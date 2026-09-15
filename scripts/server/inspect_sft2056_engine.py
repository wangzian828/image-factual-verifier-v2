"""Read-only aggregate engine and trace diagnostics; never emits API secrets."""
from pathlib import Path
from collections import Counter
import argparse
import json

ROOT = Path('/volume/ybo/wza')
WORK = ROOT/'benchmarks/sft2056-serving-ab-20260915'


def engine():
    rows = []
    for run in ('fixed-token-v1', 'fixed-token-apc16', 'fixed-token-apc16-cold-primed'):
        for path in sorted((WORK/run).glob('*/pass-*.json')):
            data = json.loads(path.read_text())
            delta = data['engine_metric_delta']
            row = {k:data[k] for k in ('pass', 'passed', 'wall_seconds', 'tokens_per_second')}
            row.update(run=run, profile=path.parent.name)
            for name in ('request_queue_time_seconds', 'request_prefill_time_seconds',
                         'request_decode_time_seconds', 'time_to_first_token_seconds',
                         'request_time_per_output_token_seconds'):
                count = delta.get('vllm:'+name+'_count')
                if count:
                    row[name] = delta['vllm:'+name+'_sum']/count
            row['prefix_hits'] = delta.get('vllm:prefix_cache_hits_total')
            row['prefix_queries'] = delta.get('vllm:prefix_cache_queries_total')
            row['preemptions'] = delta.get('vllm:num_preemptions_total')
            rows.append(row)
    print(json.dumps(rows, indent=2))


def shape():
    smoke = ROOT/'runs/eval/qwen35-sft3084-3epoch-agent-formal1527-20260915/smoke'
    trace = json.loads(next((smoke/'traces').glob('*.json')).read_text())
    print('trace_keys', sorted(trace))
    print('state_keys', sorted(trace['state']))
    for step in trace['state']['all_steps'][:5]:
        print('step_keys', sorted(step))
        print('metadata_keys', sorted(step.get('metadata', {})))
        print('tokens', step.get('tokens'))
        result = step.get('tool_result')
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                pass
        if isinstance(result, dict):
            print('tool_result_keys', sorted(result))
            print('tool_result_shape', {k: sorted(v) if isinstance(v, dict) else type(v).__name__
                  for k,v in result.items()})
    runtime = Path(trace['state']['runtime_store']['runtime_path'])
    runtime.resolve().relative_to(ROOT)
    context = json.loads(next((runtime/'context').glob('*.json')).read_text())
    print('context_keys', sorted(context))


def progress():
    output = ROOT/'runs/eval/qwen35-sft2056-epoch2-agent-formal1527-20260915'
    report = {}
    for name in ('pipeline-state.json', 'progress.json'):
        if (output/name).exists():
            report[name] = json.loads((output/name).read_text())
    traces = list((output/'smoke/traces').glob('*.json'))
    report['smoke_finished'] = len(traces)
    report['smoke_success'] = sum(not (row:=json.loads(p.read_text())).get('error')
                                  and bool(row.get('fact_check_report')) for p in traces)
    finishes, subcalls = Counter(), Counter()
    for path in traces:
        for step in json.loads(path.read_text())['state']['all_steps']:
            metadata = step.get('metadata') or {}
            if metadata.get('finish_reason'):
                finishes[metadata['finish_reason']] += 1
            for call in metadata.get('tool_subcalls') or []:
                subcalls[(call.get('provider'), call.get('status'))] += 1
    report.update(finish_reasons=dict(finishes), subcall_statuses={str(k):v for k,v in subcalls.items()})
    counts = Counter()
    max_tokens = max_images = 0
    runtime = output/'smoke/traces/runtime'
    for path in runtime.rglob('context/req-*.json'):
        row = json.loads(path.read_text())
        counts[(row.get('stage'), row.get('status'))] += 1
        max_tokens = max(max_tokens, row.get('provider_output_tokens') or 0)
        max_images = max(max_images, row.get('image_count') or 0)
    report.update(request_statuses={str(k):v for k,v in counts.items()},
                  max_output_tokens=max_tokens, max_image_slots=max_images)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['engine', 'shape', 'progress'])
    args = parser.parse_args()
    {'engine': engine, 'shape': shape, 'progress': progress}[args.stage]()

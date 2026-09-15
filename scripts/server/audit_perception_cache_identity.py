"""Read-only trace/image audit; write a new report, never rewrite old results."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/volume/ybo/wza')
OUT = ROOT/'evaluation/perception-cache-identity-audit-20260916'
SOURCES = {
    'epoch2': ROOT/'evaluation/qwen35-sft2056-epoch2-agent-budgeted1524-20260916/selected-traces.json',
    'epoch3': ROOT/'runs/eval/qwen35-sft3084-3epoch-agent-formal1527-20260915/selected-traces.json',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def collect_record(trace, *, case, image_sha, path, trace_sha):
    result = []
    for step in trace.get('state', {}).get('all_steps', []):
        if step.get('tool_name') not in {'perceive_scene', 'ocr_with_position'}:
            continue
        raw = step.get('tool_result')
        if not isinstance(raw, str) or not raw:
            continue
        result.append({'case_id': case, 'image_sha256': image_sha, 'path': str(path),
            'trace_sha256': trace_sha, 'tool': step['tool_name'], 'result_sha256': sha(raw.encode()),
            'cache_hit': step.get('metadata', {}).get('cache_hit') is True})
    return result


def summarize(records):
    grouped = {}
    for row in records:
        grouped.setdefault((row['tool'], row['result_sha256']), []).append(row)
    collisions = []
    for (tool, result_sha), group in grouped.items():
        images = {r['image_sha256'] for r in group}
        if len(images) <= 1 or not any(r['cache_hit'] for r in group):
            continue
        collisions.append({'tool': tool, 'result_sha256': result_sha,
            'distinct_images': len(images), 'cases': len({r['case_id'] for r in group}),
            'cache_hit_cases': len({r['case_id'] for r in group if r['cache_hit']}),
            'records': group})
    affected = {r['case_id'] for g in collisions for r in g['records'] if r['cache_hit']}
    return {'tool_calls': len(records), 'cache_hit_calls': sum(r['cache_hit'] for r in records),
        'collision_groups': len(collisions), 'cache_hit_cases_in_cross_image_groups': len(affected),
        'groups': collisions,
        'interpretation': 'Cross-image identical cache responses are evidence of missing image identity, '
                          'not proof that every final verdict changed; no outputs or metrics are rewritten.'}


def main():
    assert not OUT.exists(), 'Audit output already exists; inspect it rather than overwrite'
    OUT.mkdir()
    all_summary = {}
    for label, index in SOURCES.items():
        raw = index.read_bytes()
        selected = json.loads(raw)
        records, image_hashes = [], {}
        for case, entry in selected.items():
            path = Path(entry['path']).resolve()
            path.relative_to(ROOT)
            data = path.read_bytes()
            trace_sha = sha(data)
            assert trace_sha == entry['sha256'], str(path)
            trace = json.loads(data)
            image = Path(trace['image_path']).resolve()
            image.relative_to(ROOT)
            if image not in image_hashes:
                image_hashes[image] = sha(image.read_bytes())
            records.extend(collect_record(trace, case=case, image_sha=image_hashes[image],
                                          path=path, trace_sha=trace_sha))
        result = summarize(records)
        result.update(source=str(index), source_sha256=sha(raw), selected_cases=len(selected), time=time.time())
        (OUT/(label+'.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        all_summary[label] = {k:v for k,v in result.items() if k != 'groups'}
        print(label, json.dumps(all_summary[label]), flush=True)
    (OUT/'summary.json').write_text(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

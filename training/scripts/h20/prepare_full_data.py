"""Rebind a frozen delivery without changing messages/splits; audit and profile every row.

Only JSON/metadata are materialized; original images are referenced, never copied.
Run in the isolated H20 environment. No model weights or GPU training are loaded.
"""
import argparse
import concurrent.futures as futures
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'training'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    # JSONL boundaries are LF, not every Unicode separator accepted by splitlines().
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def dist(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return {'count': len(values), 'sum': sum(values), 'min': values[0], 'max': values[-1],
            'mean': sum(values)/len(values),
            **{f'p{q}': values[max(0, math.ceil(len(values)*q/100)-1)] for q in (25, 50, 75, 90, 95, 99)}}


def check_file(item):
    path, expected = item
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Source checksum mismatch: {path}')
    if path.parent.name == 'images':
        from PIL import Image
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        return {'name': path.name, 'width': width, 'height': height, 'bytes': path.stat().st_size}
    return None


def init_worker(model):
    global probe, processor, template
    spec = importlib.util.spec_from_file_location('ifv_processor_probe', REPO/'training/scripts/probe/verify_ms_swift_agent_dataset.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    from swift import get_processor, get_template
    processor = get_processor(model, model_type='qwen3_5')
    template = get_template(processor, max_length=131072, truncation_strategy='raise',
                            max_pixels=262144, padding_free=True, sequence_parallel_size=4,
                            loss_scale='ignore_empty_think', enable_thinking=False,
                            add_non_thinking_prefix=False)
    template.set_mode('train')


def encode_row(task):
    kind, split, index, row = task
    record = {'kind': kind, 'split': split, 'row_index': index}
    try:
        roles = probe._validate_roles(row['messages'], kind=kind)
        encoded = template.encode(row, return_template_inputs=True)
        ids, labels = map(probe._as_list, (encoded['input_ids'], encoded['labels']))
        assert len(ids) == len(labels) and 0 < len(ids) <= 131072
        supervised = sum(x != -100 for x in labels)
        assert supervised > 0
        image_count = len(getattr(encoded['template_inputs'], 'images', []) or [])
        assert image_count == len(row['images']), 'Image count changed'
        decoded = processor.tokenizer.decode(ids, skip_special_tokens=False)
        calls = probe._verify_rendered_tool_calls(row['messages'], decoded) if kind == 'policy' else 0
        thoughts = sum(m['role'] == 'assistant' and '<think>' in m['content'] for m in row['messages'])
        if thoughts:
            assert probe._contains_supervised_subsequence(ids, labels, probe._encode(processor.tokenizer, '<think>'))
        responses = 0
        for message in row['messages']:
            if message['role'] == 'tool_response':
                needle = probe._encode(processor.tokenizer, probe._text_for_presence_check(message['content']))
                assert not needle or probe._contains_subsequence(ids, needle), 'Tool response lost'
                responses += 1
        record.update(input_tokens=len(ids), trainable_tokens=supervised, image_count=image_count,
                      message_count=len(roles), tool_call_count=calls, tool_response_count=responses,
                      thought_targets=thoughts)
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    allowed = Path(os.environ['IFV_H20_ROOT']).resolve(strict=True)
    package = args.package.resolve(strict=True)
    output = args.output.resolve()
    assert package.is_relative_to(allowed) and output.is_relative_to(allowed)
    assert output != allowed and not output.exists(), 'Use a new derived directory'
    output.mkdir(parents=True)
    write(output/'STATUS.json', {'status': 'verifying_source', 'training_started': False})
    source_files = []
    for line in (package/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = (package/name.lstrip('*')).resolve(strict=True)
        assert path.is_relative_to(package)
        source_files.append((path, digest))
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        image_stats = [r for r in pool.map(check_file, source_files) if r is not None]
    print('SOURCE_HASHES_AND_IMAGES_OK', len(source_files), len(image_stats), flush=True)
    write(output/'source-verification.json', {'passed': True, 'files': len(source_files),
          'sha256sums_sha256': sha(package/'SHA256SUMS'), 'image_count': len(image_stats),
          'image_width': dist([i['width'] for i in image_stats]),
          'image_height': dist([i['height'] for i in image_stats]),
          'image_bytes': dist([i['bytes'] for i in image_stats])})
    source_manifest = json.loads((package/'SOURCE_MANIFEST.json').read_text())
    frozen = {r['case_id']: r for r in rows(package/'case-split/case_split.jsonl')}
    policy_cases = {r['episode_id']: r['case_id'] for r in rows(package/'ms-swift-policy/index.jsonl')}
    assert sha(package/'case-split/case_split.jsonl') == source_manifest['case_split']['sha256']
    from ifv_training.audit import audit_derived_dataset
    tasks, audits, dataset_files = [], {}, []
    features, identities = [], defaultdict(lambda: defaultdict(set))
    unique_images = set()
    known_images = {r['name'] for r in image_stats}
    for kind in ('policy', 'perception'):
        src, dst = package/f'ms-swift-{kind}', output/f'ms-swift-{kind}'
        dst.mkdir()
        manifest = json.loads((src/'manifest.json').read_text())
        source_manifest_sha = sha(src/'manifest.json')
        index = rows(src/'index.jsonl')
        shutil.copyfile(src/'index.jsonl', dst/'index.jsonl')
        by_split = {s: [r for r in index if r['split'] == s] for s in ('train', 'validation', 'test')}
        for split in ('train', 'validation', 'test'):
            data = rows(src/f'{split}.jsonl')
            assert len(data) == len(by_split[split])
            with (dst/f'{split}.jsonl').open('w') as stream:
                for i, row in enumerate(data):
                    meta = by_split[split][i]
                    assert meta['source_index'] == i, 'Split-local index ordering mismatch'
                    case = meta.get('case_id') or policy_cases.get(meta['episode_id'])
                    if case is None:
                        # Action-only perception rows have no policy entry. Bind by the
                        # frozen source-image identity; never assume all IDs are main-N.
                        candidates = [key for key, value in frozen.items()
                                      if value['image_sha256'] == meta['image_sha256'] and value['split'] == split]
                        parsed = re.fullmatch(r'initial-a\d+-n\d+-(.+)--[0-9a-f]+--r\d+', meta['episode_id'])
                        if parsed and parsed.group(1) in candidates:
                            case = parsed.group(1)
                        else:
                            assert len(candidates) == 1, f'Ambiguous perception case identity: {meta["episode_id"]}'
                            case = candidates[0]
                    assert frozen[case]['split'] == split, 'Frozen split mismatch'
                    for field, value in [('case', case), ('group', frozen[case]['split_group_id']),
                                         ('source_image', frozen[case]['image_sha256'])]:
                        identities[field][split].add(value)
                    paths = []
                    for original in row['images']:
                        name = Path(original).name
                        assert name in known_images, f'Image not in verified delivery: {name}'
                        path = package/'images'/name
                        paths.append(str(path))
                        unique_images.add(name)
                        identities['referenced_image'][split].add(name)
                    row['images'] = paths
                    identity = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    identities['exact_row'][split].add(identity)
                    tools = Counter(json.loads(m['content'])['name'] for m in row['messages'] if m['role'] == 'tool_call')
                    verdict = re.findall(r'"verdict"\s*:\s*"(real|fake)"', row['messages'][-1]['content'])
                    features.append({'kind': kind, 'split': split, 'row_index': i, 'case_id': case,
                                     'episode_id': meta['episode_id'], 'row_sha256': identity,
                                     'tool_names': dict(tools), 'teacher_verdict': verdict[-1] if verdict else 'unparsed',
                                     'assistant_think_count': sum(m['role']=='assistant' and '<think>' in m['content'] for m in row['messages'])})
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
                    tasks.append((kind, split, i, row))
            target = dst/f'{split}.jsonl'
            manifest['artifacts'][split].update(rows=len(data), sha256=sha(target))
            dataset_files.append({'kind': kind, 'path': str(target), 'size': target.stat().st_size, 'sha256': sha(target)})
        manifest['relocation'] = {'source_manifest_sha256': source_manifest_sha,
                                  'source_directory': str(src), 'changes': 'image paths only; row order/messages/splits unchanged'}
        write(dst/'manifest.json', manifest)
        audit = audit_derived_dataset(dst)
        audits[kind] = audit
        write(output/f'{kind}-strict-audit.json', audit)
        assert audit['passed'], f'{kind} strict audit failed: {audit["errors"][:3]}'
    overlaps = {field: {f'{a}:{b}': len(splits[a] & splits[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
                for field, splits in identities.items()}
    assert not any(n for field in ('case','group','source_image','exact_row') for n in overlaps[field].values()), 'Split leakage'
    write(output/'relocation-audit.json', {'passed': True, 'cross_split_overlap': overlaps,
          'referenced_unique_images': len(unique_images), 'frozen_split_counts': dict(Counter(r['split'] for r in frozen.values()))})
    print('RELOCATION_AND_STRICT_AUDIT_OK', len(tasks), flush=True)
    write(output/'STATUS.json', {'status': 'encoding_all_rows', 'total_rows': len(tasks)})
    results = []
    with (output/'row-features.jsonl').open('w') as stream, futures.ProcessPoolExecutor(
            max_workers=args.workers, initializer=init_worker,
            initargs=(str(allowed/'models/Qwen3.5-9B-local'),)) as pool:
        for feature, record in zip(features, pool.map(encode_row, tasks, chunksize=1)):
            record.update(feature)
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            stream.flush()
            results.append(record)
            if len(results) % 50 == 0 or 'error' in record:
                print('ENCODED', len(results), '/', len(tasks), 'ERRORS', sum('error' in r for r in results), flush=True)
    errors = [r for r in results if 'error' in r]
    good = [r for r in results if 'error' not in r]
    summaries = {}
    for kind in ('policy','perception'):
        for split in ('train','validation','test'):
            subset = [r for r in good if r['kind']==kind and r['split']==split]
            counts = Counter()
            for record in subset:
                counts.update(record['tool_names'])
            bounds = [8192,16384,32768,65536,98304,120000,131072]
            summaries[f'{kind}:{split}'] = {
                **{field: dist([r[field] for r in subset]) for field in ('input_tokens','trainable_tokens','image_count','message_count','tool_call_count')},
                'teacher_verdicts_not_gold': dict(Counter(r['teacher_verdict'] for r in subset)),
                'tool_calls_by_name': dict(counts),
                'length_buckets': {f'({lo},{hi}]': sum(lo < r['input_tokens'] <= hi for r in subset) for lo,hi in zip([0]+bounds[:-1],bounds)},
                'exact_duplicate_rows': len(subset)-len({r['row_sha256'] for r in subset}),
            }
    template_contract = dict(max_length=131072, truncation_strategy='raise', max_pixels=262144,
        padding_free=True, sequence_parallel_size=4, loss_scale='ignore_empty_think',
        enable_thinking=False, add_non_thinking_prefix=False, image_max_token_num=1024)
    report = {'schema_version':'ifv-ms-swift-agent-processor-verification-v2',
        'created_at':datetime.now(timezone.utc).isoformat(), 'model':str(allowed/'models/Qwen3.5-9B-local'),
        'template_contract':template_contract, 'max_context':131072, 'dataset_files':dataset_files,
        'passed':not errors, 'error_count':len(errors), 'errors':errors,
        'counts':dict(Counter(f'{r["kind"]}:{r["split"]}' for r in good)),
        'checks':{'rows_encoded':len(good), 'tool_call_preserved':sum(r['tool_call_count'] for r in good),
                  'tool_response_preserved':sum(r['tool_response_count'] for r in good),
                  'thought_targets':sum(r['thought_targets'] for r in good)},
        'summaries':summaries, 'source_package':str(package),
        'commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()}
    write(output/'processor-verification.json', report)
    write(output/'STATUS.json', {'status':'ready' if not errors else 'failed', 'rows':len(results),
          'errors':len(errors), 'training_started':False, 'validation_warning':'Only one held-out case; not sufficient for quality selection.'})
    print(json.dumps({'passed':not errors,'rows':len(results),'errors':len(errors),'summaries':summaries},ensure_ascii=False,indent=2),flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

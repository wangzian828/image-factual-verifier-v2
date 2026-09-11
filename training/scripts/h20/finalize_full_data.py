"""Bind full processor evidence to launch gates and create index-only length plans."""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import os
from pathlib import Path

from prepare_full_data import REPO, dist, rows, sha, write


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', type=Path, required=True)
    args = parser.parse_args()
    root = args.processed.resolve(strict=True)
    assert root.is_relative_to(Path(os.environ['IFV_H20_ROOT']).resolve())
    report_path = root/'processor-verification.json'
    report = json.loads(report_path.read_text())
    assert report['passed'] and report['error_count'] == 0
    features = rows(root/'row-features.jsonl')
    assert len(features) == sum(report['counts'].values())
    assert all('error' not in r for r in features)
    probe = load_module('probe_final', REPO/'training/scripts/probe/verify_ms_swift_agent_dataset.py')
    gate = load_module('gate_final', REPO/'training/scripts/probe/verify_sft_data_contract.py')
    report['input_tokens_by_dataset'] = {}
    report['input_token_boundaries_by_dataset'] = {}
    equality = {}
    source = Path(report['source_package'])
    buckets = defaultdict(list)
    for kind in ('policy','perception'):
        for split in ('train','validation','test'):
            path = root/f'ms-swift-{kind}/{split}.jsonl'
            originals, normalized = rows(source/f'ms-swift-{kind}/{split}.jsonl'), rows(path)
            subset = [r for r in features if r['kind']==kind and r['split']==split]
            assert len(originals)==len(normalized)==len(subset)
            for before, after in zip(originals, normalized):
                assert len(before['images'])==len(after['images'])
                assert [Path(p).name for p in before['images']]==[Path(p).name for p in after['images']]
                assert {k:v for k,v in before.items() if k!='images'}=={k:v for k,v in after.items() if k!='images'}
            equality[f'{kind}:{split}'] = len(subset)
            report['input_tokens_by_dataset'][str(path)] = dist([r['input_tokens'] for r in subset])
            report['input_token_boundaries_by_dataset'][str(path)] = probe._boundary_summary(subset)
            for record in subset:
                band = next(limit for limit in (8192,16384,32768,65536,98304,120000,131072) if record['input_tokens']<=limit)
                buckets[f'{kind}:{split}:le{band}'].append(record['row_index'])
    report['content_equality_audit'] = {'passed':True,'rows':equality,'changes':'image paths only'}
    write(report_path, report)
    gates = {}
    for kind in ('policy','perception'):
        dataset = root/f'ms-swift-{kind}'
        result = gate.verify_sft_data_contract(
            train_jsonl=dataset/'train.jsonl', validation_jsonl=dataset/'validation.jsonl',
            dataset_dir=dataset, processor_report_path=report_path, model=report['model'],
            expected_template_contract=report['template_contract'])
        write(root/f'{kind}-launch-data-gate.json', result)
        gates[kind] = result['passed']
        assert result['passed'], result['errors']
    # Deterministic first-fit-decreasing capacity plan. Indices only; no concatenation.
    packs, unpacked = [], []
    policy_train = [r for r in features if r['kind']=='policy' and r['split']=='train']
    for record in sorted(policy_train,key=lambda r:(-r['input_tokens'],r['row_index'])):
        if record['input_tokens']>120000:
            unpacked.append(record['row_index'])
            continue
        pack = next((p for p in packs if p['tokens']+record['input_tokens']<=120000),None)
        if pack is None:
            pack = {'row_indices':[],'tokens':0,'images':0}
            packs.append(pack)
        pack['row_indices'].append(record['row_index'])
        pack['tokens'] += record['input_tokens']
        pack['images'] += record['image_count']
    indices = unpacked+[i for p in packs for i in p['row_indices']]
    assert sorted(indices)==list(range(len(policy_train)))
    write(root/'length-bucket-index.json',dict(buckets))
    write(root/'packing-plan.json',{'algorithm':'first-fit-decreasing','budget':120000,'packs':packs,
          'unpacked_120k_to_128k':unpacked,'index_only':True,
          'warning':'Planning only, not GPU-memory validation or guaranteed ms-swift packing order; image counts also affect memory.'})
    # Identify tiny/extreme images without dropping legitimate tool results.
    from PIL import Image
    tiny, extreme = [], []
    for path in (source/'images').iterdir():
        with Image.open(path) as image:
            w,h = image.size
        item = {'name':path.name,'width':w,'height':h}
        if min(w,h)<28:
            tiny.append(item)
        if max(w,h)/min(w,h)>100:
            extreme.append(item)
    tiny_names = {r['name'] for r in tiny}
    tiny_references = []
    for kind in ('policy','perception'):
        for split in ('train','validation'):
            for i, row in enumerate(rows(root/f'ms-swift-{kind}/{split}.jsonl')):
                affected = [Path(p).name for p in row['images'] if Path(p).name in tiny_names]
                if affected:
                    tiny_references.append({'kind':kind,'split':split,'row_index':i,'images':affected})
    supplement = {'launch_data_gates':gates, 'content_unchanged':True,
        'policy_source_case_families':dict(Counter('main' if r['case_id'].startswith('main-') else 'route-aware/other' for r in policy_train)),
        'policy_unique_train_cases':len({r['case_id'] for r in policy_train}),
        'policy_teacher_verdicts_not_gold':dict(Counter(r['teacher_verdict'] for r in policy_train)),
        'policy_supervised_token_fraction':sum(r['trainable_tokens'] for r in policy_train)/sum(r['input_tokens'] for r in policy_train),
        'packing_plan':{'packs':len(packs),'tokens':dist([p['tokens'] for p in packs]),
            'images':dist([p['images'] for p in packs]),'unpacked_rows':len(unpacked),
            'fill_fraction':sum(p['tokens'] for p in packs)/(len(packs)*120000)},
        'tiny_images_min_dimension_lt28':tiny,'extreme_images_aspect_gt100':extreme,
        'tiny_image_references':tiny_references,
        'quality_scope':'Format, integrity, provenance and actual processor checks; no new teacher judge or semantic correctness certification.'}
    write(root/'data-characteristics.json',supplement)
    write(root/'RELEASE.json',{'status':'data_processing_complete','training_started':False,
        'processor_report_sha256':sha(report_path),'row_features_sha256':sha(root/'row-features.jsonl'),
        'policy_train_sha256':sha(root/'ms-swift-policy/train.jsonl'),
        'perception_train_sha256':sha(root/'ms-swift-perception/train.jsonl'),
        'validation_warning':'Only one accepted validation case. Not sufficient for model-quality selection.',
        'images':'Referenced in original delivery; do not delete the source images directory.'})
    print(json.dumps(supplement,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

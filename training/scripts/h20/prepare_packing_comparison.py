"""Select one reproducible complete-episode pool for bounded packing comparisons."""
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ['IFV_H20_ROOT'])
out = Path(os.environ['IFV_H20_RUN_ROOT'])
source = root / 'data/ifv-h20-policy-full-20260911'
report = json.loads((source / 'processor-verification.json').read_text())
assert report['passed'] and report['error_count'] == 0
path = source / 'ms-swift-policy/train.jsonl'
raw = path.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert digest == next(x['sha256'] for x in report['dataset_files'] if x['path'] == str(path))
rows = [json.loads(line) for line in raw.decode('utf-8').split('\n') if line.strip()]
features = [json.loads(line) for line in (source / 'row-features.jsonl').read_text().split('\n') if line.strip()]
features = sorted((x for x in features if x['kind'] == 'policy' and x['split'] == 'train'), key=lambda x: x['input_tokens'])
assert len(features) == len(rows) == 2578
# Equal-spaced length quantiles include both endpoints; add image-heavy extremes.
chosen = {features[round(i * (len(features) - 1) / 63)]['row_index'] for i in range(64)}
chosen.update(x['row_index'] for x in sorted(features, key=lambda x: x['image_count'])[-4:])
selected = [x for x in features if x['row_index'] in chosen]
assert max(x['input_tokens'] for x in selected) <= 65536
target = out / 'comparison.jsonl'
assert not target.exists(), target
target.write_text(''.join(json.dumps(rows[i], ensure_ascii=False) + '\n' for i in sorted(chosen)))
audit = {'source_sha256': digest, 'comparison_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
         'rows': len(chosen), 'row_indices': sorted(chosen), 'input_tokens': sum(x['input_tokens'] for x in selected),
         'min_tokens': min(x['input_tokens'] for x in selected), 'max_tokens': max(x['input_tokens'] for x in selected),
         'max_images': max(x['image_count'] for x in selected), 'features': selected,
         'note': 'Same complete-row pool in each run; length-stratified with image-heavy extremes, not a quality eval or exact full-dataset distribution. No truncation or image copies.'}
(out / 'comparison-data-audit.json').write_text(json.dumps(audit, indent=2))
print(json.dumps({k: v for k, v in audit.items() if k not in ('features', 'row_indices')}, indent=2))

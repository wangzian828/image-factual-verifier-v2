"""Explicitly remove completed benchmark weight shards, retaining audit evidence."""
import json
import os
import sys
from pathlib import Path

root = Path(os.environ['IFV_H20_RUN_ROOT']).resolve(strict=True)
run = (root / sys.argv[1]).resolve(strict=True)
assert run.parent == root, 'Expected one benchmark run directly below the run root'
assert json.loads((run / 'result.json').read_text())['exit_code'] == 0, 'Run must have finished successfully'
removed = []
for checkpoint in (run / 'output').glob('checkpoint-*'):
    assert not checkpoint.is_symlink()
    for name in ('pytorch_model_fsdp_0', 'optimizer_0'):
        directory = checkpoint / name
        if not directory.exists():
            continue
        assert not directory.is_symlink()
        assert directory.resolve().is_relative_to(run)
        for shard in directory.glob('*.distcp'):
            assert shard.is_file() and not shard.is_symlink()
            removed.append({'path': str(shard.relative_to(run)), 'bytes': shard.stat().st_size})
            shard.unlink()
    (checkpoint / 'BENCHMARK_SHARDS_REMOVED.txt').write_text(
        'Benchmark weight/optimizer shards were deliberately deleted to reclaim space. '
        'This directory is audit metadata only and cannot be resumed.\n')
report = {'removed_bytes': sum(row['bytes'] for row in removed), 'files': removed,
          'recoverable': False, 'note': 'Source model and training data are untouched.'}
report_path = run / 'checkpoint-cleanup.json'
if report_path.exists():
    report['previous_cleanup'] = json.loads(report_path.read_text())
report_path.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

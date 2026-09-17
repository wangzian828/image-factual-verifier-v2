"""Losslessly compact only completed, locked slots in the current PSD source run.

The original result-cache JSON is preserved byte-for-byte inside gzip. Canonical
and native traces keep every byte and path; equal copies share an inode. No
images, events, contexts, snapshots, old runs or weights are removed. Readers
must use the explicitly deployed gzip-aware PSD snapshot, never the old one.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import tempfile


def atomic_json(path, value, *, staging=None):
    """Publish without a disappearing temporary name inside the measured run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent if staging is None else staging
    staging.mkdir(parents=True, exist_ok=True)
    if staging.stat().st_dev != path.parent.stat().st_dev:
        raise ValueError('Atomic storage publication requires the same filesystem')
    temporary = staging/('.psd-json-'+uuid.uuid4().hex)
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def signature(path):
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for part in iter(lambda: source.read(1024*1024), b''): value.update(part)
    return value.hexdigest()


def pack_cache(cache, *, identity, staging=None):
    from ifv_training.psd_repair_storage import load_bound
    original = json.loads(cache.read_text())
    if original.get('schema_version') == 'ifv-psd-bound-gzip-v1':
        load_bound(cache, identity=identity)
        return {'already_packed': True, 'saved_bytes': 0}
    load_bound(cache, identity=identity)
    before = signature(cache)
    raw = cache.read_bytes()
    if not 0 < len(raw) <= 1024**3:
        raise ValueError('Cache outside bounded compaction size')
    sha = hashlib.sha256(raw).hexdigest()
    archive = cache.with_name(f'result-original-{sha}.json.gz')
    if archive.is_symlink(): raise ValueError('Refuse symlink cache archive')
    if archive.exists():
        with gzip.open(archive, 'rb') as f:
            if f.read(len(raw)+1) != raw: raise ValueError('Existing compressed original differs')
    else:
        stage = cache.parent if staging is None else staging
        stage.mkdir(parents=True, exist_ok=True)
        if stage.stat().st_dev != cache.parent.stat().st_dev:
            raise ValueError('Gzip publication requires same filesystem')
        temporary = stage/('.psd-gzip-'+uuid.uuid4().hex)
        with temporary.open('xb') as fd:
            with gzip.GzipFile(fileobj=fd, mode='wb', compresslevel=1, mtime=0) as stream:
                stream.write(raw)
            fd.flush(); os.fsync(fd.fileno())
        with gzip.open(temporary, 'rb') as f:
            if f.read(len(raw)+1) != raw: raise ValueError('Compression round-trip mismatch')
        os.replace(temporary, archive)
    if signature(cache) != before:
        raise ValueError('Cache changed during compaction; original path untouched')
    descriptor = {'schema_version': 'ifv-psd-bound-gzip-v1', 'archive': archive.name,
        'identity': original['identity'], 'payload_sha256': original['payload_sha256'],
        'uncompressed_bytes': len(raw), 'uncompressed_sha256': sha}
    atomic_json(cache, descriptor, staging=staging)
    load_bound(cache, identity=identity)
    return {'already_packed': False, 'saved_bytes': len(raw)-archive.stat().st_size-cache.stat().st_size,
            'original_sha256': sha, 'archive': archive.name}


def link_identical(source, target, *, root, staging=None):
    root = root.resolve()
    for p in (source, target):
        if p.is_symlink(): raise ValueError('Refuse symlink trace')
        p.resolve().relative_to(root)
    a, b = signature(source), signature(target)
    if a[:2] == b[:2]: return 0
    if a[0] != b[0] or a[2] != b[2] or digest(source) != digest(target):
        raise ValueError('Only byte-identical, same-filesystem traces may share storage')
    stage = target.parent if staging is None else staging
    stage.mkdir(parents=True, exist_ok=True)
    if stage.stat().st_dev != target.parent.stat().st_dev:
        raise ValueError('Hardlink publication requires same filesystem')
    temporary = stage/('.psd-link-'+uuid.uuid4().hex)
    os.link(source, temporary)
    try:
        if signature(source) != a or signature(target) != b:
            raise ValueError('Trace changed during deduplication')
        reclaimed = target.stat().st_blocks*512 if hasattr(target.stat(), 'st_blocks') and target.stat().st_nlink == 1 else 0
        os.replace(temporary, target)
    finally:
        if temporary.exists(): temporary.unlink()  # Only this exact temporary link.
    if signature(source)[:2] != signature(target)[:2]:
        raise ValueError('Trace hardlink verification failed')
    return reclaimed


def compact_run(run, *, staging=None):
    from contextlib import ExitStack
    from ifv_training.psd_repair_search import search_lock
    from ifv_training.psd_repair_storage import load_bound
    run = run.resolve()
    records = []
    for marker in sorted((run/'episodes/psd-infrastructure-attempts').glob('*/retry-state.json')):
        if json.loads(marker.read_text())['payload']['attempts'][-1]['status'] != 'completed':
            continue
        with ExitStack() as locks:
            try:
                locks.enter_context(search_lock(marker.parent))
            except BlockingIOError:
                # A source reviewer can briefly hold the completed-slot lock.
                # Leave its bytes untouched and reconsider on the next sweep.
                continue
            saved = json.loads(marker.read_text()); identity = saved['identity']
            state = load_bound(marker, identity=identity)
            last = state['attempts'][-1]
            if last['status'] != 'completed': continue
            directory = Path(last['directory']).resolve()
            try: directory.relative_to(run)
            except ValueError: continue  # Never touch a prior run's native source.
            episode = identity['inputs']['episode_id']
            name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in episode)+'.json'
            native, canonical = directory/'traces'/name, run/'episodes/traces'/name
            if not native.is_file() or not canonical.is_file(): continue
            # The active collector has already returned from its locked retry
            # and published this canonical trace; it never rereads completed
            # caches during this one-pass collection. All later PSD readers
            # must use the gzip-aware immutable snapshot attested by the CLI.
            cache = marker.with_name('result.json')
            header = json.loads(cache.read_text())
            if (header.get('schema_version') == 'ifv-psd-bound-gzip-v1'
                    and header.get('identity') == identity and os.path.samefile(native, canonical)):
                records.append({'episode_id': episode, 'already_packed': True,
                    'trace_bytes_saved': 0, 'saved_bytes': 0, 'skipped_prior_completed_pack': True})
                continue
            if load_bound(cache, identity=identity) != json.loads(canonical.read_text()):
                raise ValueError('Canonical result/cache mismatch')
            linked = link_identical(native, canonical, root=run, staging=staging)
            packed = pack_cache(cache, identity=identity, staging=staging)
            records.append({'episode_id': episode, 'trace_bytes_saved': linked, **packed})
    return {'time': time.time(), 'slots_checked': len(records), 'records': records,
            'bytes_reclaimed_estimate': sum(r['trace_bytes_saved']+r['saved_bytes'] for r in records),
            'all_original_cache_bytes_recoverable': True, 'trajectory_bytes_unchanged': True}


def validate_run_scope(run):
    if run.name == 'psd-production400x8-20260917-v3':
        return
    predecessors = {
        'psd-production400x8-20260917-v4': 'psd-production400x8-20260917-v3',
        'psd-production400x8-20260917-v5': 'psd-production400x8-20260917-v4',
        'psd-production400x8-20260917-v6': 'psd-production400x8-20260917-v5',
    }
    if run.name in predecessors:
        binding = json.loads((run/'binding.json').read_text())
        if (binding.get('slots') == 3200 and binding.get('concurrency') == 40
                and Path(binding.get('reuse_run', '')).resolve()
                    == run.parent/predecessors[run.name]):
            return
    raise ValueError('Only the current source run and its attested recovery are in scope')


def main():
    from ifv_training.psd_repair_search import search_lock
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--reader-snapshot', type=Path, required=True)
    p.add_argument('--follow', action='store_true')
    args = p.parse_args()
    root = Path('/volume/ybo/wza')
    run = args.run.resolve(); run.relative_to(root/'runs')
    validate_run_scope(run)
    code = args.reader_snapshot.resolve(); code.relative_to(root/'training-artifacts')
    if json.loads((code.parent/'stage-state.json').read_text())['tests_returncode'] != 0:
        raise ValueError('Gzip-aware reader snapshot not validated')
    import ifv_training.psd_repair_storage as storage
    if Path(storage.__file__).resolve() != code/'training/ifv_training/psd_repair_storage.py':
        raise ValueError('Wrong reader snapshot on PYTHONPATH')
    binding = json.loads((code.parent/'code-binding.json').read_text())
    for relative, sha in binding.items():
        if digest(code/relative) != sha: raise ValueError('Immutable reader snapshot changed')
    with search_lock(run/'storage-compaction-lock'):
        temporary_base = (root/'tmp').resolve()
        temporary_base.relative_to(root)
        staging = Path(tempfile.mkdtemp(prefix='psd-compact-', dir=temporary_base))
        staging.resolve().relative_to(temporary_base)
        atomic_json(run/'storage-reader-requirement.json', {
            'reader_snapshot': str(code), 'minimum_feature': 'ifv-psd-bound-gzip-v1',
            'old_reader_not_restartable': True, 'raw_traces_unchanged': True}, staging=staging)
        while True:
            output = run/'storage-compaction'/f'{time.time_ns()}.json'
            result = compact_run(run, staging=staging)
            atomic_json(output, result, staging=staging)
            summary = {k:v for k,v in result.items() if k!='records'}
            atomic_json(run/'storage-compaction-latest.json', summary, staging=staging)
            print(json.dumps(summary), flush=True)
            state = json.loads((run/'state.json').read_text())
            if not args.follow or state.get('phase') != 'collecting_remaining_3160': break
            time.sleep(300)


if __name__ == '__main__': main()

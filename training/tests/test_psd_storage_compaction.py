import gzip
import hashlib
import json
import os

import pytest

from ifv_training.psd_repair_storage import load_bound, save_bound
from scripts.server.compact_psd_completed_storage import compact_run, pack_cache, link_identical


def test_cache_preserves_original_bytes_and_bound_payload(tmp_path):
    p = tmp_path/'result.json'; identity = {'case': 2}; payload = {'steps': ['same '*1000]*5}
    save_bound(p, identity=identity, payload=payload)
    original = p.read_bytes()
    report = pack_cache(p, identity=identity)
    assert report['saved_bytes'] > 0
    assert load_bound(p, identity=identity) == payload
    descriptor = json.loads(p.read_text())
    with gzip.open(p.parent/descriptor['archive'], 'rb') as f:
        assert f.read() == original
    assert pack_cache(p, identity=identity)['already_packed']
    with pytest.raises(ValueError, match='binding changed'):
        load_bound(p, identity={'case': 3})


@pytest.mark.parametrize('field,value', [('archive','../outside.json.gz'),
    ('uncompressed_bytes',1), ('uncompressed_sha256','bad'), ('payload_sha256','bad')])
def test_packed_cache_tamper_fails_closed(tmp_path, field, value):
    p = tmp_path/'result.json'; save_bound(p, identity={}, payload={'x': [1,2]})
    pack_cache(p, identity={})
    d = json.loads(p.read_text()); d[field] = value; p.write_text(json.dumps(d))
    with pytest.raises(ValueError): load_bound(p, identity={})


def test_hardlink_preserves_both_paths_and_rejects_mismatch(tmp_path):
    a, b = tmp_path/'native.json', tmp_path/'canonical.json'
    a.write_bytes(b'exact original bytes'); b.write_bytes(a.read_bytes())
    before = hashlib.sha256(a.read_bytes()).hexdigest()
    link_identical(a,b,root=tmp_path)
    assert os.path.samefile(a,b)
    assert hashlib.sha256(b.read_bytes()).hexdigest() == before
    assert link_identical(a,b,root=tmp_path) == 0
    c = tmp_path/'different.json'; c.write_bytes(b'not the same')
    with pytest.raises(ValueError): link_identical(a,c,root=tmp_path)
    assert c.read_bytes() == b'not the same'


def make_slot(run, *, status='completed'):
    slot = run/'episodes/psd-infrastructure-attempts/slot'
    directory = slot/'attempt-001'
    native = directory/'traces/case.json';native.parent.mkdir(parents=True)
    payload = {'image_id':'case', 'state': {'all_steps': [{'a':'long '*100}]*2}}
    native.write_text(json.dumps(payload))
    canonical = run/'episodes/traces/case.json';canonical.parent.mkdir(parents=True)
    canonical.write_bytes(native.read_bytes())
    identity = {'inputs': {'episode_id':'case'}, 'max_attempts':3}
    save_bound(slot/'retry-state.json', identity=identity,
               payload={'attempts':[{'directory':str(directory),'status':status}]})
    save_bound(slot/'result.json',identity=identity,payload=payload)
    return slot,native,canonical,identity,payload


def test_completed_slot_compaction_keeps_ledger_and_native_content(tmp_path):
    slot,native,canonical,identity,payload = make_slot(tmp_path)
    before = (slot/'retry-state.json').read_bytes(); raw = native.read_bytes()
    result = compact_run(tmp_path)
    assert result['slots_checked'] == 1
    assert (slot/'retry-state.json').read_bytes() == before
    assert canonical.read_bytes() == native.read_bytes() == raw
    assert os.path.samefile(native,canonical)
    assert load_bound(slot/'result.json',identity=identity) == payload


def test_running_slot_is_never_rewritten(tmp_path):
    slot,native,canonical,identity,payload = make_slot(tmp_path,status='running')
    before = (slot/'result.json').read_bytes()
    assert compact_run(tmp_path)['slots_checked'] == 0
    assert (slot/'result.json').read_bytes() == before
    assert not os.path.samefile(native,canonical)


def test_busy_completed_source_is_skipped_without_stopping_compactor(tmp_path, monkeypatch):
    import contextlib
    import ifv_training.psd_repair_search as locking
    slot,native,canonical,identity,payload = make_slot(tmp_path)
    before = (slot/'result.json').read_bytes()
    original = locking.search_lock
    @contextlib.contextmanager
    def busy(root):
        raise BlockingIOError('synthetic active source reviewer')
        yield
    monkeypatch.setattr(locking, 'search_lock', busy)
    assert compact_run(tmp_path)['slots_checked'] == 0
    assert (slot/'result.json').read_bytes() == before and not os.path.samefile(native, canonical)
    monkeypatch.setattr(locking, 'search_lock', original)
    assert compact_run(tmp_path)['slots_checked'] == 1
    assert load_bound(slot/'result.json', identity=identity) == payload


def test_retry_reuses_packed_result_without_new_model_call(tmp_path):
    import asyncio
    from ifv_training.psd_infrastructure_retry import VERSION, retry_episode
    identity = {'version': VERSION, 'inputs': {'seed':1}, 'max_attempts':3}
    payload = {'verdict':'fake', 'ordinary_model_error_retained':True}
    save_bound(tmp_path/'retry-state.json', identity=identity,
               payload={'attempts':[{'index':1,'status':'completed'}]})
    save_bound(tmp_path/'result.json', identity=identity, payload=payload)
    pack_cache(tmp_path/'result.json', identity=identity)
    async def never_generate(directory):
        raise AssertionError('A completed packed result must not be resampled')
    assert asyncio.run(retry_episode(root=tmp_path,identity=identity['inputs'],generate=never_generate)) == payload


def test_temporary_publication_stays_outside_scanned_run(tmp_path,monkeypatch):
    run = tmp_path/'run';run.mkdir()
    staging = tmp_path/'staging';staging.mkdir()
    slot,native,canonical,identity,payload = make_slot(run)
    original = os.replace; seen = []
    def checked_replace(source,target):
        from pathlib import Path
        assert Path(source).parent == staging
        seen.append(Path(target).name)
        return original(source,target)
    monkeypatch.setattr(os,'replace',checked_replace)
    compact_run(run,staging=staging)
    assert 'case.json' in seen and 'result.json' in seen
    assert any(p.endswith('.json.gz') for p in seen)
    assert list(staging.iterdir()) == []
    assert load_bound(slot/'result.json',identity=identity) == payload

import json
from pathlib import Path

import pytest

from scripts.server.run_sft3084_eval import digest, successful_traces, verify_acceptance


def trace(directory: Path, case: str, **values):
    target = directory / 'traces' / f'{case}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(case_id=case, verdict='real', fact_check_report={'ok': True}, **values)))
    return target


def test_successes_preserve_valid_cases_and_reject_resampling(tmp_path):
    first, second = tmp_path / 'a', tmp_path / 'b'
    trace(first, 'one')
    trace(first, 'two', error='timeout')
    trace(second, 'two')
    assert set(successful_traces([first, second], {'one', 'two'})) == {'one', 'two'}
    trace(second, 'one')
    with pytest.raises(ValueError, match='resampled'):
        successful_traces([first, second], {'one', 'two'})
    with pytest.raises(ValueError, match='Unknown'):
        successful_traces([first], {'one'})


def test_smoke_acceptance_is_bound_to_export_and_every_trace(tmp_path):
    source = trace(tmp_path / 'smoke', 'one')
    acceptance = tmp_path / 'acceptance.json'
    acceptance.write_text(json.dumps(dict(passed=True, export_sha256='export',
        strict_audit_passed=True, external_tools_verified=True, multimodal_verified=True,
        trace_sha256={'one': digest(source)})))
    verify_acceptance(acceptance, tmp_path / 'smoke', {'one'}, 'export')
    with pytest.raises(ValueError, match='bound'):
        verify_acceptance(acceptance, tmp_path / 'smoke', {'one'}, 'different export')
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='bound'):
        verify_acceptance(acceptance, tmp_path / 'smoke', {'one'}, 'export')

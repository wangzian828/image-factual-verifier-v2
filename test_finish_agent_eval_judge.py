import json
from pathlib import Path

import pytest

from scripts.finish_agent_eval_judge import merge_runs, save_rows, terminal


def fixture_run(path, statuses):
    path.mkdir()
    records = []
    for case, status in statuses:
        (path / (case + '.json')).write_text('{}')
        records.append({'case_id': case, 'status': status, 'trace_path': case + '.json'})
    (path / 'run_results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    return path


def test_only_engineering_failure_replaced(tmp_path):
    initial = fixture_run(tmp_path / 'initial', [('a', 'success'), ('b', 'error')])
    retry = fixture_run(tmp_path / 'retry', [('b', 'success')])
    merged, ledger = merge_runs([initial, retry], ['a', 'b'])
    assert len(merged) == 2 and len(ledger) == 1
    assert Path(merged[1]['source_trace_path']).parent == retry
    assert ledger[0]['previous_status'] == 'error'


def test_no_resampling_success(tmp_path):
    initial = fixture_run(tmp_path / 'initial', [('a', 'success')])
    retry = fixture_run(tmp_path / 'retry', [('a', 'success')])
    with pytest.raises(ValueError, match='engineering failure'):
        merge_runs([initial, retry], ['a'])


@pytest.mark.parametrize('records,expected', [([('a','success')], ['a','b']),
    ([('a','error')], ['a']), ([('a','success'),('a','success')], ['a'])])
def test_bad_coverage_or_failure(tmp_path, records, expected):
    with pytest.raises(ValueError):
        merge_runs([fixture_run(tmp_path / 'run', records)], expected)


def test_frozen_ledger(tmp_path):
    path = tmp_path / 'frozen.jsonl'
    save_rows(path, [{'x': 1}])
    save_rows(path, [{'x': 1}])
    with pytest.raises(ValueError):
        save_rows(path, [{'x': 2}])


def test_negative_judge_is_terminal_but_invalid_json_is_not():
    record = {'status':'completed','quality_bucket':'rejected',
              'fact_alignment':'different_fact','reason_quality':'unsupported'}
    assert terminal(record)
    assert not terminal({**record, 'judge_json_parse_error':'truncated'})
    assert not terminal({**record, 'quality_bucket':None})
    assert terminal({'status':'not_auditable'})

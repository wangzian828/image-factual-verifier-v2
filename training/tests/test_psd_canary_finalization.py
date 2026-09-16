import pytest

from ifv_training.io import write_json, write_jsonl, sha256_file
from ifv_training.psd_repair_storage import save_bound
from scripts.server.finalize_psd_repair_canary import CASES, completed_inputs


def prepared(tmp_path, status='converged', accepted=1):
    write_json(tmp_path / 'execution-recovery-v1/result.json', {
        'results': [{'case': key, 'result': {'status': status}} for key in CASES]})
    for key in CASES:
        case = tmp_path / 'search/repairs' / key
        write_json(case / 'run-inputs.json', {})
        write_jsonl(case / 'repair_candidates.jsonl', [])
        write_jsonl(case / 'repair_attempts.jsonl', [])
        write_json(case / 'manifest.json', {
            'status': status, 'complete_reruns': 1, 'accepted_count': accepted})
        files = {str(case / name): sha256_file(case / name) for name in (
            'manifest.json', 'repair_candidates.jsonl', 'repair_attempts.jsonl')}
        save_bound(case / 'slate-state.json', identity={'version': 'test'}, payload={
            'status': status, 'rounds': [{'files': {}}], 'output_files': files})
    return tmp_path


def test_finalization_is_read_only_and_requires_verified_repairs(tmp_path):
    run = prepared(tmp_path)
    before = {p: sha256_file(p) for p in run.rglob('*') if p.is_file()}
    binding = completed_inputs(run)
    assert binding['accepted_targets'] == 3
    assert binding['new_agent_generations_allowed'] is False
    assert all(sha256_file(p) == digest for p, digest in before.items())


@pytest.mark.parametrize('status', ['repairing', 'paused_unresolved_task_review'])
def test_finalization_cannot_resume_pending_generations(tmp_path, status):
    with pytest.raises(ValueError):
        completed_inputs(prepared(tmp_path, status=status))


def test_finalization_rejects_preservation_only(tmp_path):
    with pytest.raises(ValueError, match='No verified repair'):
        completed_inputs(prepared(tmp_path, status='attempt_budget_exhausted', accepted=0))


def test_finalization_rejects_unresolved_transport_failure(tmp_path):
    run = prepared(tmp_path)
    write_json(run / 'execution-recovery-v1/result.json', {'results': [
        {'case': key, 'error_type': 'ReadTimeout'} for key in CASES]})
    with pytest.raises(ValueError, match='Engineering failures'):
        completed_inputs(run)


def test_finalization_rejects_changed_completed_artifact(tmp_path):
    run = prepared(tmp_path)
    write_jsonl(run / 'search/repairs' / CASES[0] / 'repair_attempts.jsonl', [{'changed': True}])
    with pytest.raises(ValueError, match='artifact changed'):
        completed_inputs(run)

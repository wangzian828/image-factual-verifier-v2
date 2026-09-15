import pytest
from scripts.server.submit_sft2056_judge import join_cases, check_intent


def test_subset_join_does_not_zip_shift_after_missing_case():
    cases = [{'case_id': x} for x in ['a', 'b', 'c', 'd']]
    candidates = [{'case_id': x} for x in ['d', 'a', 'c']]
    joined = join_cases(cases, candidates)
    assert [r[0] for r in joined] == [4, 1, 3]
    assert all(case['case_id'] == candidate['case_id'] for _, case, candidate in joined)


@pytest.mark.parametrize('cases,candidates', [
    ([{'case_id': 'a'}]*2, [{'case_id': 'a'}]),
    ([{'case_id': 'a'}], [{'case_id': 'a'}]*2),
    ([{'case_id': 'a'}], [{'case_id': 'b'}]),
])
def test_join_rejects_ambiguous_ids(cases, candidates):
    with pytest.raises(ValueError):
        join_cases(cases, candidates)


def test_first_submission_and_explicit_quota_resume():
    assert check_intent(None, 'hash', 10000) == 1
    prior = {'fingerprint': 'hash', 'state': 'server_rejected_429', 'attempt': 1, 'time': 100}
    assert check_intent(prior, 'hash', 4000) == 2


@pytest.mark.parametrize('change', [
    {'fingerprint': 'different'}, {'state': 'create_inflight'}, {'state': 'accepted'},
    {'state': 'ambiguous_or_rejected'}, {'attempt': 6}, {'time': 9999},
])
def test_no_replay_on_uncertain_or_exhausted_receipt(change):
    prior = {'fingerprint': 'hash', 'state': 'server_rejected_429', 'attempt': 1, 'time': 100}
    prior.update(change)
    with pytest.raises(ValueError):
        check_intent(prior, 'hash', 10000)

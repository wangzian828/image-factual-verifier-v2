import pytest
from scripts.server.run_psd_grouped_canary import select_fixed_cases, source_revision


def rows():
    return [{'case_id': str(i), 'split': 'train'} for i in range(32)]


def test_selection_preserves_frozen_order_without_outcome_input():
    canary = rows()[::-1]
    assert select_fixed_cases(canary, rows(), rows()) == ['31', '30', '29', '28']


def test_source_revision_is_stable_and_explicitly_not_a_commit():
    revision = source_revision({'a': '123', 'b': '456'})
    assert revision.startswith('source-sha256:')
    assert revision == source_revision({'b': '456', 'a': '123'})
    assert revision != source_revision({'a': '123', 'b': 'changed'})


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'not_train', 'short'])
def test_rejects_changed_or_nontraining_membership(kind):
    canary, public, splits = rows(), rows(), rows()
    if kind == 'duplicate':
        public.append(public[0])
    elif kind == 'missing':
        public.pop(0)
    elif kind == 'not_train':
        splits[0]['split'] = 'test'
    else:
        canary.pop()
    with pytest.raises(ValueError):
        select_fixed_cases(canary, public, splits)

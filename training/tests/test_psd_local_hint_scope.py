import copy
from types import SimpleNamespace

import pytest

from ifv_training.psd_repair_runtime import QwenContinuationAdapter
from src.orchestrator.react_runtime import MAX_REACT_ACTIONS


def test_only_injected_hint_is_removed_and_observed_identical_text_is_kept():
    prefix = [{"role": "user", "content": "image"}, {"role": "tool", "content": "advice"}]
    suffix = [{"role": "assistant", "content": "advice"}, {"role": "tool", "content": "response"}]
    history = prefix + [{"role": "user", "content": "advice"}] + suffix
    original = copy.deepcopy(history)
    folded = QwenContinuationAdapter._remove_local_hint(history, unhinted_prefix=prefix, hint="advice")
    assert folded == prefix + suffix
    assert history == original
    folded[0]["content"] = "changed"
    assert history == original


@pytest.mark.parametrize("mutation", ["prefix", "hint", "missing"])
def test_hint_fold_rejects_unbound_history(mutation):
    prefix = [{"role": "user", "content": "image"}]
    history = prefix + [{"role": "user", "content": "advice"}]
    history = copy.deepcopy(history)
    if mutation == "prefix":
        history[0]["content"] = "other image"
    elif mutation == "hint":
        history[1]["content"] = "other advice"
    else:
        history.pop()
    with pytest.raises(ValueError):
        QwenContinuationAdapter._remove_local_hint(history, unhinted_prefix=prefix, hint="advice")


def test_default_suffix_keeps_the_existing_agent_budget_not_eight_actions():
    state = SimpleNamespace(action_count=3)
    assert QwenContinuationAdapter._remaining_actions(state, None) == MAX_REACT_ACTIONS - 3
    assert QwenContinuationAdapter._remaining_actions(state, 100) == MAX_REACT_ACTIONS - 3
    assert QwenContinuationAdapter._remaining_actions(state, 2) == 2
    assert QwenContinuationAdapter._remaining_actions(SimpleNamespace(action_count=MAX_REACT_ACTIONS), None) == 0
    for value in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            QwenContinuationAdapter._remaining_actions(state, value)

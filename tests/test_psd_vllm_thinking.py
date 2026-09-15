"""CPU tensor checks in the installed vLLM environment; no model weights loaded."""
from __future__ import annotations

import os
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip('torch')
vllm = pytest.importorskip('vllm')
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality
from scripts.server.psd_qwen_thinking import PSDThinkingBudget


@pytest.fixture
def plugin():
    model = os.environ.get('IFV_PSD_TEST_MODEL_DIR')
    if not model:
        pytest.skip('Set IFV_PSD_TEST_MODEL_DIR to the real frozen Qwen tokenizer directory')
    assert vllm.__version__ == '0.18.1'
    return PSDThinkingBudget(NS(speculative_config=None, model_config=NS(tokenizer=model)),
                             torch.device('cpu'), False)


@pytest.mark.parametrize('budget', [1, 3, 8192])
def test_real_tokenizer_boundary_does_not_force_eos_or_touch_other_rows(plugin, budget):
    output = [10] * budget
    params = SamplingParams(max_tokens=budget + 16, extra_args={'ifv_thinking_budget': budget})
    plugin.update_state(BatchUpdate(2, [], [(0, params, plugin.enabled_suffix, output)], []))
    logits = torch.randn(2, plugin.end_token + 5)
    untouched = logits[1].clone()
    original = plugin.apply(logits)
    assert original is logits
    assert torch.isfinite(logits[0]).sum().item() == 1
    assert logits[0].argmax().item() == plugin.end_token
    assert torch.equal(logits[1], untouched)
    output.extend([plugin.end_token, 31, 32])
    plugin.update_state(None)
    fresh = torch.randn_like(logits)
    before = fresh.clone()
    assert torch.equal(plugin.apply(fresh), before)
    assert not plugin.is_argmax_invariant()


def test_actual_batch_swap_and_replacement(plugin):
    params = SamplingParams(max_tokens=100, extra_args={'ifv_thinking_budget': 3})
    output = [10, 11, 12]
    plugin.update_state(BatchUpdate(2, [], [(0, params, plugin.enabled_suffix, output)],
                                   [(0, 1, MoveDirectionality.SWAP)]))
    assert set(plugin.slots) == {1}
    plugin.update_state(BatchUpdate(2, [1], [(1, SamplingParams(max_tokens=100), [], [])], []))
    assert plugin.slots == {}


def test_prompt_contract_and_disabled_thinking(plugin):
    params = SamplingParams(max_tokens=100, extra_args={'ifv_thinking_budget': 3})
    assert plugin._new_state(params, plugin.disabled_suffix, []) is None
    for prompt in [None, [1, 2, 3]]:
        with pytest.raises(ValueError):
            plugin._new_state(params, prompt, [])


def test_request_extra_args_reach_real_sampling_params(plugin):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    request = ChatCompletionRequest(model='qwen', messages=[{'role': 'user', 'content': 'test'}],
                                    max_tokens=100, vllm_xargs={'ifv_thinking_budget': 3})
    params = request.to_sampling_params(100, {})
    assert params.extra_args['ifv_thinking_budget'] == 3
    plugin.validate_params(params)

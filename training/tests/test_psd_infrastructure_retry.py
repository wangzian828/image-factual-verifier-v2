import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from ifv_training.psd_infrastructure_retry import (
    InfrastructureRetriesExhausted, PolicyInfrastructureFailure, guard_policy_backend,
    retry_episode, validate_generated_probabilities, is_nonfinite_serialization_response)


def response(value=-.5, candidate=-1.):
    return SimpleNamespace(raw={"choices": [{"logprobs": {"content": [
        {"logprob": value, "top_logprobs": [{"logprob": candidate}]}]}}]})


async def no_sleep(_):
    pass


def nan_error():
    return {'error': {'type': 'BadRequestError', 'code': 400,
        'message': 'Out of range float values are not JSON compliant: nan'}}


def test_exact_provider_nan_400_is_not_an_ordinary_invalid_request():
    assert is_nonfinite_serialization_response(400, nan_error())
    for status, payload in [(422, nan_error()), (400, {'error': 'nan'}),
            (400, {'error': {**nan_error()['error'], 'code': '400'}}),
            (400, {'error': {**nan_error()['error'], 'message': 'invalid image'}}),
            (400, {'error': {**nan_error()['error'], 'type': 'UserToolError'}})]:
        assert not is_nonfinite_serialization_response(status, payload)


def test_native_wrapped_nan_http400_is_retryable():
    async def call(*args, **kwargs):
        request = httpx.Request('POST', 'http://policy/v1/chat/completions')
        try:
            httpx.Response(400, json=nan_error(), request=request).raise_for_status()
        except httpx.HTTPStatusError as error:
            raise RuntimeError('native wrapper') from error
    backend = SimpleNamespace(get_response=call, max_retries=0)
    guard_policy_backend(backend)
    with pytest.raises(PolicyInfrastructureFailure, match='400_nonfinite'):
        asyncio.run(backend.get_response([]))


@pytest.mark.parametrize('mode', ['nonfinite_400', 'nan_before_parser', 'ordinary_empty'])
def test_http_boundary_checks_before_unusable_response_parser(mode):
    payload = nan_error() if mode == 'nonfinite_400' else response(
        value=float('nan') if mode == 'nan_before_parser' else -.3).raw
    request = httpx.Request('POST', 'http://policy/v1/chat/completions')
    async def post(*args, **kwargs):
        return httpx.Response(400 if mode == 'nonfinite_400' else 200,
            content=json.dumps(payload).encode(), request=request)
    backend = SimpleNamespace(max_retries=0, _get_shared_client=lambda: SimpleNamespace(post=post))
    async def call(*args, **kwargs):
        await backend._get_shared_client().post('http://policy/v1/chat/completions')
        raise RuntimeError('native unusable response parser')
    backend.get_response = call
    guard_policy_backend(backend)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(backend.get_response([]))
    assert isinstance(caught.value, PolicyInfrastructureFailure) == (mode != 'ordinary_empty')


@pytest.mark.parametrize("value", [None, float('nan'), float('inf'), -float('inf'), 'NaN'])
def test_nonfinite_selected_tokens_never_execute(value):
    with pytest.raises(PolicyInfrastructureFailure):
        validate_generated_probabilities(response(value).raw)


def test_topk_negative_infinity_is_a_legal_grammar_mask():
    validate_generated_probabilities(response(candidate=-float('inf')).raw)
    for value in [None, float('inf'), float('nan')]:
        with pytest.raises(PolicyInfrastructureFailure):
            validate_generated_probabilities(response(candidate=value).raw)
    validate_generated_probabilities({'choices': [{'message': {'content': 'NaN ! ! !'}}]})


@pytest.mark.parametrize('status', [408, 429, 500, 502, 503, 504, 400, 401, 403, 404, 422, 507])
def test_only_allowlisted_model_http_failures_retry(status):
    async def call(*args, **kwargs):
        request = httpx.Request('POST', 'http://policy/v1/chat/completions')
        try:
            httpx.Response(status, request=request).raise_for_status()
        except httpx.HTTPStatusError as error:
            raise RuntimeError('native API wrapper') from error
    backend = SimpleNamespace(get_response=call, max_retries=0)
    guard_policy_backend(backend)
    expected = PolicyInfrastructureFailure if status in {408, 429, 500, 502, 503, 504} else RuntimeError
    with pytest.raises(expected) as caught:
        asyncio.run(backend.get_response([]))
    if status in {400, 401, 403, 404, 422, 507}:
        assert not isinstance(caught.value, PolicyInfrastructureFailure)


def test_boundary_guard_is_idempotent_and_does_not_change_request():
    calls = []
    async def call(*args, **kwargs):
        calls.append((args, kwargs))
        return response()
    backend = SimpleNamespace(get_response=call, max_retries=0)
    guard_policy_backend(backend)
    wrapped = backend.get_response
    guard_policy_backend(backend)
    assert backend.get_response is wrapped
    asyncio.run(backend.get_response(['image'], seed=123, temperature=.7))
    assert calls == [((['image'],), {'seed': 123, 'temperature': .7})]
    with pytest.raises(ValueError, match='request retries=0'):
        guard_policy_backend(SimpleNamespace(get_response=call, max_retries=2))


def test_retry_uses_fresh_directories_and_keeps_ordinary_wrong_answer(tmp_path):
    calls, delays = [], []
    async def generate(directory):
        calls.append(directory)
        if len(calls) == 1:
            raise PolicyInfrastructureFailure('model_nonfinite_selected_logprob')
        return {'verdict': 'fake', 'termination': 'success', 'correct': False}
    async def sleep(delay):
        delays.append(delay)
    kwargs = dict(root=tmp_path, identity={'seed': 17, 'slot': 3}, generate=generate, sleep=sleep)
    result = asyncio.run(retry_episode(**kwargs))
    assert not result['correct'] and len(calls) == 2 and delays == [5]
    assert calls[0] != calls[1]
    assert asyncio.run(retry_episode(**kwargs)) == result and len(calls) == 2
    saved = json.loads((tmp_path/'retry-state.json').read_text())['payload']['attempts']
    assert [r['status'] for r in saved] == ['infrastructure_failed', 'completed']
    with pytest.raises(ValueError, match='binding changed'):
        asyncio.run(retry_episode(**{**kwargs, 'identity': {'seed': 18}}))


def test_budget_persists_and_exhausted_slot_is_never_accepted(tmp_path):
    calls = []
    async def generate(directory):
        calls.append(directory)
        raise PolicyInfrastructureFailure('model_http_504')
    for _ in range(2):
        with pytest.raises(InfrastructureRetriesExhausted):
            asyncio.run(retry_episode(root=tmp_path, identity={}, generate=generate, sleep=no_sleep))
    assert len(calls) == 3
    assert not (tmp_path/'result.json').exists()


@pytest.mark.parametrize('error', [RuntimeError('bad tool syntax'), ValueError('wrong verdict'),
    OSError('disk full'), httpx.ConnectError('tool service unavailable'), asyncio.CancelledError()])
def test_unmarked_errors_and_cancellation_are_not_retried(tmp_path, error):
    calls = []
    async def generate(directory):
        calls.append(directory)
        raise error
    with pytest.raises(type(error)):
        asyncio.run(retry_episode(root=tmp_path, identity={}, generate=generate, sleep=no_sleep))
    with pytest.raises(RuntimeError, match='unresolved'):
        asyncio.run(retry_episode(root=tmp_path, identity={}, generate=generate, sleep=no_sleep))
    assert len(calls) == 1


def test_parallel_same_slot_cannot_double_dispatch(tmp_path):
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()
        async def generate(directory):
            entered.set()
            await release.wait()
            return {'ok': True}
        first = asyncio.create_task(retry_episode(root=tmp_path, identity={}, generate=generate))
        await entered.wait()
        with pytest.raises(OSError):
            await retry_episode(root=tmp_path, identity={}, generate=generate)
        release.set()
        await first
    asyncio.run(exercise())


@pytest.mark.parametrize('mode', ['recover', 'exhaust', 'format_failure'])
def test_collector_restarts_full_episode_without_touching_native_agent(monkeypatch, tmp_path, mode):
    from scripts.collect_psd_rollouts import PSDWorkflow
    from src.workflow import VerificationWorkflow
    import ifv_training.psd_infrastructure_retry as recovery
    original_retry = recovery.retry_episode
    async def fast_retry(**kwargs):
        return await original_retry(**kwargs, sleep=no_sleep)
    monkeypatch.setattr(recovery, 'retry_episode', fast_retry)
    image = tmp_path/'image.png'
    image.write_bytes(b'fixed input, mock does not decode it')
    calls, closed = [], []
    async def native_run(self, image_path, image_id, **kwargs):
        calls.append((self, self.config.output_dir, self.config.sampling_seed, self.config.resume_from))
        assert image_path == str(image) and image_id == 'case--r003'
        if mode == 'exhaust' or (mode == 'recover' and len(calls) == 1):
            raise PolicyInfrastructureFailure('model_http_503')
        result = {'image_id': image_id, 'termination': 'success', 'verdict': 'fake'}
        if mode == 'format_failure':
            error = RuntimeError('Final report contract failed')
            error._ifv_result = {**result, 'termination': 'error', 'error': str(error)}
            raise error
        return result
    async def close(self):
        closed.append(self)
    monkeypatch.setattr(VerificationWorkflow, 'run_single', native_run)
    monkeypatch.setattr(VerificationWorkflow, 'aclose', close)
    config = SimpleNamespace(output_dir=str(tmp_path/'run/traces'), sampling_seed=71,
        model_name='sft', provider='qwen_local', llm_base_url='http://policy', timeout=3000,
        resume_from='old_context_must_not_be_used')
    workflow = PSDWorkflow(config)
    result = asyncio.run(workflow.run_single(str(image), 'case--r003'))
    assert len(calls) == {'recover': 2, 'exhaust': 3, 'format_failure': 1}[mode]
    assert len(closed) == len(calls) and len({id(row[0]) for row in calls}) == len(calls)
    assert all(row[2:] == (71, None) for row in calls)
    assert len({row[1] for row in calls}) == len(calls)
    assert config.output_dir == str(tmp_path/'run/traces')
    canonical = tmp_path/'run/traces/case--r003.json'
    assert canonical.exists() == (mode != 'exhaust')
    if mode == 'exhaust':
        assert result['psd_infrastructure_pending']
    elif mode == 'format_failure':
        assert result['termination'] == 'error'


def test_collection_cannot_pass_with_a_quarantined_slot():
    from ifv_training.psd_collection import verify_collection
    from test_psd_collection import collection
    rows, manifest = collection()
    manifest['agent']['psd_sampling']['infrastructure_retry'] = {
        'slots': 16, 'unresolved': ['one_bad_slot'], 'selection_by_answer': False}
    with pytest.raises(ValueError, match='unresolved'):
        verify_collection(rows, case_ids=['a', 'b'], manifest=manifest, expected_rollouts=8)

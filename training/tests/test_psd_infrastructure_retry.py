import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from ifv_training.psd_infrastructure_retry import (
    InfrastructureRetriesExhausted, PolicyInfrastructureFailure, guard_policy_backend,
    retry_episode, retry_episode_compact, validate_generated_probabilities,
    is_nonfinite_serialization_response, unusable_chat_response_reason)


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


def test_request_local_retry_preserves_messages_and_does_not_restart_episode(monkeypatch):
    calls = []
    async def call(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise PolicyInfrastructureFailure('model_http_400_nonfinite_serialization')
        return response()
    async def no_sleep(_):
        pass
    monkeypatch.setattr('ifv_training.psd_infrastructure_retry.asyncio.sleep', no_sleep)
    backend = SimpleNamespace(get_response=call, max_retries=0)
    guard_policy_backend(backend, max_request_retries=2)
    result = asyncio.run(backend.get_response(['same-prefix'], seed=23))
    assert result is not None
    assert calls == [((['same-prefix'],), {'seed': 23})] * 2
    guard_policy_backend(backend, max_request_retries=2)
    with pytest.raises(ValueError, match='policy changed'):
        guard_policy_backend(backend, max_request_retries=1)


def test_request_local_retry_catches_exact_http_nan_before_native_parser(monkeypatch):
    calls = []
    request = httpx.Request('POST', 'http://policy/v1/chat/completions')
    async def post(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            return httpx.Response(400, json=nan_error(), request=request)
        return httpx.Response(200, json=response().raw, request=request)
    async def call(*args, **kwargs):
        reply = await backend._get_shared_client().post('http://policy/v1/chat/completions')
        reply.raise_for_status()
        return SimpleNamespace(raw=reply.json())
    async def no_sleep(_):
        pass
    monkeypatch.setattr('ifv_training.psd_infrastructure_retry.asyncio.sleep', no_sleep)
    backend = SimpleNamespace(get_response=call, max_retries=0,
        _get_shared_client=lambda: SimpleNamespace(post=post))
    guard_policy_backend(backend, max_request_retries=2)
    assert asyncio.run(backend.get_response(['same-prefix'])).raw == response().raw
    assert len(calls) == 2 and calls[0] == calls[1]


def test_request_local_retry_exhausts_without_accepting_invalid_output(monkeypatch):
    calls = []
    async def call(*args, **kwargs):
        calls.append(1)
        return response(value=float('nan'))
    async def no_sleep(_):
        pass
    monkeypatch.setattr('ifv_training.psd_infrastructure_retry.asyncio.sleep', no_sleep)
    backend = SimpleNamespace(get_response=call, max_retries=0)
    guard_policy_backend(backend, max_request_retries=2)
    with pytest.raises(PolicyInfrastructureFailure, match='model_nonfinite_selected_logprob'):
        asyncio.run(backend.get_response([]))
    assert len(calls) == 3


def test_request_local_retry_does_not_retry_non_infrastructure_error(monkeypatch):
    calls = []
    async def call(*args, **kwargs):
        calls.append(1)
        raise ValueError('bad bound prefix')
    async def no_sleep(_):
        raise AssertionError('must not sleep')
    monkeypatch.setattr('ifv_training.psd_infrastructure_retry.asyncio.sleep', no_sleep)
    backend = SimpleNamespace(get_response=call, max_retries=0)
    guard_policy_backend(backend, max_request_retries=2)
    with pytest.raises(ValueError, match='bad bound prefix'):
        asyncio.run(backend.get_response([]))
    assert len(calls) == 1


def test_aborted_empty_choice_is_infrastructure_not_a_completed_action(monkeypatch):
    message = ('Chat Completions returned an unusable response: choices=1, '
        'finish_reason=abort, content_chars=0, reasoning_chars=0, '
        'reasoning_fallback_requested=False')
    assert unusable_chat_response_reason(RuntimeError(message)) == 'model_unusable_http_choice'
    calls = []
    async def call(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError(message)
        return response()
    async def no_sleep(_):
        pass
    monkeypatch.setattr('ifv_training.psd_infrastructure_retry.asyncio.sleep', no_sleep)
    backend = SimpleNamespace(get_response=call, max_retries=0)
    guard_policy_backend(backend, max_request_retries=2)
    assert asyncio.run(backend.get_response([])).raw == response().raw
    assert len(calls) == 2


@pytest.mark.parametrize('limit', [-1, 3, True, 1.5])
def test_request_local_retry_limit_is_bounded(limit):
    backend = SimpleNamespace(get_response=lambda: None, max_retries=0)
    with pytest.raises(ValueError, match='0..2'):
        guard_policy_backend(backend, max_request_retries=limit)


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
    OSError('disk full'), asyncio.CancelledError()])
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


@pytest.mark.parametrize('error', [httpx.ConnectError('policy service unavailable'),
    httpx.ReadError('policy connection reset'), httpx.ConnectTimeout('policy connect timeout'),
    httpx.ReadTimeout('policy read timeout')])
def test_direct_transport_errors_use_the_existing_infrastructure_budget(tmp_path, error):
    calls = []
    async def generate(directory):
        calls.append(directory)
        if len(calls) == 1:
            raise error
        return {'termination': 'success'}
    result = asyncio.run(retry_episode(root=tmp_path, identity={}, generate=generate, sleep=no_sleep))
    assert result == {'termination': 'success'} and len(calls) == 2
    attempts = json.loads((tmp_path/'retry-state.json').read_text())['payload']['attempts']
    assert [row['status'] for row in attempts] == ['infrastructure_failed', 'completed']
    assert attempts[0]['error_type'] == type(error).__name__


def test_unusable_backend_choice_consumes_bounded_attempt_not_new_budget(tmp_path):
    calls = []
    async def generate(directory):
        calls.append(directory)
        if len(calls) == 1:
            raise RuntimeError('Chat Completions returned an unusable response: '
                'choices=1, finish_reason=length, content_chars=0, '
                'reasoning_chars=514, reasoning_fallback_requested=False')
        return {'termination': 'success'}
    result = asyncio.run(retry_episode(root=tmp_path, identity={}, generate=generate, sleep=no_sleep))
    assert result == {'termination': 'success'} and len(calls) == 2
    attempts = json.loads((tmp_path/'retry-state.json').read_text())['payload']['attempts']
    assert [(row['status'], row.get('reason')) for row in attempts] == [
        ('infrastructure_failed', 'model_unusable_http_choice'), ('completed', None)]


def test_missing_final_judgment_uses_episode_retry_budget(tmp_path):
    calls = []
    async def generate(directory):
        calls.append(directory)
        if len(calls) == 1:
            raise RuntimeError('teacher continuation did not reach final Judgment')
        return {'termination': 'success'}

    result = asyncio.run(retry_episode(root=tmp_path, identity={}, generate=generate,
                                       sleep=no_sleep))
    assert result == {'termination': 'success'} and len(calls) == 2
    attempts = json.loads((tmp_path/'retry-state.json').read_text())['payload']['attempts']
    assert [(row['status'], row.get('reason')) for row in attempts] == [
        ('infrastructure_failed', 'model_missing_final_judgment'), ('completed', None)]


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


def test_compact_retry_promotes_one_trace_without_result_copy(tmp_path):
    root, canonical = tmp_path / "slot", tmp_path / "run/traces/episode.json"
    calls = []

    async def generate(directory):
        calls.append(directory)
        trace = directory / "traces/episode.json"
        trace.parent.mkdir()
        payload = {"image_id": "episode", "termination": "success"}
        trace.write_text(json.dumps(payload))
        return payload, trace

    kwargs = dict(root=root, identity={"episode_id": "episode"},
        canonical_path=canonical, generate=generate, sleep=no_sleep)
    assert asyncio.run(retry_episode_compact(**kwargs))["termination"] == "success"
    assert canonical.is_file() and not (root / "result.json").exists()
    assert not (calls[0] / "traces/episode.json").exists()
    assert asyncio.run(retry_episode_compact(**kwargs))["image_id"] == "episode"
    assert len(calls) == 1


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
        from test_psd_source_completion import complete_trace
        result = {**complete_trace(), 'image_id': image_id}
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
    assert len(calls) == {'recover': 2, 'exhaust': 3, 'format_failure': 3}[mode]
    assert len(closed) == len(calls) and len({id(row[0]) for row in calls}) == len(calls)
    assert all(row[2:] == (71, None) for row in calls)
    assert len({row[1] for row in calls}) == len(calls)
    assert config.output_dir == str(tmp_path/'run/traces')
    canonical = tmp_path/'run/traces/case--r003.json'
    assert canonical.exists() == (mode == 'recover')
    if mode in ('exhaust', 'format_failure'):
        assert result['psd_infrastructure_pending']


def test_collection_cannot_pass_with_a_quarantined_slot():
    from ifv_training.psd_collection import verify_collection
    from test_psd_collection import collection
    rows, manifest = collection()
    manifest['agent']['psd_sampling']['infrastructure_retry'] = {
        'slots': 16, 'unresolved': ['one_bad_slot'], 'selection_by_answer': False}
    with pytest.raises(ValueError, match='unresolved'):
        verify_collection(rows, case_ids=['a', 'b'], manifest=manifest, expected_rollouts=8)

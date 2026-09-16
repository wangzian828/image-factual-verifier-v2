"""Regression cases found by the full PSD contract audit, not quality scores."""
import asyncio
import copy
from types import SimpleNamespace

import pytest

from ifv_training.psd_gemini_judge import trace_steps
from ifv_training.psd_repair import _sha
from ifv_training.psd_repair_verifier import build_complete_hinted_episode_trace
from src.orchestrator.stage_runner import StageStep
from test_psd_repair_finalize import _base_trace, _site, _teacher_steps
from test_psd_source_review import artifact, Client


def test_reviewer_can_bind_report_ids_without_exposing_unrelated_metadata():
    source = _base_trace()
    metadata = source['state']['all_steps'][0]['metadata']
    metadata.update(api_key='secret', policy_token_capture={'prompt_token_ids': [999]},
                    private_reference='private answer')
    projected = trace_steps(source)
    assert projected[0]['function_call_id'] == 'failed-call'
    assert 'secret' not in str(projected) and 'private answer' not in str(projected)
    assert 'policy_token_capture' not in str(projected)
    assert 'function_call_id' not in trace_steps(source, include_transport_ids=False)[0]


def test_new_source_review_and_legacy_binding_are_distinguished(monkeypatch, tmp_path):
    from ifv_training import psd_source_review as review
    source, gold = _base_trace(), {'case_id': 'case-1', 'factual_status': 'supported'}
    old = artifact(source, gold=gold)
    assert review.validate_source_review(old, trace=source, gold=gold) == 'pass'
    monkeypatch.setattr(review, 'review_images', lambda *a, **kw: ([], old['media']))
    new = asyncio.run(review.judge_source(Client(old['decision']), source,
        gold=gold, image_path=tmp_path / 'unused', model='test', cache_dir=tmp_path / 'cache'))
    assert new['trace_projection'] == 'transport_ids_v2'
    assert review.validate_source_review(new, trace=source, gold=gold) == 'pass'
    assert new['provenance']['request_binding']['packet_sha256'] != old['provenance']['request_binding']['packet_sha256']
    new.pop('trace_projection')
    with pytest.raises(ValueError, match='packet binding'):
        review.validate_source_review(new, trace=source, gold=gold)


def test_complete_episode_keeps_rejected_judgment_and_usage():
    source, steps = _base_trace(), _teacher_steps()
    rejection = StageStep(stage_name='psd_teacher_judgment', action_type='format_error',
        thought='Rejected draft', tokens={'completion': 123},
        metadata={'policy_action': {'verdict_observation_ids': ['invented']},
                  'judgment_output_rejected': True})
    steps.insert(-1, rejection)
    episode = build_complete_hinted_episode_trace(source, failure_site=_site(source),
        teacher_steps=steps, stop_reason='model_finished')
    rows = episode['state']['all_steps']
    assert [r['action_type'] for r in rows] == ['tool_call', 'format_error', 'output']
    assert rows[1]['stage'] == 'unified_judgment'
    assert rows[1]['metadata']['policy_action']['verdict_observation_ids'] == ['invented']
    assert episode['token_usage']['completion'] == 123 + 4 + 8
    assert source == _base_trace()


def test_slate_verifier_receives_and_binds_actual_hint(monkeypatch):
    from ifv_training import psd_slate as slate, psd_gemini_judge as judge
    source = _base_trace()
    episode = build_complete_hinted_episode_trace(source, failure_site=_site(source),
        teacher_steps=_teacher_steps(), stop_reason='model_finished')
    episode['psd_repair']['slate'] = {'used_positions': [0]}
    original = copy.deepcopy(episode)
    hint = 'Check the unresolved relation.'
    targets = [{'position': 0, 'hint': hint, 'hint_record': {'text': hint}}]
    monkeypatch.setattr(judge, 'review_images', lambda *a, **kw: ([], {}))
    packets = []
    async def request(client, packet, **kw):
        packets.append(packet)
        assert packet['hints'] == {'0': hint}
        assert packet['repaired_steps'][0]['function_call_id'] == 'repaired-call'
        assert packet['repaired_steps'][0]['injected_procedural_hint'] == hint
        assert 'String\nfilters are not proof' in kw['prompt']
        return {'status': 'fail', 'passing_positions': [], 'failed_position': 0,
            'explanation': 'Synthetic negative hint audit.', 'evidence': [
                {'position': 0, 'step_index': 0, 'trace': 'repaired', 'quote': hint}]}, {}
    monkeypatch.setattr(judge, '_request', request)
    args = dict(source=source, episode=episode, gold={}, image_path='unused', model='judge', cache_dir=None)
    result = asyncio.run(slate.review_slate(None, **args, targets=targets))
    assert result['hints_sha256'] == _sha({'0': hint}) and result['decision']['status'] == 'fail'
    assert episode == original
    for invalid in ([], [*targets, *targets], [{'position': 0, 'hint': hint, 'hint_record': {'text': 'different'}}]):
        with pytest.raises(ValueError, match='exact hints'):
            asyncio.run(slate.review_slate(None, **args, targets=invalid))
    assert len(packets) == 1


def test_slate_writer_lock_precedes_any_provider_or_input_write(monkeypatch, tmp_path):
    import scripts.run_psd_repair_driver as driver
    calls = []
    args = SimpleNamespace(search_mode='slate', skip_auto_judge=False,
                           verification_bundle=None, output_dir=tmp_path / 'case')
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def child(args):
            calls.append(args)
            entered.set()
            await release.wait()
            return {'status': 'bounded_test'}
        monkeypatch.setattr(driver, '_run_single', child)
        first = asyncio.create_task(driver._run(args))
        await entered.wait()
        try:
            with pytest.raises(OSError):
                await driver._run(args)
            assert len(calls) == 1 and not args.output_dir.exists()
        finally:
            release.set()
            await first
        assert await driver._run(args) == {'status': 'bounded_test'}
        assert len(calls) == 2
    asyncio.run(run())

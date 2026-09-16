import asyncio
import copy
from types import SimpleNamespace

from ifv_training.psd_repair import FailureSite, PSDModelRoles, HintProposal
from ifv_training.psd_repair_runtime import QwenContinuationAdapter
from ifv_training.psd_slate import capture_target
from src.orchestrator.investigation_models import InvestigationSegmentOutput
from src.orchestrator.react_runtime import compile_react_judgment_basis
from src.orchestrator.stage_runner import StageRunner, StageStep


def adapter():
    roles = PSDModelRoles(hint_constructor_provider='gemini', hint_constructor_model='judge',
        frozen_self_teacher_provider='qwen_local', frozen_self_teacher_model='frozen',
        round_start_checkpoint='model', round_start_checkpoint_manifest_sha256='a' * 64,
        trainable_student_provider='qwen_local', trainable_student_model='frozen',
        trainable_student_initial_checkpoint='model')
    return QwenContinuationAdapter(
        policy_llm=SimpleNamespace(provider='qwen_local', wire_api='chat_completions', model_name='frozen'),
        hint_constructor_llm=SimpleNamespace(provider='gemini', model_name='judge'),
        model_roles=roles, tools=[], image_path='', require_runtime_archive=False,
        react_system_prompt='base react prompt', judgment_system_prompt='base judgment prompt')


def test_repair_renders_native_system_exactly_once_from_same_base_prompt():
    model = adapter()
    normal = StageRunner(llm=model.policy_llm, tools=[], system_prompt=model.react_system_prompt,
                         output_schema=InvestigationSegmentOutput)
    archived = normal._build_native_chat_system_content(available_tool_names=['old_tool'])
    site = FailureSite(step_index=0, step_id='step', stage='unified_react', example_type='tool_call',
                       policy_input={'system_instruction': archived}, policy_action={})
    runner = model._runner(site=site, history=[{'role': 'system', 'content': archived},
        {'role': 'user', 'content': 'image'}], include_hint_as_pending_user=True, stage_name='psd_teacher_repair')
    current = runner._build_native_chat_system_content(available_tool_names=['new_tool'])
    assert current == normal._build_native_chat_system_content(available_tool_names=['new_tool'])
    assert current.count('Runtime tool availability:') == 1
    assert current.count('Qwen local structured-output compatibility:') == 1
    assert 'old_tool' not in current


def test_judgment_uses_original_base_prompt_not_rendered_source_system():
    model = adapter()
    archived = 'base judgment prompt\nRuntime tool availability:\nold schema'
    runner = model._judgment_runner(history=[{'role': 'system', 'content': archived},
        {'role': 'user', 'content': 'original image'}], basis={}, include_pending_user=True,
        system_instruction=archived)
    assert runner.system_prompt == model.judgment_system_prompt
    assert runner.native_history == [{'role': 'user', 'content': 'original image'}]
    assert runner._build_native_chat_system_content().count('Runtime tool availability:') == 1


def test_psd_diagnostic_stage_names_do_not_erase_real_observation_ids():
    state = SimpleNamespace(objective='verify', action_count=3, stop_reason='done', finish_rationale='done')
    def step(stage, call, result):
        return StageStep(stage_name=stage, action_type='tool_call', tool_name='text_search',
            tool_result=result, metadata={'function_call_id': call})
    rows = [step('unified_react', 'source', '{"status":"success","data":{}}'),
            step('psd_teacher_repair', 'repaired', '{"status":"success","data":{}}'),
            step('psd_teacher_repair', 'outage', '{"status":"error","error":"timeout"}')]
    original = copy.deepcopy(rows)
    assert compile_react_judgment_basis(state, rows)['observation_ids'] == ['source']
    basis = QwenContinuationAdapter._judgment_basis(state, rows)
    assert basis['observation_ids'] == ['source', 'repaired']
    assert [row['status'] for row in basis['observations']] == ['success', 'success', 'error']
    assert rows == original and rows[1].stage_name == 'psd_teacher_repair'


def test_exact_capture_uses_current_bound_system_not_source_availability(monkeypatch):
    import src.orchestrator.runtime_events as events
    prefix = [{'role': 'user', 'content': 'original image'}]
    current = 'base prompt with current available tools'
    messages = [{'role': 'system', 'content': current}, *prefix, {'role': 'user', 'content': 'advice'}]
    request = {'input_payload': copy.deepcopy(messages), 'tool_schema': []}
    monkeypatch.setattr(events, 'reconstruct_archived_request', lambda *_: copy.deepcopy(request))
    step = StageStep(action_type='tool_call', metadata={'context_request_id': 'req',
        'policy_input': StageRunner._policy_input_snapshot(system_instruction=current,
            input_payload=messages[1:], tools=[], response_format=None),
        'policy_token_capture': {'status': 'complete', 'prompt_token_ids': [1, 2, 3], 'completion_token_ids': [4]}})
    seen = []
    async def tokenize(body):
        seen.append(body)
        assert body['messages'][0]['content'] == current
        return [1, 2, 3] if len(seen) == 1 else [1, 2]
    result = asyncio.run(capture_target(position=3,
        hint=HintProposal(text='advice', level=1, provider='test', model='test', proposal_id='id', audit={}),
        steps=[step], unhinted_prefix=prefix, runtime_store=SimpleNamespace(root='archive'),
        system_instruction='source prompt with old tools', tokenize=tokenize, model='frozen'))
    assert result['student_prompt_ids'] == [1, 2]
    assert seen[1]['messages'] == messages[:-1]

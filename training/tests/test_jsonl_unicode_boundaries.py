import importlib.util
import json
from pathlib import Path

from ifv_training.io import load_jsonl


def test_jsonl_preserves_unicode_line_separators(tmp_path):
    value = {'text': 'before\u2028middle\u2029after\u0085end'}
    path = tmp_path / 'rows.jsonl'
    path.write_text(json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8')
    assert load_jsonl(path) == [value]
    probe_path = Path(__file__).resolve().parents[1] / 'scripts/probe/verify_ms_swift_agent_dataset.py'
    spec = importlib.util.spec_from_file_location('probe_unicode', probe_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert list(module._rows(path)) == [(0, value)]


def test_sft_gate_preserves_unicode_line_separators(tmp_path):
    row = {
        'messages': [
            {'role': 'assistant', 'content': '<think>before\u2028after</think>'},
            {
                'role': 'tool_call',
                'content': '{"name":"current_time","arguments":"{}"}',
                'loss': False,
            },
        ]
    }
    path = tmp_path / 'rows.jsonl'
    path.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
    probe_path = (
        Path(__file__).resolve().parents[1]
        / 'scripts/probe/verify_sft_data_contract.py'
    )
    spec = importlib.util.spec_from_file_location('sft_gate_unicode', probe_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._expected_loss_targets([path]) == {
        'supervised_tool_call_targets': 0,
        'masked_tool_call_targets': 1,
        'supervised_thought_targets': 1,
        'masked_thought_targets': 0,
    }

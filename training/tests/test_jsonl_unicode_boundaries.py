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


def test_legacy_processor_probes_preserve_unicode_line_separators(tmp_path):
    values = [
        {'messages': [], 'text': 'before\u2028after'},
        {'messages': [], 'text': 'before\u2029after'},
    ]
    path = tmp_path / 'rows.jsonl'
    path.write_text(
        ''.join(json.dumps(value, ensure_ascii=False) + '\n' for value in values),
        encoding='utf-8',
    )
    probe_root = Path(__file__).resolve().parents[1] / 'scripts/probe'

    audit_spec = importlib.util.spec_from_file_location(
        'audit_processor_unicode', probe_root / 'audit_ms_swift_processor.py'
    )
    audit_module = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit_module)
    assert [row for _, row in audit_module._rows(path)] == values

    template_spec = importlib.util.spec_from_file_location(
        'template_processor_unicode', probe_root / 'ms_swift_template.py'
    )
    template_module = importlib.util.module_from_spec(template_spec)
    template_spec.loader.exec_module(template_module)
    assert template_module._row(path, 0) == values[0]
    assert template_module._row(path, 1) == values[1]

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

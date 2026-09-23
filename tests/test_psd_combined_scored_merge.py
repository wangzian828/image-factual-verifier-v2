import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/server/merge_psd_combined_scored.py"
spec = importlib.util.spec_from_file_location("merge_psd_combined_scored", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_stat_binding_changes_if_selected_source_changes(tmp_path):
    path = tmp_path / "targets.jsonl"
    path.write_text("first\n")
    initial = module._stat(path)
    path.write_text("first\nsecond\n")
    assert module._stat(path) != initial

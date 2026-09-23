import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/server/score_psd_old1000_raw_fourgpu.py"
spec = importlib.util.spec_from_file_location("score_psd_old1000_raw_fourgpu", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_small_state_receipt_is_atomic_and_stat_bound(tmp_path):
    path = tmp_path / "state.json"
    module._save(path, {"phase": "launched", "formal_training_started": False})
    assert module._load(path) == {"phase": "launched", "formal_training_started": False}
    before = module._stat(path)
    module._save(path, {"phase": "raw_teacher_scoring", "formal_training_started": False})
    assert module._stat(path) != before
    assert not (tmp_path / ".state.json.tmp").exists()

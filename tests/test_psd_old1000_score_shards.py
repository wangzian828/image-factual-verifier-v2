import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/server/prepare_psd_old1000_score_shards.py"
spec = importlib.util.spec_from_file_location("prepare_psd_old1000_score_shards", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_shards_are_assigned_to_least_estimated_token_work():
    assert module._least_loaded([100, 90, 90, 100], [1, 2, 1, 0]) == 2
    assert module._least_loaded([0, 0, 0, 0], [0, 0, 0, 0]) == 0
    with pytest.raises(ValueError, match="four"):
        module._least_loaded([1, 2], [1, 2])

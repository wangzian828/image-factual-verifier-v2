import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/h20/run_psd_lightweight_resume.py"
SPEC = importlib.util.spec_from_file_location("psd_lightweight_resume_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_retry_delay_is_short_after_progress_and_bounded_on_stalls():
    assert MODULE.retry_delay_seconds(0) == 5
    assert [MODULE.retry_delay_seconds(i) for i in range(1, 8)] == [15, 30, 60, 120, 240, 300, 300]
    with pytest.raises(ValueError):
        MODULE.retry_delay_seconds(-1)

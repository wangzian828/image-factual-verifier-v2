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


def test_prepare_command_resumes_search_directly_with_attestation(tmp_path, monkeypatch):
    deploy = tmp_path / "deploy"
    code = deploy / "code"
    deploy.mkdir()
    (deploy / "fast-resume-inputs.json").write_text("{}")
    (deploy / "selected-candidates-offset-index.json").write_text("{}")
    monkeypatch.setattr(MODULE, "DEPLOY", deploy)
    monkeypatch.setattr(MODULE, "CODE", code)
    command = MODULE.prepare_command()
    assert command[2] == str(code / "scripts/run_psd_feedback_canary.py")
    assert "run_psd_round.py" not in " ".join(command)
    assert "--defer-topk" not in command
    assert "--resume-input-receipt" in command
    assert "--resume-selection-index" in command

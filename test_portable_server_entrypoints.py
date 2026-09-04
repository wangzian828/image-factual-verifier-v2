from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_generic_server_entrypoints_have_no_machine_specific_paths() -> None:
    paths = [
        "scripts/server/ifv_env.sh",
        "scripts/server/run_ifv.sh",
        "scripts/server/bootstrap_runtime.sh",
        "scripts/server/start_eval.sh",
        "scripts/server/start_gemini_eval.sh",
        "scripts/server/start_teacher_rollout.sh",
        "scripts/server/eval_worker.sh",
        "scripts/server/poll_eval.sh",
        "scripts/server/doctor.py",
        "configs/runtime.env.example",
    ]
    forbidden = (
        "/gsdata/home/",
        "/gs/home/",
        "C:\\Users\\",
        "gpu-13",
        "100.10.1.210",
    )
    for path in paths:
        source = _source(path)
        for value in forbidden:
            assert value not in source, f"{path} contains {value}"


def test_generic_launcher_uses_explicit_runtime_configuration() -> None:
    env = _source("scripts/server/ifv_env.sh")
    runner = _source("scripts/server/run_ifv.sh")
    starter = _source("scripts/server/start_eval.sh")
    rollout = _source("scripts/server/start_teacher_rollout.sh")

    assert "XDG_DATA_HOME" in env
    assert "IFV_SERVER_PROXY" in env
    assert 'source "${_ifv_runtime_env}"' in env
    assert "IFV_EXPECTED_REPO_ROOT" in runner
    assert "IFV_EXPECTED_BRANCH" in runner
    assert "GEMINI_EVAL_MAX_CONCURRENCY" in starter
    assert "GEMINI_MAX_INFLIGHT_REQUESTS" in starter
    assert "IFV_EVAL_MODULE=src.eval.run_cases" in rollout


def test_runtime_template_contains_no_credentials() -> None:
    template = _source("configs/runtime.env.example")

    assert "GEMINI_API_KEY=" in template
    assert "SERPER_API_KEY=" in template
    assert "000000" not in template

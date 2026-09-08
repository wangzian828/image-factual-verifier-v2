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
        "scripts/server/start_teacher_rollout_autopilot.sh",
        "scripts/server/start_teacher_rollout_portable.sh",
        "scripts/server/prepare_factcheck_dataset.sh",
        "scripts/server/run_teacher_sft_pipeline.sh",
        "scripts/trajectory/verify_teacher_sft_delivery.py",
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
    autopilot = _source("scripts/server/start_teacher_rollout_autopilot.sh")
    portable = _source("scripts/server/start_teacher_rollout_portable.sh")
    dataset = _source("scripts/server/prepare_factcheck_dataset.sh")

    assert "XDG_DATA_HOME" in env
    assert "IFV_SERVER_PROXY" in env
    assert 'source "${_ifv_runtime_env}"' in env
    assert "IFV_EXPECTED_REPO_ROOT" in runner
    assert "IFV_EXPECTED_BRANCH" in runner
    assert "GEMINI_EVAL_MAX_CONCURRENCY" in starter
    assert "GEMINI_MAX_INFLIGHT_REQUESTS" in starter
    assert "IFV_EVAL_MODULE=src.eval.run_cases" in rollout
    assert "IFV_DATA_ROOT" in autopilot
    assert "run_teacher_rollout_autopilot.py" in autopilot
    assert "nohup" in autopilot
    assert "factcheck_train-8490-20260907.tar.gz" in dataset
    assert "2fda3ca7144d899e355fcbbaa4e6b93350878dbf5225ab423402123d5e37a448" in dataset
    assert 'run_mode="smoke_then_full"' in portable
    assert 'smoke_limit="${IFV_SMOKE_CASE_COUNT:-10}"' in portable
    assert "--full" in portable
    assert "--smoke-only" in portable
    assert "--smoke-then-full" in portable
    assert "--validation-count" in portable
    assert "run_teacher_sft_pipeline.sh" in portable
    assert 'delivery_scope="smoke_not_final"' in portable
    assert "IFV_REQUIRE_FULL_TEACHER_DELIVERY" in portable
    assert "QWEN_TEACHER_BASE_URL" in portable
    assert 'export QWEN_TEACHER_MODEL="${rollout_model}"' in portable
    assert 'export QWEN_TEACHER_VISION_MODEL="${rollout_model}"' in portable
    assert "QWEN_TEACHER_VISION_MODEL must match the main teacher model" in portable
    assert 'export QWEN_LOCAL_API_KEY="${QWEN_TEACHER_API_KEY:-none}"' in portable
    assert "--sft-judge-api-key-env" in portable
    assert 'export BROWSE_EXTRACT_PROVIDER="qwen_local"' in portable
    assert 'export BROWSE_EXTRACT_BASE_URL="${QWEN_TEACHER_BASE_URL}"' in portable
    assert "OCR_BACKEND=baidu is required" in portable
    assert "local OCR and OCR fallback are prohibited" in portable
    assert "SERPER_API_KEY is required" in portable
    assert "JINA_API_KEY is required" in portable
    assert "IMAGE_UPLOAD_PROVIDER must be explicitly set to oss, custom, or temp" in portable


def test_runtime_template_contains_no_credentials() -> None:
    template = _source("configs/runtime.env.example")

    assert "GEMINI_API_KEY=" in template
    assert "SERPER_API_KEY=" in template
    assert "OCR_BACKEND=baidu" in template
    assert "BAIDU_OCR_API_KEY=" in template
    assert "BROWSE_FETCH_PROVIDER=jina" in template
    assert "IMAGE_UPLOAD_PROVIDER=oss" in template
    assert "000000" not in template


def test_teacher_sft_pipeline_requires_explicit_output_and_processor_opt_in() -> None:
    source = _source("scripts/server/run_teacher_sft_pipeline.sh")
    assert "--output-dir is required" in source
    assert "IFV_REQUIRE_PROCESSOR" in source
    assert "IFV_MODEL_ID" in source
    assert "IFV_TRAINING_PYTHON" in source
    assert "--run-training" in source
    assert "training_status" in source
    assert "verify_teacher_sft_delivery.py" in source
    assert "final-delivery.json" in source
    assert 'phase=smoke output_dir=%s\\n' in source
    assert 'phase=full output_dir=%s\\n' in source

"""Run a real-provider benchmark canary and enforce trace-level acceptance.

No providers, model responses, tools, or network calls are mocked here. The input
must be a finalized runtime release with real image assets and evaluator-private
companions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_real_trace import audit_trace, discover_trace_files  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


SEARCH_TOOLS = frozenset({"reverse_image_search", "text_search", "crop_and_search"})
REVERSE_IMAGE_TOOLS = frozenset({"reverse_image_search"})
VISIT_TOOLS = frozenset({"visit"})
VISUAL_TOOLS = frozenset(
    {
        "ocr_with_position",
        "compare_with_reference",
        "crop_and_inspect",
        "check_consistency",
        "analyze_visual_anomalies",
        "count_objects",
    }
)


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"real canary requires environment variable {name}")
    return value


def _validate_provider_environment() -> None:
    if not (os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()):
        raise RuntimeError("real canary requires GEMINI_API_KEY or GOOGLE_API_KEY")
    _required_env("SERPER_API_KEY")

    upload_provider = os.getenv("IMAGE_UPLOAD_PROVIDER", "").strip().lower()
    visual_provider = os.getenv("VISUAL_SEARCH_PROVIDER", "").strip().lower()
    browse_provider = os.getenv("BROWSE_FETCH_PROVIDER", "").strip().lower()
    if upload_provider not in {"oss", "custom", "temp"}:
        raise RuntimeError(
            "set IMAGE_UPLOAD_PROVIDER explicitly to oss, custom, or temp"
        )
    if visual_provider not in {"serper_lens", "zhipu_image_search"}:
        raise RuntimeError(
            "set VISUAL_SEARCH_PROVIDER explicitly to serper_lens or zhipu_image_search"
        )
    if browse_provider not in {"jina", "direct"}:
        raise RuntimeError("set BROWSE_FETCH_PROVIDER explicitly to jina or direct")
    if upload_provider == "oss":
        for name in (
            "OSS_ACCESS_KEY_ID",
            "OSS_ACCESS_KEY_SECRET",
            "OSS_ENDPOINT",
            "OSS_BUCKET_NAME",
        ):
            _required_env(name)
    elif upload_provider == "custom":
        _required_env("IMAGE_UPLOAD_API_URL")
    if visual_provider == "zhipu_image_search":
        _required_env("ZHIPU_API_KEY")
    if browse_provider == "jina":
        _required_env("JINA_API_KEY")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _tool_result_succeeded(step: Mapping[str, Any]) -> bool:
    result = step.get("tool_result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return False
    return isinstance(result, Mapping) and result.get("status") == "success"


def _accepted_tools(trace: Mapping[str, Any]) -> set[str]:
    state = _mapping(trace.get("state"))
    return {
        str(step.get("tool_name", "")).strip()
        for step in _rows(state.get("all_steps"))
        if step.get("action_type") == "tool_call"
        and _tool_result_succeeded(step)
        and str(step.get("tool_name", "")).strip()
    }


def _require_real_run_artifacts(
    run_dir: Path,
    *,
    required_claim_modes: set[str],
) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "summary.json"
    trace_dir = run_dir / "traces"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise RuntimeError("real canary did not produce run_manifest.json and summary.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    agent = _mapping(manifest.get("agent"))
    model = str(agent.get("model", "")).lower()
    if str(agent.get("provider", "")).lower() != "gemini":
        raise RuntimeError("real canary must use provider=gemini")
    if any(marker in model for marker in ("fake", "fixture", "scripted", "mock")):
        raise RuntimeError(f"real canary rejected non-real model name: {model}")
    if manifest.get("status") != "completed":
        raise RuntimeError(f"real canary run status is {manifest.get('status')!r}")
    if int(summary.get("num_errors", 0) or 0):
        raise RuntimeError("real canary contains engineering errors")

    trace_files = discover_trace_files(trace_dir)
    if not trace_files:
        raise RuntimeError("real canary produced no canonical traces")
    tools: set[str] = set()
    observed_claim_modes: set[str] = set()
    for path in trace_files:
        report = audit_trace(path)
        failures = report.failures(strict_scheduler=True)
        if failures:
            rendered = "; ".join(f"{item.code}: {item.message}" for item in failures)
            raise RuntimeError(f"strict trace audit failed for {path.name}: {rendered}")
        trace = json.loads(path.read_text(encoding="utf-8"))
        state = _mapping(trace.get("state"))
        verification_case = _mapping(state.get("verification_case"))
        claim_mode = str(verification_case.get("claim_mode", "")).strip()
        if claim_mode:
            observed_claim_modes.add(claim_mode)
        if trace.get("termination") != "success":
            raise RuntimeError(f"trace did not terminate successfully: {path.name}")
        if int(trace.get("llm_api_calls", 0) or 0) <= 0:
            raise RuntimeError(f"trace recorded no real model calls: {path.name}")
        tools.update(_accepted_tools(trace))

    missing_classes = []
    if not tools.intersection(SEARCH_TOOLS):
        missing_classes.append("search")
    if not tools.intersection(REVERSE_IMAGE_TOOLS):
        missing_classes.append("reverse-image search and image upload")
    if not tools.intersection(VISIT_TOOLS):
        missing_classes.append("visit")
    if not tools.intersection(VISUAL_TOOLS):
        missing_classes.append("visual observation")
    if missing_classes:
        raise RuntimeError(
            "real canary did not exercise required tool classes: "
            + ", ".join(missing_classes)
        )
    missing_modes = sorted(required_claim_modes - observed_claim_modes)
    if missing_modes:
        raise RuntimeError(
            "real canary did not exercise required claim modes: "
            + ", ".join(missing_modes)
        )
    return {
        "passed": True,
        "run_dir": str(run_dir.resolve()),
        "trace_count": len(trace_files),
        "claim_modes": sorted(observed_claim_modes),
        "successful_tools": sorted(tools),
        "summary": summary,
    }


def _command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.eval.run_eval",
        "--benchmark",
        str(args.benchmark),
        "--output-dir",
        str(args.output_dir),
        "--provider",
        "gemini",
        "--model",
        args.model,
        "--llm-wire-api",
        "interactions",
        "--vlm-wire-api",
        "interactions",
        "--concurrency",
        "1",
        "--limit",
        str(args.limit),
    ]
    if args.source_access_policy:
        command.extend(["--source-access-policy", str(args.source_access_policy)])
    if args.evaluator_private:
        command.extend(["--evaluator-private", str(args.evaluator_private)])
    if args.evaluation_gold:
        command.extend(["--evaluation-gold", str(args.evaluation_gold)])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a no-mock real-provider canary and strict trace audit."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--source-access-policy", type=Path)
    parser.add_argument("--evaluator-private", type=Path)
    parser.add_argument("--evaluation-gold", type=Path)
    parser.add_argument(
        "--required-claim-mode",
        action="append",
        choices=["external_claim", "embedded_claim"],
        default=[],
        help=(
            "Claim mode that must appear in successful traces. Repeat as needed. "
            "Defaults to requiring both external_claim and embedded_claim."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"real canary output directory must be new or empty: {args.output_dir}"
        )
    _validate_provider_environment()
    completed = subprocess.run(_command(args), cwd=REPO_ROOT, check=False)
    if completed.returncode:
        return completed.returncode
    required_claim_modes = set(
        args.required_claim_mode or ["external_claim", "embedded_claim"]
    )
    result = _require_real_run_artifacts(
        args.output_dir,
        required_claim_modes=required_claim_modes,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

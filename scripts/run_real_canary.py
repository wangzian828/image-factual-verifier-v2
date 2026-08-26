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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import load_project_dotenv  # noqa: E402
from src.eval.release_adapter import (  # noqa: E402
    DATA_PIPELINE_DECISION_POLICY_VERSION,
    load_runtime_release,
)
from src.workflow import AGENT_DECISION_POLICY_VERSION  # noqa: E402
from scripts.audit_real_trace import audit_trace, discover_trace_files  # noqa: E402
from src.orchestrator.investigation_models import target_fact_rows  # noqa: E402

load_project_dotenv(REPO_ROOT)


SEARCH_TOOLS = frozenset({"reverse_image_search", "text_search", "crop_and_search"})
VISIT_TOOLS = frozenset({"visit"})
EVIDENCE_INSPECTION_TOOLS = VISIT_TOOLS | frozenset(
    {
        "compare_with_reference",
        "crop_and_inspect",
        "check_consistency",
        "analyze_visual_anomalies",
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


def _git_output(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "real canary requires a readable Git checkout"
        ) from exc
    return completed.stdout.strip()


def _require_clean_runtime_checkout() -> str:
    """Bind the canary manifest to the exact clean source tree being executed."""

    actual_commit = _git_output("rev-parse", "HEAD")
    configured_commit = os.getenv("GIT_COMMIT", "").strip()
    if configured_commit and configured_commit != actual_commit:
        raise RuntimeError(
            "GIT_COMMIT does not match the runtime checkout: "
            f"configured={configured_commit}, actual={actual_commit}"
        )
    dirty = _git_output("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError(
            "real canary requires a clean Git worktree; commit or move source "
            "changes before running"
        )
    return actual_commit


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


def _missing_required_tool_classes(tools: set[str]) -> list[str]:
    """Require a real retrieval route and one route-specific evidence check.

    An image identity case may close through reverse search plus same-capture
    comparison, while a geographic or ecological case may close through text
    search plus page inspection. Requiring both visit and visual comparison in
    every canary would force unnecessary post-determination actions.
    """

    missing_classes = []
    if not tools.intersection(SEARCH_TOOLS):
        missing_classes.append("search")
    if not tools.intersection(EVIDENCE_INSPECTION_TOOLS):
        missing_classes.append("evidence inspection")
    return missing_classes


def _require_real_run_artifacts(
    run_dir: Path,
) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "summary.json"
    trace_dir = run_dir / "traces"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise RuntimeError("real canary did not produce run_manifest.json and summary.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    agent = _mapping(manifest.get("agent"))
    benchmark = _mapping(manifest.get("benchmark"))
    model = str(agent.get("model", "")).lower()
    if str(agent.get("provider", "")).lower() != "gemini":
        raise RuntimeError("real canary must use provider=gemini")
    if agent.get("profile_id") != "teacher-gemini":
        raise RuntimeError("real canary must use profile=teacher-gemini")
    if any(marker in model for marker in ("fake", "fixture", "scripted", "mock")):
        raise RuntimeError(f"real canary rejected non-real model name: {model}")
    if manifest.get("status") != "completed":
        raise RuntimeError(f"real canary run status is {manifest.get('status')!r}")
    if int(summary.get("num_errors", 0) or 0):
        raise RuntimeError("real canary contains engineering errors")
    if benchmark.get("input_mode") != "image_only":
        raise RuntimeError("real canary requires benchmark input_mode=image_only")
    if (
        benchmark.get("decision_policy_version")
        != DATA_PIPELINE_DECISION_POLICY_VERSION
    ):
        raise RuntimeError(
            "real canary requires data-pipeline decision_policy_version="
            f"{DATA_PIPELINE_DECISION_POLICY_VERSION}"
        )
    if agent.get("decision_policy_version") != AGENT_DECISION_POLICY_VERSION:
        raise RuntimeError(
            "real canary requires Agent decision_policy_version="
            f"{AGENT_DECISION_POLICY_VERSION}"
        )

    trace_files = discover_trace_files(trace_dir)
    if not trace_files:
        raise RuntimeError("real canary produced no canonical traces")
    tools: set[str] = set()
    for path in trace_files:
        report = audit_trace(path)
        failures = report.failures(strict_scheduler=True)
        if failures:
            rendered = "; ".join(f"{item.code}: {item.message}" for item in failures)
            raise RuntimeError(f"strict trace audit failed for {path.name}: {rendered}")
        trace = json.loads(path.read_text(encoding="utf-8"))
        state = _mapping(trace.get("state"))
        if trace.get("input_mode") != "image_only":
            raise RuntimeError(f"trace is not image_only: {path.name}")
        if trace.get("decision_policy_version") != AGENT_DECISION_POLICY_VERSION:
            raise RuntimeError(
                f"trace is not {AGENT_DECISION_POLICY_VERSION}: {path.name}"
            )
        investigation = _mapping(state.get("investigation_state"))
        if investigation.get("core_verdict_fact_id"):
            raise RuntimeError(f"trace activates a legacy core fact: {path.name}")
        claims = target_fact_rows(investigation)
        if not claims or not any(
            claim.get("salience") == "high" for claim in claims
        ):
            raise RuntimeError(
                f"trace has no high-salience ImageClaim: {path.name}"
            )
        basis = _mapping(trace.get("verdict_basis"))
        if not basis.get("claim_ids"):
            raise RuntimeError(f"trace has no verdict basis claims: {path.name}")
        decision_mode = str(
            basis.get("decision_mode", "evidence_determined")
            or "evidence_determined"
        )
        if (
            trace.get("verdict") == "fake"
            and decision_mode == "evidence_determined"
            and not basis.get("discrepancy_ids")
        ):
            raise RuntimeError(
                f"fake trace has no selected discrepancy: {path.name}"
            )
        if decision_mode == "bounded_binary_judgment" and (
            not basis.get("unresolved_gaps")
            or investigation.get("stop_reason")
            not in {
                "meaningful_routes_exhausted",
                "hard_budget_exhausted",
            }
        ):
            raise RuntimeError(
                f"bounded binary trace lacks its terminal gaps: {path.name}"
            )
        if trace.get("termination") != "success":
            raise RuntimeError(f"trace did not terminate successfully: {path.name}")
        if int(trace.get("llm_api_calls", 0) or 0) <= 0:
            raise RuntimeError(f"trace recorded no real model calls: {path.name}")
        tools.update(_accepted_tools(trace))

    missing_classes = _missing_required_tool_classes(tools)
    if missing_classes:
        raise RuntimeError(
            "real canary did not exercise required tool classes: "
            + ", ".join(missing_classes)
        )
    return {
        "passed": True,
        "run_dir": str(run_dir.resolve()),
        "trace_count": len(trace_files),
        "input_mode": "image_only",
        "data_pipeline_decision_policy_version": (
            DATA_PIPELINE_DECISION_POLICY_VERSION
        ),
        "agent_decision_policy_version": AGENT_DECISION_POLICY_VERSION,
        "successful_tools": sorted(tools),
        "summary": summary,
    }


def _command(args: argparse.Namespace) -> list[str]:
    requested_case_ids = list(args.case_id or [])
    effective_limit = len(requested_case_ids) if requested_case_ids else args.limit
    command = [
        sys.executable,
        "-m",
        "src.eval.run_eval",
        "--benchmark",
        str(args.benchmark),
        "--output-dir",
        str(args.output_dir),
        "--profile",
        "teacher-gemini",
        "--image-access-mode",
        getattr(args, "image_access_mode", "direct_multimodal"),
        "--concurrency",
        "1",
        "--limit",
        str(effective_limit),
    ]
    for case_id in requested_case_ids:
        command.extend(["--case-id", case_id])
    if args.source_access_policy:
        command.extend(["--source-access-policy", str(args.source_access_policy)])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a no-mock real-provider canary and strict trace audit."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument(
        "--image-access-mode",
        choices=["direct_multimodal", "separate_vlm"],
        default="direct_multimodal",
        help="Image access mode for the main LLM.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help=(
            "Explicit canary case ID. Repeat for an ordered multi-case canary; "
            "the effective limit becomes the number of selected IDs."
        ),
    )
    parser.add_argument("--source-access-policy", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    requested_case_ids = list(args.case_id or [])
    if len(requested_case_ids) != len(set(requested_case_ids)):
        raise ValueError("--case-id values must be unique")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"real canary output directory must be new or empty: {args.output_dir}"
        )
    release = load_runtime_release(args.benchmark)
    if release.input_mode != "image_only":
        raise RuntimeError("real canary accepts image-only v0.3 releases only")
    if (
        release.decision_policy_version
        != DATA_PIPELINE_DECISION_POLICY_VERSION
    ):
        raise RuntimeError(
            "real canary accepts data-pipeline policy "
            f"{DATA_PIPELINE_DECISION_POLICY_VERSION} releases only"
        )
    _validate_provider_environment()
    runtime_commit = _require_clean_runtime_checkout()
    child_env = os.environ.copy()
    child_env["GIT_COMMIT"] = runtime_commit
    completed = subprocess.run(
        _command(args),
        cwd=REPO_ROOT,
        check=False,
        env=child_env,
    )
    if completed.returncode:
        return completed.returncode
    result = _require_real_run_artifacts(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Test-only harness for replaying the runtime-v3-final-20260717 contract."""

from __future__ import annotations

import time
from types import MethodType
from typing import Any

from src.orchestrator.bootstrap import build_bootstrap_investigation
from src.orchestrator.coverage import audit_coverage, compile_verdict_basis
from src.orchestrator.runtime_case import verify_case_image
from src.orchestrator.stage_runner import InteractionSession
from src.orchestrator.state import VerificationState
from src.orchestrator.task_store import state_from_bootstrap


def install_frozen_v3_runtime(orchestrator: Any) -> None:
    """Bind the removed v3 public path only inside deterministic replay tests."""

    orchestrator.run = MethodType(_run_frozen_v3, orchestrator)


async def _run_frozen_v3(
    self: Any,
    image_path: str,
    runtime_case: Any,
    *,
    decision_policy_version: str = "reinspect-v2",
) -> dict[str, Any]:
    if decision_policy_version != "reinspect-v2":
        raise RuntimeError("frozen v3 replay requires reinspect-v2")
    verify_case_image(runtime_case, image_path)
    state = VerificationState(
        image_path=image_path,
        image_id=runtime_case.case_id,
        runtime_case=runtime_case,
        input_mode="image_only",
        decision_policy_version="reinspect-v2",
        tool_health=self.tool_health_summary,
    )
    self.last_state = state
    started = time.time()
    try:
        self._validate_image_only_bootstrap_configuration()
        state.perception = await self._run_perception(state, image_path)
        investigation = state_from_bootstrap(
            build_bootstrap_investigation(runtime_case, state.perception)
        )
        state.investigation_state = investigation
        self._sync_image_only_state(state, investigation)
        session = InteractionSession()
        await self._run_image_only_target_planning(
            state,
            investigation,
            image_path=image_path,
            interaction_session=session,
        )
        await self._run_image_only_investigation(
            state,
            investigation,
            image_path,
            runtime_case,
            interaction_session=session,
        )
        self._require_successful_image_only_investigation(state)
        coverage = (
            investigation.coverage_audits[-1]
            if investigation.coverage_audits
            else audit_coverage(investigation)
        )
        verdict, basis = compile_verdict_basis(investigation)
        judgment = await self._run_image_only_judgment(
            state,
            investigation,
            coverage,
            verdict,
            basis,
            interaction_session=session,
        )
        judgment = self._normalize_incomplete_judgment(investigation, judgment)
        investigation.judgment = judgment
        state.judgment = judgment
        state.termination = "success"
        self._sync_image_only_state(state, investigation)
    except Exception as exc:
        state.termination = "error"
        state.errors.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state.stage_timings["total"] = round(time.time() - started, 2)

    return {
        "image_id": state.image_id,
        "image_path": state.image_path,
        "input_mode": state.input_mode,
        "decision_policy_version": state.decision_policy_version,
        "judgment": judgment.model_dump(mode="json"),
        "verdict": judgment.verdict,
        "confidence": judgment.confidence,
        "overall_assessment": judgment.overall_assessment,
        "investigation_status": self._investigation_status(investigation),
        "verification_layers": self._verification_layers(investigation),
        "verdict_basis": basis.model_dump(mode="json"),
        "state": state.to_dict(),
        "termination": state.termination,
        "time_taken": state.stage_timings["total"],
        "token_usage": state.token_usage,
        "total_tool_calls": state.total_tool_calls,
        "total_tool_subcalls": state.total_tool_subcalls,
        "tool_subcalls_by_kind": state.tool_subcalls_by_kind,
        "llm_api_calls": state.llm_api_calls,
        "error": None,
    }

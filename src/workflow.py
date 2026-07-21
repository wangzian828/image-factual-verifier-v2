# -*- coding: utf-8 -*-
"""Image-only v4 workflow: the single entry point for factual investigation.

Usage:
    from src.workflow import VerificationWorkflow, WorkflowConfig

    workflow = VerificationWorkflow(WorkflowConfig())
    result = await workflow.run_single("path/to/image.jpg")
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Limit thread usage to prevent memory explosion
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from src.orchestrator.runtime_case import image_sha256, verify_case_image
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.runtime_events import (
    CANONICAL_TRACE_SCHEMA_VERSION,
    CaseRuntimeStore,
    atomic_write_json,
    bind_case_runtime_store,
    reset_case_runtime_store,
)
from src.orchestrator.state import ImageOnlyRuntimeCase
from src.orchestrator.source_access import SourceAccessPolicy
from src.provider_profiles import resolve_provider_settings
from src.redaction import sanitize_for_persistence
from src.storage import default_trace_dir


AGENT_DECISION_POLICY_VERSION = "discrepancy-first-v4"


@dataclass
class WorkflowConfig:
    """Configuration for the verification workflow."""

    # LLM settings
    profile_id: Optional[str] = None
    provider: Optional[str] = None
    model_name: Optional[str] = None
    vlm_provider: Optional[str] = None  # Defaults to provider
    vlm_model: Optional[str] = None  # Defaults to model_name
    llm_wire_api: Optional[str] = None
    vlm_wire_api: Optional[str] = None
    llm_base_url: Optional[str] = field(default=None, init=False)
    vlm_base_url: Optional[str] = field(default=None, init=False)
    temperature: float = 0.0
    max_tokens: int = 8192

    # Runtime settings
    timeout: float = 1800.0

    # Output
    output_dir: str = field(default_factory=default_trace_dir)
    save_traces: bool = True
    source_access_policy: Optional[SourceAccessPolicy] = None
    decision_policy_version: str = AGENT_DECISION_POLICY_VERSION

    def __post_init__(self) -> None:
        resolved = resolve_provider_settings(
            profile_id=self.profile_id,
            provider=self.provider,
            model_name=self.model_name,
            vlm_provider=self.vlm_provider,
            vlm_model=self.vlm_model,
            llm_wire_api=self.llm_wire_api,
            vlm_wire_api=self.vlm_wire_api,
        )
        self.profile_id = resolved.profile_id
        self.provider = resolved.provider
        self.model_name = resolved.model_name
        self.vlm_provider = resolved.vlm_provider
        self.vlm_model = resolved.vlm_model
        self.llm_wire_api = resolved.llm_wire_api
        self.vlm_wire_api = resolved.vlm_wire_api
        self.llm_base_url = resolved.base_url
        self.vlm_base_url = resolved.vlm_base_url


class VerificationWorkflow:
    """Main workflow for image factual verification."""

    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self._orchestrator: Optional[Orchestrator] = None

    def _get_orchestrator(self, *, validate_startup: bool = True) -> Orchestrator:
        """Lazy init orchestrator."""
        if self._orchestrator is None:
            self._orchestrator = Orchestrator(
                provider=self.config.provider,
                model_name=self.config.model_name,
                vlm_provider=self.config.vlm_provider,
                vlm_model=self.config.vlm_model,
                llm_wire_api=self.config.llm_wire_api,
                vlm_wire_api=self.config.vlm_wire_api,
                llm_base_url=self.config.llm_base_url,
                vlm_base_url=self.config.vlm_base_url,
                timeout=self.config.timeout,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                source_access_policy=self.config.source_access_policy,
                validate_startup=validate_startup,
            )
        return self._orchestrator

    async def run_single(
        self,
        image_path: str,
        image_id: str = "",
        *,
        runtime_case: Optional[ImageOnlyRuntimeCase] = None,
    ) -> Dict[str, Any]:
        """Run one v3 image-only investigation.

        Non-image-only runtime inputs are intentionally unsupported. When no case is
        supplied, the workflow constructs the three-field public case locally.
        """
        if runtime_case is None:
            resolved_image_path = os.path.abspath(image_path)
            runtime_case = ImageOnlyRuntimeCase(
                case_id=image_id or os.path.basename(image_path) or image_path,
                image_path=resolved_image_path,
                image_sha256=image_sha256(resolved_image_path),
            )
            image_path = resolved_image_path
        elif not isinstance(runtime_case, ImageOnlyRuntimeCase):
            raise TypeError(
                "v3 accepts ImageOnlyRuntimeCase only"
            )

        orchestrator = self._get_orchestrator(validate_startup=False)
        runtime_store = CaseRuntimeStore(
            self.config.output_dir,
            case_id=runtime_case.case_id,
        )
        runtime_token = bind_case_runtime_store(runtime_store)
        try:
            verify_case_image(runtime_case, image_path)
            result = await orchestrator.run(
                image_path,
                runtime_case,
                decision_policy_version=self.config.decision_policy_version,
            )
        except Exception as exc:
            state = getattr(orchestrator, "last_state", None)
            error_result: Optional[Dict[str, Any]] = None
            if state is not None:
                error_result = {
                    "image_id": state.image_id,
                    "image_path": state.image_path,
                    "verdict": "error",
                    "confidence": 0.0,
                    "termination": state.termination or "error",
                    "error": " | ".join(state.errors) or str(exc),
                    "time_taken": state.stage_timings.get("total", 0.0),
                    "total_tool_calls": state.total_tool_calls,
                    "llm_api_calls": state.llm_api_calls,
                    "token_usage": state.token_usage,
                    "state": state.to_dict(),
                }
            if self.config.save_traces and state is not None:
                runtime_store.write_snapshot("engineering_error", state.to_dict())
                self._save_trace(
                    {
                        "image_id": state.image_id,
                        "image_path": state.image_path,
                        "input_mode": state.input_mode,
                        "decision_policy_version": state.decision_policy_version,
                        "verdict": "error",
                        "confidence": 0.0,
                        "overall_assessment": "Verification terminated with an engineering error.",
                        "state": state.to_dict(),
                        "termination": "error",
                        "time_taken": state.stage_timings.get("total", 0.0),
                        "token_usage": state.token_usage,
                        "total_tool_calls": state.total_tool_calls,
                        "llm_api_calls": state.llm_api_calls,
                        "error": " | ".join(state.errors),
                    }
                )
            if error_result is not None:
                setattr(exc, "_ifv_result", error_result)
            raise
        finally:
            reset_case_runtime_store(runtime_token)

        # Save trace if configured
        if self.config.save_traces:
            runtime_store.write_snapshot("final_state", result.get("state", {}))
            self._save_trace(result)

        return result

    async def run_batch(
        self,
        image_paths: List[str],
        image_ids: Optional[List[str]] = None,
        runtime_cases: Optional[List[Optional[ImageOnlyRuntimeCase]]] = None,
        concurrency: int = 1,
    ) -> List[Dict[str, Any]]:
        """Run verification on multiple images.

        Args:
            image_paths: List of image file paths.
            image_ids: Optional list of identifiers.
            concurrency: Max concurrent verifications.

        Returns:
            List of results.
        """
        if image_ids is None:
            image_ids = [os.path.basename(p) for p in image_paths]
        if runtime_cases is None:
            runtime_cases = [None] * len(image_paths)
        if len(runtime_cases) != len(image_paths):
            raise ValueError("runtime_cases must match image_paths length")

        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def _verify(
            path: str,
            img_id: str,
            runtime_case: Optional[ImageOnlyRuntimeCase],
        ) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.run_single(
                        path,
                        img_id,
                        runtime_case=runtime_case,
                    )
                except Exception as exc:
                    error_result = getattr(exc, "_ifv_result", None)
                    if isinstance(error_result, dict):
                        return error_result
                    raise

        tasks = [
            _verify(path, img_id, runtime_case)
            for path, img_id, runtime_case in zip(
                image_paths,
                image_ids,
                runtime_cases,
            )
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error dicts
        final = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                final.append({
                    "image_id": image_ids[i],
                    "image_path": image_paths[i],
                    "verdict": "error",
                    "termination": "error",
                    "error": str(r),
                })
            else:
                final.append(r)

        return final

    def _save_trace(self, result: Dict[str, Any]) -> None:
        """Save the canonical verification trace to the output directory."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        image_id = result.get("image_id", "unknown")
        # Sanitize filename
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in image_id)
        trace_path = os.path.join(self.config.output_dir, f"{safe_id}.json")

        # Remove non-serializable fields
        serializable = sanitize_for_persistence({
            k: v for k, v in result.items()
            if k != "state" or isinstance(v, dict)
        })
        serializable.setdefault("schema_version", CANONICAL_TRACE_SCHEMA_VERSION)
        atomic_write_json(trace_path, serializable)

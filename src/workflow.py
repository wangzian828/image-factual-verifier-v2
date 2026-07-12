# -*- coding: utf-8 -*-
"""Verification workflow: the single entry point for running image verification.

Usage:
    from src.workflow import VerificationWorkflow, WorkflowConfig

    workflow = VerificationWorkflow(WorkflowConfig())
    result = await workflow.run_single("path/to/image.jpg")
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Limit thread usage to prevent memory explosion
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

load_dotenv()

from src.orchestrator.ledger import build_verification_case
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.state import VerificationCase
from src.orchestrator.source_access import SourceAccessPolicy
from src.redaction import sanitize_for_persistence
from src.storage import default_trace_dir


@dataclass
class WorkflowConfig:
    """Configuration for the verification workflow."""

    # LLM settings
    provider: str = "gemini"
    model_name: str = "gemini-3.5-flash"
    vlm_provider: Optional[str] = None  # Defaults to provider
    vlm_model: Optional[str] = None  # Defaults to model_name
    llm_wire_api: Optional[str] = None
    vlm_wire_api: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 8192

    # Stage settings
    max_rounds_verification: int = 12
    max_verification_iterations: Optional[int] = None
    min_verification_iterations: Optional[int] = None
    low_information_gain_patience: Optional[int] = None
    timeout: float = 1800.0

    # Output
    output_dir: str = field(default_factory=default_trace_dir)
    save_traces: bool = True
    source_access_policy: Optional[SourceAccessPolicy] = None


class VerificationWorkflow:
    """Main workflow for image factual verification."""

    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self._orchestrator: Optional[Orchestrator] = None

    def _get_orchestrator(self) -> Orchestrator:
        """Lazy init orchestrator."""
        if self._orchestrator is None:
            self._orchestrator = Orchestrator(
                provider=self.config.provider,
                model_name=self.config.model_name,
                vlm_provider=self.config.vlm_provider,
                vlm_model=self.config.vlm_model,
                llm_wire_api=self.config.llm_wire_api,
                vlm_wire_api=self.config.vlm_wire_api,
                max_rounds_verification=self.config.max_rounds_verification,
                max_verification_iterations=self.config.max_verification_iterations,
                min_verification_iterations=self.config.min_verification_iterations,
                low_information_gain_patience=self.config.low_information_gain_patience,
                timeout=self.config.timeout,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                source_access_policy=self.config.source_access_policy,
            )
        return self._orchestrator

    async def run_single(
        self,
        image_path: str,
        image_id: str = "",
        *,
        user_claim: Optional[str] = None,
        claim_observed_at: Optional[str] = None,
        verification_case: Optional[VerificationCase] = None,
    ) -> Dict[str, Any]:
        """Run verification on a single image.

        Args:
            image_path: Path to the image file.
            image_id: Optional identifier.

        Returns:
            Dict with verdict, confidence, assessment, and full state.
        """
        orchestrator = self._get_orchestrator()
        try:
            if verification_case is None and claim_observed_at is not None:
                verification_case = build_verification_case(
                    image_path,
                    case_id=image_id or os.path.basename(image_path) or image_path,
                    user_claim=user_claim,
                    claim_observed_at=claim_observed_at,
                )
            if verification_case is None and user_claim is None:
                result = await orchestrator.run(image_path, image_id)
            else:
                result = await orchestrator.run(
                    image_path,
                    image_id,
                    verification_case=verification_case,
                    user_claim=user_claim,
                )
        except Exception:
            state = getattr(orchestrator, "last_state", None)
            if self.config.save_traces and state is not None:
                self._save_trace(
                    {
                        "image_id": state.image_id,
                        "image_path": state.image_path,
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
            raise

        # Save trace if configured
        if self.config.save_traces:
            self._save_trace(result)

        return result

    async def run_batch(
        self,
        image_paths: List[str],
        image_ids: Optional[List[str]] = None,
        user_claims: Optional[List[Optional[str]]] = None,
        claim_observed_ats: Optional[List[Optional[str]]] = None,
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
        if user_claims is None:
            user_claims = [None] * len(image_paths)
        if len(user_claims) != len(image_paths):
            raise ValueError("user_claims must match image_paths length")
        if claim_observed_ats is None:
            claim_observed_ats = [None] * len(image_paths)
        if len(claim_observed_ats) != len(image_paths):
            raise ValueError("claim_observed_ats must match image_paths length")

        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def _verify(
            path: str,
            img_id: str,
            claim: Optional[str],
            claim_observed_at: Optional[str],
        ) -> Dict[str, Any]:
            async with semaphore:
                return await self.run_single(
                    path,
                    img_id,
                    user_claim=claim,
                    claim_observed_at=claim_observed_at,
                )

        tasks = [
            _verify(path, img_id, claim, claim_observed_at)
            for path, img_id, claim, claim_observed_at in zip(
                image_paths, image_ids, user_claims, claim_observed_ats
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

        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)

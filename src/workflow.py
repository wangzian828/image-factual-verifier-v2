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

from src.orchestrator.pipeline import Orchestrator


@dataclass
class WorkflowConfig:
    """Configuration for the verification workflow."""

    # LLM settings
    provider: str = "necodex"
    model_name: str = "gpt-5.5"
    vlm_provider: Optional[str] = None  # Defaults to provider
    vlm_model: Optional[str] = None  # Defaults to model_name
    temperature: float = 0.0
    max_tokens: int = 8192

    # Stage settings
    max_rounds_perception: int = 3
    max_rounds_verification: int = 8
    timeout: float = 300.0

    # Output
    output_dir: str = "outputs/traces"
    save_traces: bool = True


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
                max_rounds_perception=self.config.max_rounds_perception,
                max_rounds_verification=self.config.max_rounds_verification,
                timeout=self.config.timeout,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        return self._orchestrator

    async def run_single(self, image_path: str, image_id: str = "") -> Dict[str, Any]:
        """Run verification on a single image.

        Args:
            image_path: Path to the image file.
            image_id: Optional identifier.

        Returns:
            Dict with verdict, confidence, assessment, and full state.
        """
        orchestrator = self._get_orchestrator()
        result = await orchestrator.run(image_path, image_id)

        # Save trace if configured
        if self.config.save_traces:
            self._save_trace(result)

        return result

    async def run_batch(
        self,
        image_paths: List[str],
        image_ids: Optional[List[str]] = None,
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

        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def _verify(path: str, img_id: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.run_single(path, img_id)

        tasks = [
            _verify(path, img_id)
            for path, img_id in zip(image_paths, image_ids)
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
        """Save verification trace to output directory."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        image_id = result.get("image_id", "unknown")
        # Sanitize filename
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in image_id)
        trace_path = os.path.join(self.config.output_dir, f"{safe_id}.json")

        # Remove non-serializable fields
        serializable = {
            k: v for k, v in result.items()
            if k != "state" or isinstance(v, dict)
        }

        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)

# -*- coding: utf-8 -*-
"""Context rendering and compression for each pipeline stage.

Handles:
- Converting stage outputs to compact text for the next stage's input
- Compressing tool results (search results, VLM outputs)
- Building the input_context string for each stage
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.orchestrator.state import (
    EvidenceItem,
    PerceptionReport,
    VerificationPlan,
    VerificationResult,
)


class ContextRenderer:
    """Renders inter-stage context: converts structured data to compact text."""

    # --- Stage 2 (Planning) input ---

    @staticmethod
    def render_for_planning(perception: PerceptionReport) -> str:
        """Render PerceptionReport as compact text for the Planning stage."""
        parts = ["## Perception Report\n"]

        # Entities (top 10)
        if perception.entities:
            parts.append("### Entities")
            for i, ent in enumerate(perception.entities[:10]):
                attrs = ", ".join(f"{k}={v}" for k, v in ent.attributes.items()) if ent.attributes else ""
                bbox_str = f" bbox={ent.bbox}" if ent.bbox else ""
                parts.append(
                    f"  {i}. [{ent.entity_type}] {ent.name}{bbox_str}"
                    + (f" ({attrs})" if attrs else "")
                )

        # Text regions
        if perception.text_regions:
            parts.append("\n### Text Found")
            for tr in perception.text_regions:
                conf = f" (conf={tr.confidence:.2f})" if tr.confidence > 0 else ""
                parts.append(f'  - "{tr.text}" [{tr.language}]{conf}')

        # Faces
        if perception.faces:
            parts.append(f"\n### Faces Detected: {len(perception.faces)}")
            for i, face in enumerate(perception.faces):
                parts.append(
                    f"  {i}. age≈{face.age}, {face.gender}, conf={face.confidence:.2f}"
                )

        # Scene
        parts.append(f"\n### Scene: {perception.scene_description}")
        parts.append(f"### Image Type: {perception.image_type}")

        return "\n".join(parts)

    # --- Stage 3 (Verification) input ---

    @staticmethod
    def render_for_verification(
        perception: PerceptionReport,
        plan: VerificationPlan,
    ) -> str:
        """Render context for the Verification stage."""
        parts = ["## Context for Verification\n"]

        # Compact perception summary
        parts.append("### Image Content")
        parts.append(f"Type: {perception.image_type}")
        parts.append(f"Scene: {perception.scene_description}")

        if perception.entities:
            entity_names = [e.name for e in perception.entities[:8]]
            parts.append(f"Key entities: {', '.join(entity_names)}")

        if perception.text_regions:
            texts = [tr.text for tr in perception.text_regions[:5]]
            parts.append(f"Text found: {'; '.join(texts)}")

        if perception.faces:
            parts.append(f"Faces: {len(perception.faces)} detected")

        # Verification plan
        parts.append("\n### Verification Plan")
        parts.append(f"Image intent: {plan.image_intent}")
        parts.append(f"Trying to be real: {plan.is_trying_to_be_real}")
        parts.append(f"Risk: {plan.risk_assessment}")

        parts.append("\n### Questions to Investigate")
        for q in plan.questions:
            priority_mark = "⚡" if q.priority == 1 else ("•" if q.priority == 2 else "○")
            parts.append(f"  {priority_mark} [{q.question_id}] {q.question}")
            if q.suggested_queries:
                parts.append(f"    Suggested queries: {q.suggested_queries[:3]}")
            if q.suggested_tools:
                parts.append(f"    Suggested tools: {q.suggested_tools}")

        return "\n".join(parts)

    # --- Stage 4 (Judgment) input ---

    @staticmethod
    def render_for_judgment(
        perception: PerceptionReport,
        plan: VerificationPlan,
        verification: VerificationResult,
    ) -> str:
        """Render context for the Judgment stage."""
        parts = ["## Evidence Summary for Final Judgment\n"]

        # Image basics
        parts.append(f"Image type: {perception.image_type}")
        parts.append(f"Scene: {perception.scene_description}")
        parts.append(f"Image intent: {plan.image_intent}")
        parts.append(f"Is trying to be real: {plan.is_trying_to_be_real}")
        parts.append(f"Risk assessment: {plan.risk_assessment}")

        # Evidence collected
        if verification.evidence:
            parts.append("\n### Evidence Collected")
            for ev in verification.evidence:
                direction_icon = {"supports": "✓", "refutes": "✗", "neutral": "—"}.get(ev.direction, "?")
                parts.append(
                    f"  {direction_icon} [{ev.quality}] {ev.summary} (via {ev.tool_used}, re: {ev.related_question})"
                )

        # Visual anomalies
        if verification.visual_anomalies:
            parts.append("\n### Visual Anomalies Found")
            for anom in verification.visual_anomalies:
                if isinstance(anom, dict):
                    parts.append(f"  - {anom.get('name', 'unnamed')}: {anom.get('phenomenon', '')}")

        # Key findings
        if verification.key_findings:
            parts.append("\n### Key Findings")
            for finding in verification.key_findings:
                parts.append(f"  - {finding}")

        # Authenticity
        parts.append(f"\n### Authenticity Assessment: {verification.authenticity_assessment}")

        # Questions and what was found
        parts.append("\n### Investigation Questions Status")
        answered_qs = set(ev.related_question for ev in verification.evidence if ev.related_question)
        for q in plan.questions:
            status = "answered" if q.question_id in answered_qs else "unanswered"
            parts.append(f"  [{status}] {q.question}")

        return "\n".join(parts)

    # --- Tool result compression ---

    @staticmethod
    def compress_search_result(raw_result: str, max_items: int = 5) -> str:
        """Compress search tool results to top-N with 1-sentence summaries."""
        try:
            data = json.loads(raw_result)
        except (json.JSONDecodeError, TypeError):
            # Not JSON, truncate raw text
            return raw_result[:2000] if len(raw_result) > 2000 else raw_result

        if isinstance(data, list):
            # List of search results
            compressed = []
            for item in data[:max_items]:
                if isinstance(item, dict):
                    title = item.get("title", "")
                    snippet = item.get("snippet", item.get("description", ""))[:150]
                    url = item.get("url", item.get("link", ""))
                    compressed.append(f"- {title}: {snippet} [{url}]")
            return "\n".join(compressed) if compressed else raw_result[:1000]

        if isinstance(data, dict):
            # Single result or wrapped results
            results = data.get("results", data.get("organic", []))
            if isinstance(results, list):
                compressed = []
                for item in results[:max_items]:
                    if isinstance(item, dict):
                        title = item.get("title", "")
                        snippet = item.get("snippet", "")[:150]
                        compressed.append(f"- {title}: {snippet}")
                return "\n".join(compressed) if compressed else json.dumps(data, ensure_ascii=False)[:1000]

        return raw_result[:2000]

    @staticmethod
    def compress_visit_result(raw_result: str, max_chars: int = 2000) -> str:
        """Truncate visit (web page) results."""
        if len(raw_result) <= max_chars:
            return raw_result
        return raw_result[:max_chars] + "\n... (truncated)"

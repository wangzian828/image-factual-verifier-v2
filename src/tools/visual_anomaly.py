# -*- coding: utf-8 -*-
"""Visual Anomaly Detection Tool.

Analyzes images for semantic/physical/logical anomalies that indicate
AI generation, manipulation, or inconsistencies. Supports two modes:
- Targeted analysis (with focus_areas from search/perception) — high accuracy
- Broad scan (no focus) — lower accuracy but wider coverage

References AnomAgent (ICLR 2026) pipeline design but simplified to a single
VLM call with structured prompting.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool


# --- Prompts ---

TARGETED_ANALYSIS_PROMPT = """\
You are an expert image forensics analyst. Analyze this image for visual anomalies \
based on the specific focus areas and context provided.

## Focus Areas
{focus_areas}

## Context (what we already know)
{context}

## Check Type: {check_type}

## Instructions
For each focus area, carefully examine the image and determine:
1. Identify the specific entities involved (objects, people, structures)
2. Check the RELATIONSHIP between these entities — contact points, support, deformation, grip
3. Is there a visual anomaly or inconsistency in how they interact?
4. What exactly is wrong (describe the phenomenon in terms of entity interactions)?
5. Why is it anomalous (reasoning based on physics, anatomy, common sense)?
6. How severe is it (0-100, where 100 = completely implausible)?

## Output Format
Return a JSON object:
{{
  "anomalies": [
    {{
      "name": "short descriptive name",
      "region": "where in the image",
      "phenomenon": "what you observe — describe specific entity interactions that are wrong",
      "reasoning": "why this is anomalous (physics/anatomy/common sense)",
      "severity": 0-100,
      "type": "ai_generation | manipulation | physical_inconsistency | logical_inconsistency",
      "entities_involved": ["entity A", "entity B"]
    }}
  ],
  "overall_authenticity": "authentic | likely_ai | likely_manipulated | uncertain",
  "confidence": 0.0-1.0,
  "notes": "any additional observations"
}}

IMPORTANT:
- Focus on RELATIONSHIPS between entities, not just individual objects.
- Describe anomalies in terms of specific entity interactions.
- If no anomalies are found, return an empty anomalies list with overall_authenticity="authentic".
- Only report anomalies you are genuinely confident about.
"""

BROAD_SCAN_PROMPT = """\
You are an expert image forensics analyst. Perform a comprehensive anomaly scan \
on this image to determine if it is authentic, AI-generated, or manipulated.

## Analysis Steps (follow in order)

### Step 1: Entity Inventory
List ALL major entities (objects, people, text, structures) in the image with their key attributes:
- Name/description
- Position in image
- Key visual attributes (size, color, material, state)

### Step 2: Attribute Anomaly Check
For EACH entity, verify its internal consistency:
- Anatomy: correct number of fingers, limbs, proportions, joints
- Material/texture: consistent surface properties
- Shape/structure: physically plausible geometry
- Text: readable, correctly formed characters

### Step 3: Relationship Anomaly Check (CRITICAL)
For EACH PAIR of interacting entities, verify their relationship is physically and logically consistent:
- Does the interaction between them obey basic physics and common sense?
- Are the visual consequences of their interaction correctly depicted?
- Would a real-world version of this interaction look the same?

### Step 4: Global Consistency
- Perspective/vanishing points consistent?
- Lighting direction uniform?
- Background coherent (no repeating patterns, warping)?

## Check Type: {check_type}

## Output Format
Return a JSON object:
{{
  "anomalies": [
    {{
      "name": "short descriptive name",
      "region": "where in the image",
      "phenomenon": "what you observe that is wrong — describe the specific entities involved and their relationship",
      "reasoning": "why this is anomalous — reference physics, anatomy, or common sense",
      "severity": 0-100,
      "type": "ai_generation | manipulation | physical_inconsistency | logical_inconsistency",
      "entities_involved": ["entity A", "entity B"]
    }}
  ],
  "overall_authenticity": "authentic | likely_ai | likely_manipulated | uncertain",
  "confidence": 0.0-1.0,
  "notes": "any additional observations"
}}

IMPORTANT:
- Focus on RELATIONSHIPS between entities, not just individual objects.
- Describe anomalies in terms of "entity A vs entity B" interactions.
- Be conservative. Only report anomalies you are genuinely confident about.
- Do NOT hallucinate issues — if the image looks authentic, say so.
"""


@dataclass
class VisualAnomalyTool(BaseTool):
    """Analyze image for visual anomalies indicating AI generation or manipulation.

    Two modes:
    - Targeted (focus_areas provided): high accuracy, checks specific aspects
    - Broad scan (no focus_areas): lower accuracy, comprehensive check
    """

    name: str = "analyze_visual_anomalies"
    description: str = (
        "Analyze the image for visual anomalies that indicate AI generation, "
        "manipulation, or physical/logical inconsistencies. "
        "Pass focus_areas for targeted high-accuracy analysis, or leave empty for broad scan. "
        "Pass context with what you already know from search results to guide the analysis."
    )
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "focus_areas": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Specific areas/aspects to check. Examples: "
                    "'check if the third spire is consistent with the other two', "
                    "'verify hand anatomy of person on the left', "
                    "'check shadow directions for all objects'. "
                    "If empty, performs a broad anomaly scan."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "What you already know from search/perception that should guide analysis. "
                    "Example: 'Search results say this tower has 2 spires, but image shows 3.'"
                ),
            },
            "check_type": {
                "type": "string",
                "enum": ["ai_generation", "manipulation", "physical_consistency", "all"],
                "description": "What type of anomalies to focus on. Default: all.",
            },
        },
        "required": [],
    })

    # Injected at construction time
    vlm_backend: Any = None  # LLMBackend instance
    image_path: str = ""  # Set per-run by harness

    def call(self, params: Dict[str, Any]) -> Any:
        """Synchronous entry point — runs async code in a new event loop."""
        import asyncio
        try:
            # If there's already a running loop (e.g., called from SyncToolWrapper
            # in a thread), create a new loop for this thread
            loop = asyncio.get_running_loop()
            # We're in an async context — shouldn't happen via SyncToolWrapper
            # but handle gracefully
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(asyncio.run, self.call_async(params))
                return future.result(timeout=120)
        except RuntimeError:
            # No running loop — normal case when called from SyncToolWrapper thread
            return asyncio.run(self.call_async(params))

    async def call_async(self, params: Dict[str, Any]) -> str:
        """Async implementation — calls VLM with image + analysis prompt."""
        focus_areas = params.get("focus_areas", [])
        context = params.get("context", "")
        check_type = params.get("check_type", "all")

        if not self.vlm_backend:
            return json.dumps({
                "error": "VLM backend not configured for visual anomaly analysis.",
                "anomalies": [],
                "overall_authenticity": "uncertain",
                "confidence": 0.0,
            }, ensure_ascii=False)

        # Build prompt based on mode
        if focus_areas:
            prompt = TARGETED_ANALYSIS_PROMPT.format(
                focus_areas="\n".join(f"- {fa}" for fa in focus_areas),
                context=context or "No additional context provided.",
                check_type=check_type,
            )
        else:
            prompt = BROAD_SCAN_PROMPT.format(check_type=check_type)
            if context:
                prompt += f"\n\n## Additional Context\n{context}"

        # Build messages with image
        from src.tools.vision_utils import image_to_data_url
        image_data_url = image_to_data_url(self.image_path)

        messages = [
            {"role": "system", "content": "You are an expert image forensics analyst. Always respond with valid JSON."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": prompt},
            ]},
        ]

        # Call VLM
        response = await self.vlm_backend.get_response(messages, max_tokens=4096)
        content = response.text

        if not content:
            return json.dumps({
                "anomalies": [],
                "overall_authenticity": "uncertain",
                "confidence": 0.0,
                "error": "VLM returned empty response.",
            }, ensure_ascii=False)

        # Parse JSON from response
        result = self._parse_response(content)
        return self._format_output(result, focus_areas)

    def _call_sync(self, params: Dict[str, Any]) -> str:
        """Sync fallback."""
        import asyncio
        return asyncio.run(self.call_async(params))

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON from VLM response, handling markdown code blocks."""
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in text
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: return raw text as notes
        return {
            "anomalies": [],
            "overall_authenticity": "uncertain",
            "confidence": 0.0,
            "notes": f"Could not parse VLM response. Raw: {content[:500]}",
        }

    def _format_output(self, result: Dict[str, Any], focus_areas: List[str]) -> str:
        """Format the analysis result as a readable string for the agent."""
        anomalies = result.get("anomalies", [])
        authenticity = result.get("overall_authenticity", "uncertain")
        confidence = result.get("confidence", 0.0)
        notes = result.get("notes", "")

        lines = []
        lines.append(f"=== Visual Anomaly Analysis ===")
        lines.append(f"Mode: {'Targeted' if focus_areas else 'Broad scan'}")
        lines.append(f"Overall authenticity: {authenticity} (confidence: {confidence:.1%})")
        lines.append("")

        if anomalies:
            lines.append(f"Found {len(anomalies)} anomaly(ies):")
            for i, a in enumerate(anomalies, 1):
                lines.append(f"  [{i}] {a.get('name', 'Unknown')}")
                lines.append(f"      Region: {a.get('region', 'N/A')}")
                lines.append(f"      Phenomenon: {a.get('phenomenon', 'N/A')}")
                lines.append(f"      Reasoning: {a.get('reasoning', 'N/A')}")
                lines.append(f"      Severity: {a.get('severity', 0)}/100")
                lines.append(f"      Type: {a.get('type', 'unknown')}")
                lines.append("")
        else:
            lines.append("No anomalies detected.")

        if notes:
            lines.append(f"Notes: {notes}")

        # Also append raw JSON for structured processing
        lines.append("")
        lines.append(f"[RAW_JSON]{json.dumps(result, ensure_ascii=False)}[/RAW_JSON]")

        return "\n".join(lines)

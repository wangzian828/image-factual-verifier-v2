# -*- coding: utf-8 -*-
"""Compare With Reference Tool.

Downloads a reference image (found via search) and compares it with the
current image to identify differences that indicate manipulation.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool


COMPARE_PROMPT = """\
You are an expert image forensics analyst. Compare these two images to identify \
differences that may indicate manipulation or editing.

## Reference Image (Image 1)
This is the reference/original image found from a trusted source.

## Current Image (Image 2)
This is the image being verified.

## Focus
{focus}

## Instructions
1. Identify all visual differences between the two images.
2. For each difference, determine if it indicates:
   - Cropping/resizing (benign)
   - Color/brightness adjustment (benign)
   - Content addition (manipulation — something added that wasn't in original)
   - Content removal (manipulation — something removed)
   - Content modification (manipulation — something changed)
   - Complete replacement (different image entirely)
3. Assess whether the current image is a manipulated version of the reference.

## Output Format
Return a JSON object:
{{
  "is_same_scene": true/false,
  "differences": [
    {{
      "region": "where in the image",
      "description": "what is different",
      "type": "crop | color_adjust | addition | removal | modification | unrelated",
      "significance": "high | medium | low"
    }}
  ],
  "manipulation_verdict": "authentic_copy | benign_edit | manipulated | unrelated_images",
  "manipulation_details": "explanation of what was manipulated and how",
  "confidence": 0.0-1.0
}}

Be precise. If the images are simply different photos of the same subject, say so.
Only flag manipulation if there is clear evidence of editing.
"""


@dataclass
class CompareWithReferenceTool(BaseTool):
    """Compare current image with a reference image to find manipulation evidence."""

    name: str = "compare_with_reference"
    description: str = (
        "Compare the current image with a reference image (found via search) "
        "to identify differences, additions, or removals that indicate manipulation. "
        "Use after reverse_image_search finds a similar/original image."
    )
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "reference_url": {
                "type": "string",
                "description": "URL of the reference/original image to compare against.",
            },
            "focus": {
                "type": "string",
                "description": (
                    "What to focus the comparison on. "
                    "E.g., 'number of spires on the tower', 'person's face on the left'."
                ),
            },
        },
        "required": ["reference_url"],
    })

    # Injected at construction time
    vlm_backend: Any = None  # LLMBackend instance
    image_path: str = ""  # Set per-run by harness

    def call(self, params: Dict[str, Any]) -> Any:
        """Synchronous entry point — runs async code in a new event loop."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(asyncio.run, self.call_async(params))
                return future.result(timeout=120)
        except RuntimeError:
            return asyncio.run(self.call_async(params))

    async def call_async(self, params: Dict[str, Any]) -> str:
        """Async implementation — downloads reference and compares."""
        reference_url = params.get("reference_url", "")
        focus = params.get("focus", "general comparison")

        if not reference_url:
            return "Error: reference_url is required."

        if not self.vlm_backend:
            return "Error: VLM backend not configured for image comparison."

        # Download reference image
        reference_data_url = await self._download_reference(reference_url)
        if not reference_data_url:
            return f"Error: Could not download reference image from {reference_url}"

        # Build current image data URL
        from src.tools.vision_utils import image_to_data_url
        current_data_url = image_to_data_url(self.image_path)

        # Build comparison prompt
        prompt = COMPARE_PROMPT.format(focus=focus)

        messages = [
            {"role": "system", "content": "You are an expert image forensics analyst. Always respond with valid JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": "Image 1 (Reference):"},
                {"type": "image_url", "image_url": {"url": reference_data_url}},
                {"type": "text", "text": "Image 2 (Current - being verified):"},
                {"type": "image_url", "image_url": {"url": current_data_url}},
                {"type": "text", "text": prompt},
            ]},
        ]

        # Call VLM
        response = await self.vlm_backend.get_response(messages, max_tokens=4096)
        content = response.text

        if not content:
            return "Error: VLM returned empty response for comparison."

        # Parse and format
        result = self._parse_response(content)
        return self._format_output(result, reference_url)

    def _call_sync(self, params: Dict[str, Any]) -> str:
        """Sync fallback."""
        import asyncio
        return asyncio.run(self.call_async(params))

    async def _download_reference(self, url: str) -> Optional[str]:
        """Download reference image and convert to data URL."""
        import base64
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return None

                content_type = resp.headers.get("content-type", "image/jpeg")
                if ";" in content_type:
                    content_type = content_type.split(";")[0].strip()

                # Validate it's an image
                if not content_type.startswith("image/"):
                    content_type = "image/jpeg"  # Assume JPEG

                img_b64 = base64.b64encode(resp.content).decode()
                return f"data:{content_type};base64,{img_b64}"

        except Exception:
            return None

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON from VLM response."""
        import re

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {
            "is_same_scene": False,
            "differences": [],
            "manipulation_verdict": "uncertain",
            "manipulation_details": f"Could not parse response. Raw: {content[:300]}",
            "confidence": 0.0,
        }

    def _format_output(self, result: Dict[str, Any], reference_url: str) -> str:
        """Format comparison result as readable string."""
        differences = result.get("differences", [])
        verdict = result.get("manipulation_verdict", "uncertain")
        details = result.get("manipulation_details", "")
        confidence = result.get("confidence", 0.0)
        is_same = result.get("is_same_scene", False)

        lines = []
        lines.append("=== Image Comparison Analysis ===")
        lines.append(f"Reference: {reference_url[:80]}")
        lines.append(f"Same scene: {'Yes' if is_same else 'No'}")
        lines.append(f"Verdict: {verdict} (confidence: {confidence:.1%})")
        lines.append("")

        if differences:
            lines.append(f"Found {len(differences)} difference(s):")
            for i, d in enumerate(differences, 1):
                sig = d.get("significance", "?")
                lines.append(f"  [{i}] [{sig.upper()}] {d.get('description', 'N/A')}")
                lines.append(f"      Region: {d.get('region', 'N/A')}")
                lines.append(f"      Type: {d.get('type', 'N/A')}")
                lines.append("")
        else:
            lines.append("No significant differences found.")

        if details:
            lines.append(f"Details: {details}")

        lines.append("")
        lines.append(f"[RAW_JSON]{json.dumps(result, ensure_ascii=False)}[/RAW_JSON]")

        return "\n".join(lines)

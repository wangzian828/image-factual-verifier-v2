# -*- coding: utf-8 -*-
"""Question-driven multi-view inspection of the original input image."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.integrations.gemini import RUNTIME_METRICS_KEY, exception_runtime_metrics
from src.tools.base import BaseTool


FOCUSED_VISUAL_INSPECTION_PROMPT = """\
You are the visual observation stage of an image fact-search agent.

Inspect the supplied views of one original input image to answer one focused
visual question that arose during web investigation. The first image is always
the complete original. Later images are deterministic detail views derived from
pixel/OCR anchors in that same original.

Treat names, identities, places, events, and expected properties in the request
as hypotheses, not as facts. Report only what the supplied pixels show. Separate
literal observations from interpretation, preserve ambiguity, and do not use
outside knowledge, search memory, source reputation, or the surrounding factual
claim to fill missing visual details. Do not decide the benchmark verdict.

Return:
- answer_status=observed only when the expected visible property is actually
  present in the supplied pixels;
- answer_status=not_observed only when the relevant region is visible enough and
  materially incompatible with that property;
- answer_status=ambiguous when resolution, occlusion, framing, or visual
  similarity prevents a reliable answer.

Every observation must identify the supplied view where it is visible.
"""


FOCUSED_VISUAL_INSPECTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_status": {
            "type": "string",
            "enum": ["observed", "not_observed", "ambiguous"],
        },
        "summary": {"type": "string", "maxLength": 1200},
        "observations": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "view_index": {"type": "integer"},
                    "statement": {"type": "string", "maxLength": 600},
                    "property_status": {
                        "type": "string",
                        "enum": ["observed", "not_observed", "ambiguous"],
                    },
                    "confidence": {"type": "number"},
                },
            },
        },
        "limitations": {
            "type": "array",
            "maxItems": 6,
            "items": {"type": "string", "maxLength": 400},
        },
    },
}


@dataclass
class FocusedVisualInspectionTool(BaseTool):
    """Inspect the original image and deterministic anchor views as one packet."""

    name: str = "focused_visual_inspection"
    description: str = (
        "Reinspect the complete original image and deterministic anchor-derived "
        "detail views to answer one focused visual question discovered during "
        "the investigation."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {
                    "type": "string",
                    "description": "Local path to the original input image.",
                },
                "visual_question_id": {"type": "string"},
                "question": {"type": "string"},
                "expected_property": {"type": "string"},
                "scope": {
                    "type": "string",
                    "enum": [
                        "subject",
                        "relation",
                        "scene",
                        "text",
                        "integrity",
                    ],
                },
                "anchor_regions": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                "active_fact": {"type": "string"},
                "evidence_context": {"type": "string"},
            },
            "required": [
                "image_input",
                "visual_question_id",
                "question",
                "expected_property",
                "scope",
                "anchor_regions",
                "active_fact",
                "evidence_context",
            ],
        }
    )

    client: Optional[Any] = field(default=None, repr=False)
    provider: str = "gemini"
    model_name: str = "gemini-3.5-flash"

    def _get_client(self) -> Any:
        if self.client is None:
            from src.integrations.vlm.factory import build_vlm_client

            self.client = build_vlm_client(
                provider=self.provider,
                model_name=self.model_name,
            )
        method = getattr(self.client, "create_images_json", None)
        if not callable(method):
            raise RuntimeError(
                "The configured visual model client does not implement the "
                "multi-view create_images_json contract."
            )
        return self.client

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_path = str(params.get("image_input", "")).strip()
        if not image_path:
            return {"status": "error", "error": "image_input is required."}
        try:
            anchor_regions = _normalize_regions(params.get("anchor_regions", []))
            image_inputs, views, temporary_paths = _prepare_views(
                image_path,
                anchor_regions,
                scope=str(params.get("scope", "scene")).strip(),
            )
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Visual view preparation failed: {type(exc).__name__}: {exc}",
            }

        request_context = {
            "visual_question_id": str(params.get("visual_question_id", "")).strip(),
            "question": str(params.get("question", "")).strip(),
            "expected_property": str(params.get("expected_property", "")).strip(),
            "scope": str(params.get("scope", "")).strip(),
            "active_fact": str(params.get("active_fact", "")).strip(),
            "evidence_context": str(params.get("evidence_context", "")).strip(),
            "views": views,
        }
        try:
            parsed = self._get_client().create_images_json(
                system_prompt=FOCUSED_VISUAL_INSPECTION_PROMPT,
                user_text=(
                    "Answer the focused visual question using the supplied views "
                    "in their listed order.\n\n"
                    + json.dumps(request_context, ensure_ascii=False, indent=2)
                ),
                image_inputs=image_inputs,
                max_tokens=2400,
                model_name=self.model_name,
                response_schema=FOCUSED_VISUAL_INSPECTION_SCHEMA,
            )
            normalized = _normalize_model_output(parsed, views)
        except Exception as exc:
            error = {
                "status": "error",
                "error": (
                    "Focused visual inspection failed: "
                    f"{type(exc).__name__}: {exc or '<no message>'}"
                ),
            }
            metrics = exception_runtime_metrics(exc)
            if metrics:
                error[RUNTIME_METRICS_KEY] = metrics
            return error
        finally:
            for path in temporary_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

        return {
            "status": "success",
            "visual_question_id": request_context["visual_question_id"],
            "question": request_context["question"],
            "expected_property": request_context["expected_property"],
            "scope": request_context["scope"],
            "answer_status": normalized["answer_status"],
            "summary": normalized["summary"],
            "observations": normalized["observations"],
            "limitations": normalized["limitations"],
            "views": views,
            RUNTIME_METRICS_KEY: parsed.get(RUNTIME_METRICS_KEY, {}),
        }


def _normalize_regions(value: Any) -> List[List[float]]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise ValueError("anchor_regions must be a list of normalized bboxes")
    regions: List[List[float]] = []
    for raw in value:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise ValueError("every anchor region must be [x1,y1,x2,y2]")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in raw
        ):
            raise ValueError("anchor region coordinates must be numeric")
        x1, y1, x2, y2 = [float(item) for item in raw]
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("anchor regions must be normalized and non-empty")
        region = [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]
        if region not in regions:
            regions.append(region)
    return regions[:4]


def _prepare_views(
    image_path: str,
    anchor_regions: Sequence[Sequence[float]],
    *,
    scope: str,
) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image_inputs: List[str] = [image_path]
    views: List[Dict[str, Any]] = [
        {
            "view_index": 0,
            "kind": "original",
            "region": [0.0, 0.0, 1.0, 1.0],
        }
    ]
    temporary_paths: List[str] = []
    regions = [list(item) for item in anchor_regions]
    derived: List[tuple[str, List[float]]] = []
    if scope == "relation" and len(regions) >= 2:
        derived.append(("relation_context", _expand_region(_union_region(regions), 0.08)))
    for region in regions:
        derived.append(("anchor_detail", _expand_region(region, 0.06)))

    seen = {(0.0, 0.0, 1.0, 1.0)}
    for kind, region in derived:
        key = tuple(region)
        if key in seen:
            continue
        seen.add(key)
        x1 = max(0, min(width - 1, int(region[0] * width)))
        y1 = max(0, min(height - 1, int(region[1] * height)))
        x2 = max(x1 + 1, min(width, int(region[2] * width)))
        y2 = max(y1 + 1, min(height, int(region[3] * height)))
        crop = image.crop((x1, y1, x2, y2))
        handle = tempfile.NamedTemporaryFile(
            prefix="ifv_visual_view_",
            suffix=".png",
            delete=False,
        )
        handle.close()
        crop.save(handle.name)
        temporary_paths.append(handle.name)
        image_inputs.append(handle.name)
        views.append(
            {
                "view_index": len(views),
                "kind": kind,
                "region": region,
            }
        )
        if len(views) >= 5:
            break
    return image_inputs, views, temporary_paths


def _expand_region(region: Sequence[float], margin: float) -> List[float]:
    x1, y1, x2, y2 = [float(item) for item in region]
    width = x2 - x1
    height = y2 - y1
    return [
        round(max(0.0, x1 - max(margin, width * 0.12)), 6),
        round(max(0.0, y1 - max(margin, height * 0.12)), 6),
        round(min(1.0, x2 + max(margin, width * 0.12)), 6),
        round(min(1.0, y2 + max(margin, height * 0.12)), 6),
    ]


def _union_region(regions: Sequence[Sequence[float]]) -> List[float]:
    return [
        min(float(item[0]) for item in regions),
        min(float(item[1]) for item in regions),
        max(float(item[2]) for item in regions),
        max(float(item[3]) for item in regions),
    ]


def _normalize_model_output(
    parsed: Mapping[str, Any],
    views: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    allowed_status = {"observed", "not_observed", "ambiguous"}
    answer_status = str(parsed.get("answer_status", "")).strip()
    if answer_status not in allowed_status:
        raise ValueError("answer_status is missing or invalid")
    summary = " ".join(str(parsed.get("summary", "")).split())
    if not summary:
        raise ValueError("summary is required")
    view_by_index = {
        int(item["view_index"]): item
        for item in views
    }
    observations: List[Dict[str, Any]] = []
    for raw in parsed.get("observations", []) or []:
        if not isinstance(raw, Mapping):
            continue
        view_index = raw.get("view_index")
        if isinstance(view_index, bool) or not isinstance(view_index, int):
            continue
        view = view_by_index.get(view_index)
        if view is None:
            continue
        statement = " ".join(str(raw.get("statement", "")).split())
        property_status = str(raw.get("property_status", "")).strip()
        if not statement or property_status not in allowed_status:
            continue
        confidence = raw.get("confidence", 0.0)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            confidence = 0.0
        observations.append(
            {
                "view_index": view_index,
                "view_kind": str(view.get("kind", "")),
                "region": list(view.get("region", [])),
                "statement": statement[:600],
                "property_status": property_status,
                "confidence": round(max(0.0, min(1.0, float(confidence))), 4),
            }
        )
    limitations = [
        " ".join(str(item).split())[:400]
        for item in parsed.get("limitations", []) or []
        if " ".join(str(item).split())
    ][:6]
    return {
        "answer_status": answer_status,
        "summary": summary[:1200],
        "observations": observations[:8],
        "limitations": limitations,
    }

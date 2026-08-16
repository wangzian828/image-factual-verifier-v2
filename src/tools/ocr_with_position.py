# -*- coding: utf-8 -*-
"""Positioned OCR backed by the PaddleOCR cloud job API."""

from __future__ import annotations

import hashlib
import html
import io
import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import requests

from src.tools.base import BaseTool


DEFAULT_JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
DEFAULT_MODEL = "PaddleOCR-VL-1.6"
DEFAULT_POLL_SECONDS = 2.0
DEFAULT_JOB_TIMEOUT_SECONDS = 180.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_SUBMIT_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0


class PaddleOCRAPIError(RuntimeError):
    """An explicit remote OCR provider failure."""


@dataclass
class OCRWithPositionTool(BaseTool):
    """Extract positioned text through PaddleOCR's asynchronous cloud API.

    The API is deliberately the only runtime OCR backend.  Missing credentials,
    failed jobs, malformed provider output, and timeouts are explicit tool
    failures; there is no local PaddleOCR fallback.
    """

    name: str = "ocr_with_position"
    description: str = (
        "Extract text from the image with position information through the "
        "PaddleOCR cloud OCR API. Returns text regions, bounding boxes, "
        "confidence scores, and detected language."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {
                    "type": "string",
                    "description": "Local path to the image file.",
                },
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "Optional target region [x1,y1,x2,y2], using normalized "
                        "0-1 coordinates or absolute pixels."
                    ),
                },
                "goal": {
                    "type": "string",
                    "description": "What visible text or property this OCR should check.",
                },
                "source_evidence_id": {
                    "type": "string",
                    "description": "Evidence id that motivated this visual revisit.",
                },
                "expected_property": {
                    "type": "string",
                    "description": "Discriminative text or property expected in the region.",
                },
                "visual_question_id": {
                    "type": "string",
                    "description": "Pending visual question this OCR resolves.",
                },
                "source_discovery_id": {
                    "type": "string",
                    "description": "Discovery id that motivated this revisit.",
                },
            },
            "required": ["image_input"],
        }
    )
    min_confidence: float = 0.5
    job_url: str = ""
    model: str = ""
    poll_seconds: float = DEFAULT_POLL_SECONDS
    job_timeout_seconds: float = DEFAULT_JOB_TIMEOUT_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    http_client: Any = field(default=None, repr=False)

    def _job_url(self) -> str:
        return (
            self.job_url.strip()
            or os.getenv("PADDLEOCR_API_URL", "").strip()
            or DEFAULT_JOB_URL
        )

    def _model(self) -> str:
        return (
            self.model.strip()
            or os.getenv("PADDLEOCR_API_MODEL", "").strip()
            or DEFAULT_MODEL
        )

    def _token(self) -> str:
        token = os.getenv("PADDLEOCR_API_TOKEN", "").strip()
        if not token:
            raise PaddleOCRAPIError(
                "PADDLEOCR_API_TOKEN is required for ocr_with_position"
            )
        return token

    def _poll_interval(self) -> float:
        raw = os.getenv(
            "PADDLEOCR_API_POLL_SECONDS",
            str(self.poll_seconds),
        )
        return max(0.2, float(raw))

    def _job_timeout(self) -> float:
        raw = os.getenv(
            "PADDLEOCR_API_JOB_TIMEOUT_SECONDS",
            str(self.job_timeout_seconds),
        )
        return max(5.0, float(raw))

    def _request_timeout(self) -> float:
        raw = os.getenv(
            "PADDLEOCR_API_REQUEST_TIMEOUT_SECONDS",
            str(self.request_timeout_seconds),
        )
        return max(1.0, float(raw))

    def _submit_retries(self) -> int:
        raw = os.getenv(
            "PADDLEOCR_API_SUBMIT_RETRIES",
            str(DEFAULT_SUBMIT_RETRIES),
        )
        return max(0, min(5, int(raw)))

    def _retry_backoff_seconds(self, attempt: int) -> float:
        raw = os.getenv(
            "PADDLEOCR_API_RETRY_BACKOFF_SECONDS",
            str(DEFAULT_RETRY_BACKOFF_SECONDS),
        )
        return min(60.0, max(0.0, float(raw)) * (2**attempt))

    def _http(self) -> Any:
        return self.http_client or requests

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_path = str(params.get("image_input", "")).strip()
        artifact_bytes = b""
        requested_bbox: List[float] = []

        try:
            from PIL import Image

            if not image_path:
                raise ValueError("image_input is required")
            with Image.open(image_path).convert("RGB") as image:
                image_width, image_height = image.size
                requested_bbox = self._normalize_bbox(
                    params.get("bbox"),
                    image_width,
                    image_height,
                )
                if requested_bbox:
                    px1 = int(round(requested_bbox[0] * image_width))
                    py1 = int(round(requested_bbox[1] * image_height))
                    px2 = int(round(requested_bbox[2] * image_width))
                    py2 = int(round(requested_bbox[3] * image_height))
                    image = image.crop((px1, py1, px2, py2))
                    offset_x, offset_y = px1, py1
                else:
                    offset_x = offset_y = 0
                ocr_width, ocr_height = image.size
                artifact_bytes = self._image_bytes(image)

            payload, attempts = self._submit_and_poll(artifact_bytes)
            raw_regions, markdown_text = self._extract_regions(payload)
            text_regions: List[Dict[str, Any]] = []
            rejected_text_regions: List[Dict[str, Any]] = []

            for raw_bbox, text, confidence in raw_regions:
                region = self._build_region(
                    raw_bbox,
                    text,
                    confidence,
                    width=ocr_width,
                    height=ocr_height,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    original_width=image_width,
                    original_height=image_height,
                )
                if (
                    region["text"].strip()
                    and float(region["confidence"]) >= float(self.min_confidence)
                ):
                    text_regions.append(region)
                else:
                    rejected_text_regions.append(region)

            position_quality = "api_layout"
            if not text_regions and markdown_text.strip():
                # Keep visible text usable when a provider version returns
                # Markdown but omits per-block coordinates.
                fallback = self._build_region(
                    [[0, 0], [ocr_width, 0], [ocr_width, ocr_height], [0, ocr_height]],
                    markdown_text.strip(),
                    1.0,
                    width=ocr_width,
                    height=ocr_height,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    original_width=image_width,
                    original_height=image_height,
                )
                text_regions.append(fallback)
                position_quality = "whole_crop_fallback"

            full_text = " ".join(
                item["text"] for item in text_regions if item["text"].strip()
            )
            return {
                "status": "success",
                "text_regions": text_regions,
                "total_regions": len(text_regions),
                "full_text": full_text,
                "rejected_text_regions": rejected_text_regions,
                "requested_bbox": requested_bbox,
                "goal": str(params.get("goal", "")),
                "source_evidence_id": str(
                    params.get("source_evidence_id", "")
                ),
                "expected_property": str(
                    params.get("expected_property", "")
                ),
                "artifact_sha256": hashlib.sha256(
                    artifact_bytes
                ).hexdigest(),
                "ocr_backend": "paddleocr_api",
                "ocr_model": self._model(),
                "position_quality": position_quality,
                "backend_attempts": attempts,
                "subcalls": self._ocr_subcalls(attempts),
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": f"OCR API failed: {exc}",
                "text_regions": [],
                "total_regions": 0,
                "full_text": "",
                "rejected_text_regions": [],
                "requested_bbox": requested_bbox,
                "artifact_sha256": (
                    hashlib.sha256(artifact_bytes).hexdigest()
                    if artifact_bytes
                    else ""
                ),
                "ocr_backend": "paddleocr_api",
                "ocr_model": self._model(),
                "backend_attempts": [
                    {
                        "backend": "paddleocr_api",
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ],
                "subcalls": [
                    {
                        "kind": "ocr",
                        "provider": "paddleocr_api",
                        "status": "error",
                        "request_count": 1,
                    }
                ],
            }

    def _submit_and_poll(
        self,
        image_bytes: bytes,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        job_url = self._job_url()
        headers = {"Authorization": f"bearer {self._token()}"}
        optional_payload = {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        }
        attempts: List[Dict[str, Any]] = []
        response = None
        max_attempts = self._submit_retries() + 1
        for attempt in range(max_attempts):
            response = self._http().post(
                job_url,
                headers=headers,
                data={
                    "model": self._model(),
                    "optionalPayload": json.dumps(optional_payload),
                },
                files={
                    "file": (
                        "ocr-input.png",
                        image_bytes,
                        "image/png",
                    )
                },
                timeout=self._request_timeout(),
            )
            if response.ok:
                attempts.append(
                    {
                        "backend": "paddleocr_api_submit",
                        "status": "success",
                        "attempt": attempt + 1,
                    }
                )
                break

            if (
                self._retryable_submit_response(response)
                and attempt + 1 < max_attempts
            ):
                delay = self._retry_backoff_seconds(attempt)
                attempts.append(
                    {
                        "backend": "paddleocr_api_submit",
                        "status": "retry",
                        "http_status": response.status_code,
                        "retry_after_seconds": delay,
                        "attempt": attempt + 1,
                    }
                )
                time.sleep(delay)
                continue

            attempts.append(
                {
                    "backend": "paddleocr_api_submit",
                    "status": "error",
                    "http_status": response.status_code,
                    "attempt": attempt + 1,
                }
            )
            raise PaddleOCRAPIError(
                f"submit returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        if response is None or not response.ok:
            raise PaddleOCRAPIError("submit did not return a usable response")
        try:
            job_id = str(response.json()["data"]["jobId"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise PaddleOCRAPIError(
                "submit response lacks data.jobId"
            ) from exc
        if not job_id:
            raise PaddleOCRAPIError("submit response returned an empty jobId")

        deadline = time.monotonic() + self._job_timeout()
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"PaddleOCR job {job_id} exceeded "
                    f"{self._job_timeout():.1f}s"
                )
            status_response = self._http().get(
                f"{job_url}/{job_id}",
                headers=headers,
                timeout=self._request_timeout(),
            )
            if not status_response.ok:
                raise PaddleOCRAPIError(
                    f"poll returned HTTP {status_response.status_code}: "
                    f"{status_response.text[:500]}"
                )
            try:
                data = status_response.json()["data"]
                state = str(data.get("state", "")).lower()
            except (KeyError, TypeError, ValueError) as exc:
                raise PaddleOCRAPIError(
                    "poll response lacks data.state"
                ) from exc
            attempts.append(
                {
                    "backend": "paddleocr_api_poll",
                    "status": state or "unknown",
                }
            )
            if state == "done":
                json_url = str(
                    (data.get("resultUrl") or {}).get("jsonUrl", "")
                ).strip()
                if not json_url:
                    raise PaddleOCRAPIError(
                        "completed job lacks data.resultUrl.jsonUrl"
                    )
                result_response = self._http().get(
                    json_url,
                    timeout=self._request_timeout(),
                )
                if not result_response.ok:
                    raise PaddleOCRAPIError(
                        "result download returned HTTP "
                        f"{result_response.status_code}"
                    )
                attempts.append(
                    {
                        "backend": "paddleocr_api_result",
                        "status": "success",
                    }
                )
                return self._decode_jsonl_result(result_response.text), attempts
            if state == "failed":
                raise PaddleOCRAPIError(
                    f"job failed: {str(data.get('errorMsg', 'unknown error'))[:500]}"
                )
            time.sleep(min(self._poll_interval(), max(0.2, deadline - time.monotonic())))

    @staticmethod
    def _retryable_submit_response(response: Any) -> bool:
        if response.status_code in {408, 425, 429, 500, 502, 503, 504}:
            return True
        try:
            payload = response.json()
        except (TypeError, ValueError):
            return False
        code = str(payload.get("code", "")).strip()
        message = str(payload.get("msg", "")).lower()
        return code == "10010" or "queue" in message or "队列" in message

    @staticmethod
    def _decode_jsonl_result(text: str) -> Dict[str, Any]:
        rows = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
        if not rows:
            raise PaddleOCRAPIError("result JSONL is empty")
        first = rows[0]
        if not isinstance(first, Mapping):
            raise PaddleOCRAPIError("result JSONL row is not an object")
        result = first.get("result", first)
        if not isinstance(result, Mapping):
            raise PaddleOCRAPIError("result JSONL lacks an object result")
        return dict(result)

    @classmethod
    def _extract_regions(
        cls,
        payload: Mapping[str, Any],
    ) -> tuple[List[tuple[Any, str, float]], str]:
        regions: List[tuple[Any, str, float]] = []
        markdown_parts: List[str] = []
        pages = payload.get("layoutParsingResults")
        if not isinstance(pages, list):
            pages = [payload]

        for page in pages:
            if not isinstance(page, Mapping):
                continue
            markdown_parts.extend(cls._markdown_texts(page))
            parsed = cls._json_value(page.get("prunedResult"))
            candidates = [
                page,
                parsed,
                parsed.get("overall_ocr_res")
                if isinstance(parsed, Mapping)
                else None,
            ]
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                regions.extend(cls._array_regions(candidate))
                if regions:
                    break
            if not regions:
                for candidate in candidates:
                    if isinstance(candidate, Mapping):
                        regions.extend(cls._block_regions(candidate))
                        if regions:
                            break

        deduped: List[tuple[Any, str, float]] = []
        seen: set[tuple[str, str]] = set()
        for bbox, text, confidence in regions:
            key = (json.dumps(bbox, sort_keys=True), text.strip())
            if key not in seen and text.strip():
                seen.add(key)
                deduped.append((bbox, text, confidence))
        return deduped, "\n".join(markdown_parts)

    @classmethod
    def _array_regions(
        cls,
        data: Mapping[str, Any],
    ) -> List[tuple[Any, str, float]]:
        texts = cls._as_list(
            cls._first_present(data, "rec_texts", "texts")
        )
        boxes = cls._as_list(
            cls._first_present(data, "rec_polys", "dt_polys", "rec_boxes")
        )
        scores = cls._as_list(
            cls._first_present(data, "rec_scores", "scores")
        )
        if not texts or not boxes:
            return []
        if len(scores) < len(texts):
            scores.extend([1.0] * (len(texts) - len(scores)))
        return [
            (boxes[index], str(texts[index] or "").strip(), float(scores[index]))
            for index in range(min(len(texts), len(boxes)))
        ]

    @classmethod
    def _block_regions(
        cls,
        data: Mapping[str, Any],
    ) -> List[tuple[Any, str, float]]:
        for key in ("parsing_res_list", "layout_res_list", "blocks"):
            items = data.get(key)
            if not isinstance(items, list):
                continue
            regions: List[tuple[Any, str, float]] = []
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                text = str(
                    cls._first_present(
                        item,
                        "block_content",
                        "text",
                        "content",
                    )
                    or ""
                ).strip()
                bbox = cls._first_present(
                    item,
                    "block_bbox",
                    "bbox",
                    "box",
                    "coordinate",
                )
                if text and bbox is not None:
                    regions.append(
                        (
                            bbox,
                            text,
                            float(item.get("score", item.get("confidence", 1.0))),
                        )
                    )
            if regions:
                return regions
        return []

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return value

    @staticmethod
    def _markdown_texts(page: Mapping[str, Any]) -> List[str]:
        markdown = page.get("markdown")
        if isinstance(markdown, Mapping):
            value = markdown.get("text") or markdown.get("markdown_texts")
            if isinstance(value, list):
                return [
                    cleaned
                    for item in value
                    if (cleaned := OCRWithPositionTool._clean_markdown_text(item))
                ]
            if value:
                cleaned = OCRWithPositionTool._clean_markdown_text(value)
                return [cleaned] if cleaned else []
        return []

    @staticmethod
    def _clean_markdown_text(value: Any) -> str:
        text = html.unescape(str(value))
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
        text = re.sub(r"<img\b[^>]*>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"`{1,3}", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _build_region(
        self,
        bbox: Any,
        text: str,
        confidence: float,
        *,
        width: int,
        height: int,
        offset_x: int,
        offset_y: int,
        original_width: int,
        original_height: int,
    ) -> Dict[str, Any]:
        quad = self._normalize_quad(bbox, index=0)
        if any(
            point[0] < 0
            or point[1] < 0
            or point[0] > width
            or point[1] > height
            for point in quad
        ):
            raise PaddleOCRAPIError(
                "PaddleOCR API region coordinates exceed the uploaded image"
            )
        global_bbox = [
            [point[0] + offset_x, point[1] + offset_y]
            for point in quad
        ]
        xs = [point[0] for point in global_bbox]
        ys = [point[1] for point in global_bbox]
        return {
            "text": str(text).strip(),
            "bbox_quad": [
                [
                    round(float(point[0]) / original_width, 4),
                    round(float(point[1]) / original_height, 4),
                ]
                for point in global_bbox
            ],
            "bbox": [
                round(min(xs) / original_width, 4),
                round(min(ys) / original_height, 4),
                round(max(xs) / original_width, 4),
                round(max(ys) / original_height, 4),
            ],
            "confidence": round(float(confidence), 3),
            "language": (
                "zh"
                if any(0x4E00 <= ord(char) <= 0x9FFF for char in str(text))
                else "en"
            ),
        }

    @staticmethod
    def _image_bytes(image: Any) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _ocr_subcalls(
        attempts: Iterable[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [
            {
                "kind": "ocr",
                "provider": str(item.get("backend", "paddleocr_api")),
                "status": str(item.get("status", "error")),
                "request_count": 1,
            }
            for item in attempts
        ]

    @staticmethod
    def _normalize_bbox(raw_bbox: Any, width: int, height: int) -> List[float]:
        if raw_bbox in (None, []):
            return []
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2].")
        if not all(isinstance(value, (int, float)) for value in raw_bbox):
            raise ValueError("bbox coordinates must be numeric.")
        values = [float(value) for value in raw_bbox]
        if all(0.0 <= value <= 1.0 for value in values):
            x1, y1, x2, y2 = values
        else:
            x1, y1, x2, y2 = (
                values[0] / width,
                values[1] / height,
                values[2] / width,
                values[3] / height,
            )
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError("bbox must be ordered and remain inside the image")
        if x2 - x1 <= 0.001 or y2 - y1 <= 0.001:
            raise ValueError("bbox must describe a non-empty region inside the image.")
        return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]

    @staticmethod
    def _first_present(data: Mapping[str, Any], *names: str) -> Any:
        for name in names:
            if name in data and data[name] is not None:
                return data[name]
        return None

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        try:
            return list(value)
        except TypeError:
            return [value]

    @staticmethod
    def _normalize_quad(value: Any, *, index: int) -> List[List[float]]:
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
        if (
            isinstance(value, list)
            and len(value) == 4
            and all(
                isinstance(point, (list, tuple))
                and len(point) == 2
                and all(
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    for item in point
                )
                for point in value
            )
        ):
            return [[float(point[0]), float(point[1])] for point in value]
        if (
            isinstance(value, list)
            and len(value) == 4
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                for item in value
            )
        ):
            x1, y1, x2, y2 = [float(item) for item in value]
            if x1 < x2 and y1 < y2:
                return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        raise ValueError(
            f"PaddleOCR API region {index} requires a quadrilateral or bbox"
        )

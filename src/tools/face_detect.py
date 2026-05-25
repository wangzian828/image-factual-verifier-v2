# -*- coding: utf-8 -*-
"""Face detection using InsightFace.

Returns face bounding boxes, 512-d embeddings, age, and gender.
Runs on CPU with ONNX runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.tools.base import BaseTool


@dataclass
class FaceDetectTool(BaseTool):
    """Face detection tool using InsightFace buffalo_l model.

    Provides:
    - Face bounding boxes
    - 512-dimensional face embeddings (for identity comparison)
    - Age and gender estimation
    - Detection confidence scores
    - Runs on CPU (~0.3-1s per image)
    """

    name: str = "face_detect"
    description: str = (
        "Detect faces in the image and extract embeddings for identity verification. "
        "Returns bounding boxes, 512-d face embeddings (for comparing with known faces), "
        "estimated age, gender, and confidence. Use when people are visible and you need "
        "to verify identity or count faces."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {
                    "type": "string",
                    "description": "Local path to the image file.",
                },
            },
            "required": ["image_input"],
        }
    )

    _app: Optional[Any] = field(default=None, repr=False)

    def _get_app(self):
        """Lazy initialization of InsightFace (loads model on first call)."""
        if self._app is None:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
        return self._app

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect faces in the image and return structured results."""
        image_path = params["image_input"]

        try:
            import cv2

            img = cv2.imread(image_path)
            if img is None:
                return {
                    "status": "error",
                    "error": f"Cannot read image: {image_path}",
                    "faces": [],
                }

            app = self._get_app()
            faces = app.get(img)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Face detection failed: {str(e)}",
                "faces": [],
            }

        detections: List[Dict[str, Any]] = []
        img_h, img_w = img.shape[:2]

        for face in faces:
            # Normalize bbox to 0-1 range
            bbox = face.bbox.tolist()
            normalized_bbox = [
                round(bbox[0] / img_w, 4),
                round(bbox[1] / img_h, 4),
                round(bbox[2] / img_w, 4),
                round(bbox[3] / img_h, 4),
            ]

            # Embedding: keep as list for JSON serialization
            # In practice, embeddings are compared in-memory (not serialized to LLM)
            embedding = face.embedding.tolist() if face.embedding is not None else []

            detections.append({
                "bbox": normalized_bbox,
                "embedding_available": len(embedding) > 0,
                "age": int(face.age) if hasattr(face, "age") else 0,
                "gender": "male" if getattr(face, "gender", 0) == 1 else "female",
                "confidence": round(float(face.det_score), 3),
            })

        # Store embeddings separately (not sent to LLM, used by verify_face_identity)
        self._last_embeddings = [
            face.embedding for face in faces if face.embedding is not None
        ]

        return {
            "status": "success",
            "faces": detections,
            "total_faces": len(detections),
        }

    def get_embeddings(self) -> List[np.ndarray]:
        """Get raw embeddings from the last detection call (for identity verification)."""
        return getattr(self, "_last_embeddings", [])

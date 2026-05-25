# -*- coding: utf-8 -*-
"""Face detection using facenet-pytorch (MTCNN + InceptionResnetV1).

Returns face bounding boxes, 512-d embeddings, and confidence scores.
Uses PyTorch backend — no onnxruntime dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.tools.base import BaseTool


@dataclass
class FaceDetectTool(BaseTool):
    """Face detection tool using facenet-pytorch.

    Uses MTCNN for detection and InceptionResnetV1 (pretrained on VGGFace2)
    for 512-d embedding extraction. Pure PyTorch, GPU-accelerated.

    Provides:
    - Face bounding boxes (normalized 0-1)
    - 512-dimensional face embeddings (for identity comparison)
    - Detection confidence scores
    """

    name: str = "face_detect"
    description: str = (
        "Detect faces in the image and extract embeddings for identity verification. "
        "Returns bounding boxes, 512-d face embeddings (for comparing with known faces), "
        "and confidence. Use when people are visible and you need "
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

    _mtcnn: Optional[Any] = field(default=None, repr=False)
    _resnet: Optional[Any] = field(default=None, repr=False)
    _last_embeddings: List[Any] = field(default_factory=list, repr=False)

    def _get_models(self):
        """Lazy initialization of MTCNN + InceptionResnetV1."""
        if self._mtcnn is None:
            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._mtcnn = MTCNN(
                keep_all=True,
                device=device,
                post_process=True,  # normalize face tensors for resnet
            )
            self._resnet = InceptionResnetV1(
                pretrained="vggface2",
            ).eval().to(device)
            self._device = device

        return self._mtcnn, self._resnet

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect faces in the image and return structured results."""
        image_path = params["image_input"]

        try:
            from PIL import Image
            import torch

            img = Image.open(image_path).convert("RGB")
            img_w, img_h = img.size

            mtcnn, resnet = self._get_models()

            # Detect faces — returns boxes and probabilities
            boxes, probs = mtcnn.detect(img)

            if boxes is None or len(boxes) == 0:
                self._last_embeddings = []
                return {
                    "status": "success",
                    "faces": [],
                    "total_faces": 0,
                }

            # Get face tensors for embedding extraction
            faces_tensor = mtcnn(img)  # Returns aligned face tensors

            if faces_tensor is None:
                self._last_embeddings = []
                return {
                    "status": "success",
                    "faces": [],
                    "total_faces": 0,
                }

            # Extract embeddings
            with torch.no_grad():
                embeddings = resnet(faces_tensor.to(self._device))

            detections: List[Dict[str, Any]] = []
            self._last_embeddings = []

            for i, (box, prob) in enumerate(zip(boxes, probs)):
                # Normalize bbox to 0-1 range
                normalized_bbox = [
                    round(float(box[0]) / img_w, 4),
                    round(float(box[1]) / img_h, 4),
                    round(float(box[2]) / img_w, 4),
                    round(float(box[3]) / img_h, 4),
                ]

                emb = embeddings[i].cpu().numpy()
                self._last_embeddings.append(emb)

                detections.append({
                    "bbox": normalized_bbox,
                    "embedding_available": True,
                    "confidence": round(float(prob), 3) if prob is not None else 0.0,
                })

        except Exception as e:
            self._last_embeddings = []
            return {
                "status": "error",
                "error": f"Face detection failed: {str(e)}",
                "faces": [],
                "total_faces": 0,
            }

        return {
            "status": "success",
            "faces": detections,
            "total_faces": len(detections),
        }

    def get_embeddings(self) -> List[np.ndarray]:
        """Get raw embeddings from the last detection call (for identity verification)."""
        return self._last_embeddings

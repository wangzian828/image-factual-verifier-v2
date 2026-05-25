# -*- coding: utf-8 -*-
"""Face identity verification using embedding cosine similarity.

Supports:
1. Compare two faces in the same image
2. Compare a face against a reference embedding
3. Compare a face against a reference image URL (downloads + detects + compares)
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.tools.base import BaseTool


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class VerifyFaceIdentityTool(BaseTool):
    """Compare face embeddings to verify identity.

    Can compare:
    1. Two faces in the same image (face_index_a vs face_index_b)
    2. A detected face against a known person's embedding (face_index_a + reference_embedding)
    3. A detected face against a face in a reference image URL (face_index_a + reference_image_url)

    Mode 3 enables cross-image verification: download a reference photo from
    search results, detect faces in it, and compare embeddings.

    Threshold: similarity > 0.4 is typically a match for InsightFace buffalo_l.
    """

    name: str = "verify_face_identity"
    description: str = (
        "Compare faces to verify identity. Three modes:\n"
        "1. Same image: provide face_index_a and face_index_b\n"
        "2. Known embedding: provide face_index_a and reference_embedding (512-d list)\n"
        "3. Reference image: provide face_index_a and reference_image_url — "
        "the tool will download the image, detect faces, and compare.\n"
        "Requires face_detect to have been called first on the main image."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "face_index_a": {
                    "type": "integer",
                    "description": "Index of the face in the main image (0-based, from face_detect results).",
                },
                "face_index_b": {
                    "type": "integer",
                    "description": "Index of another face in the same image to compare (0-based). Use -1 if using reference.",
                },
                "reference_embedding": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "A known person's 512-d embedding to compare against.",
                },
                "reference_image_url": {
                    "type": "string",
                    "description": (
                        "URL of a reference image containing the person to compare against. "
                        "The tool will download it, detect faces, and compare the largest face. "
                        "Use this when you found a photo of the person via search."
                    ),
                },
                "reference_face_index": {
                    "type": "integer",
                    "description": "Which face in the reference image to use (0-based). Default: 0 (largest face).",
                },
            },
            "required": ["face_index_a"],
        }
    )

    # Reference to the face_detect tool to get embeddings
    face_detect_tool: Optional[Any] = field(default=None, repr=False)

    # Known person embeddings database (name → embedding)
    known_faces: Dict[str, np.ndarray] = field(default_factory=dict)

    # Similarity threshold for a match
    # facenet-pytorch (VGGFace2) embeddings are L2-normalized,
    # cosine similarity > 0.6 is typically a match
    match_threshold: float = 0.6

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compare face embeddings."""
        face_index_a = params["face_index_a"]
        face_index_b = params.get("face_index_b", -1)
        reference_embedding = params.get("reference_embedding")
        reference_image_url = params.get("reference_image_url")
        reference_face_index = params.get("reference_face_index", 0)

        # Get embeddings from face_detect tool
        if self.face_detect_tool is None:
            return {
                "status": "error",
                "error": "face_detect tool not available. Call face_detect first.",
            }

        embeddings = self.face_detect_tool.get_embeddings()
        if not embeddings:
            return {
                "status": "error",
                "error": "No face embeddings available. Call face_detect first.",
            }

        if face_index_a >= len(embeddings):
            return {
                "status": "error",
                "error": f"face_index_a={face_index_a} out of range. Only {len(embeddings)} faces detected.",
            }

        emb_a = embeddings[face_index_a]

        # Determine what to compare against
        if reference_image_url:
            # Mode 3: download reference image, detect face, extract embedding
            result = self._compare_with_reference_image(
                emb_a, reference_image_url, reference_face_index
            )
            return result
        elif reference_embedding is not None:
            # Mode 2: compare with provided embedding
            emb_b = np.array(reference_embedding, dtype=np.float32)
            compare_desc = "reference embedding"
        elif face_index_b >= 0:
            # Mode 1: compare two faces in same image
            if face_index_b >= len(embeddings):
                return {
                    "status": "error",
                    "error": f"face_index_b={face_index_b} out of range. Only {len(embeddings)} faces detected.",
                }
            emb_b = embeddings[face_index_b]
            compare_desc = f"face #{face_index_b}"
        else:
            return {
                "status": "error",
                "error": "Must provide face_index_b, reference_embedding, or reference_image_url.",
            }

        similarity = cosine_similarity(emb_a, emb_b)
        is_match = similarity > self.match_threshold

        return {
            "status": "success",
            "similarity": round(similarity, 4),
            "is_match": is_match,
            "threshold": self.match_threshold,
            "comparison": f"face #{face_index_a} vs {compare_desc}",
            "verdict": "same person" if is_match else "different people",
        }

    def _compare_with_reference_image(
        self,
        emb_a: np.ndarray,
        reference_url: str,
        reference_face_index: int = 0,
    ) -> Dict[str, Any]:
        """Download a reference image, detect faces, and compare embeddings."""
        import requests

        # Save main image embeddings (face_detect_tool.call will overwrite them)
        saved_embeddings = self.face_detect_tool.get_embeddings().copy()

        # Download reference image
        tmp_path = None
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
                ),
            }
            proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("https_proxy") or os.environ.get("http_proxy")
            proxies = {"http": proxy, "https": proxy} if proxy else None
            resp = requests.get(reference_url, headers=headers, timeout=15, proxies=proxies)
            resp.raise_for_status()

            # Save to temp file
            suffix = ".jpg"
            if "png" in resp.headers.get("Content-Type", ""):
                suffix = ".png"
            tmp_path = os.path.join(
                tempfile.gettempdir(), f"ref_face_{os.getpid()}_{id(self)}{suffix}"
            )
            with open(tmp_path, "wb") as f:
                f.write(resp.content)

        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to download reference image: {str(e)}",
                "reference_url": reference_url,
            }

        # Detect faces in reference image
        try:
            # Reuse face_detect_tool to detect faces in the reference image
            ref_result = self.face_detect_tool.call({"image_input": tmp_path})

            if ref_result.get("status") == "error":
                return {
                    "status": "error",
                    "error": f"Face detection on reference failed: {ref_result.get('error')}",
                    "reference_url": reference_url,
                }

            if ref_result.get("total_faces", 0) == 0:
                return {
                    "status": "error",
                    "error": "No faces detected in reference image.",
                    "reference_url": reference_url,
                }

            ref_embeddings = self.face_detect_tool.get_embeddings()
            if not ref_embeddings:
                return {
                    "status": "error",
                    "error": "Could not extract embedding from reference face.",
                }

            if reference_face_index >= len(ref_embeddings):
                reference_face_index = 0

            ref_embedding = ref_embeddings[reference_face_index]

        except Exception as e:
            return {
                "status": "error",
                "error": f"Face detection on reference image failed: {str(e)}",
            }
        finally:
            # Restore main image embeddings
            self.face_detect_tool._last_embeddings = saved_embeddings
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # Compare
        similarity = cosine_similarity(emb_a, ref_embedding)
        is_match = similarity > self.match_threshold

        return {
            "status": "success",
            "similarity": round(similarity, 4),
            "is_match": is_match,
            "threshold": self.match_threshold,
            "comparison": f"main image face vs reference image face (from {reference_url})",
            "verdict": "same person" if is_match else "different people",
            "reference_faces_found": len(ref_faces),
        }

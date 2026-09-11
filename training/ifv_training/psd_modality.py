"""Prevent text-only PSD adapters from silently dropping visual conditioning.

The current target format has no pixel tensors or image-grid/position data.
Qwen3.5 token IDs below are verified against the deployed 9B config.json.
These guards must remain until both scorer and trainer support bound media.
"""

from typing import Any, Mapping, Sequence


QWEN35_VISUAL_TOKEN_IDS = frozenset({248053, 248054, 248056, 248057})


def require_text_only_psd(
    row: Mapping[str, Any], *token_sequences: Sequence[int]
) -> None:
    visual_fields = (
        "images", "videos", "multi_modal_data", "pixel_values",
        "pixel_values_videos", "image_grid_thw", "video_grid_thw",
        "multimodal_inputs",
    )
    # Avoid truth-testing tensors; an explicitly present payload is unsupported.
    has_media = any(row.get(field) is not None for field in visual_fields)
    has_visual_ids = any(
        token in QWEN35_VISUAL_TOKEN_IDS
        for sequence in token_sequences for token in sequence
    )
    if has_media or has_visual_ids:
        raise ValueError(
            "psd_multimodal_not_supported: token-only scoring/training would "
            "drop visual conditioning; requires bound media, processor grids "
            "and multimodal position IDs"
        )

"""Opt-in bootstrap for the repository-owned ms-swift encode cache."""

from __future__ import annotations

import os
import traceback


if os.environ.get("IFV_ENCODE_CACHE_ENABLED", "").strip().casefold() in {
    "1",
    "true",
    "yes",
}:
    try:
        from ifv_training.encode_cache import install_ms_swift_encode_cache

        install_ms_swift_encode_cache()
    except Exception:
        traceback.print_exc()
        os._exit(70)

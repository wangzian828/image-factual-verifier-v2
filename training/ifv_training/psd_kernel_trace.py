"""Log FLA autotuner choices for bounded native-resume diagnostics.

Only kernel names, shape-key digests and launch configurations are recorded.
No sample, image, model parameter or gradient payload is inspected.
"""

import logging
import os


logger = logging.getLogger(__name__)


def install_fla_tuning_trace() -> None:
    if os.environ.get("IFV_PSD_FLA_TRACE") != "1":
        return
    from fla.ops.utils.cache import AutotuneKey, CachedAutotuner

    if getattr(CachedAutotuner, "_ifv_psd_trace_installed", False):
        return
    original = CachedAutotuner.run
    observed: set[tuple[str, str]] = set()

    def traced(self, *args, **kwargs):
        key = AutotuneKey.build(self.arg_names, self.keys, args, kwargs)
        result = original(self, *args, **kwargs)
        digest = AutotuneKey.key_hash(key.autotune_key)
        identity = (self.kernel_name, digest)
        if identity not in observed and len(observed) < 1024:
            config = self.cache.get(key.autotune_key)
            if config is not None:
                observed.add(identity)
                logger.warning(
                    "IFV PSD FLA rank=%s kernel=%s key=%s kwargs=%s warps=%s stages=%s ctas=%s",
                    os.environ.get("RANK", "?"), self.kernel_name, digest,
                    sorted(config.kwargs.items()), config.num_warps,
                    config.num_stages, getattr(config, "num_ctas", 1),
                )
        return result

    CachedAutotuner.run = traced
    CachedAutotuner._ifv_psd_trace_installed = True

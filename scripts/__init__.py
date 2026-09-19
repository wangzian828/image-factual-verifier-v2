"""Project utility scripts exposed as an importable package."""

# Deployment overlays may contribute one changed script without copying the
# whole validated code snapshot. Keep the package path composable so unchanged
# scripts continue to resolve from the immutable base snapshot.
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

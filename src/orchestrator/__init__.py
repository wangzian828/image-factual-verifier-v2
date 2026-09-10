from src.orchestrator.investigation_models import DiscrepancyJudgment
from src.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    TextRegion,
    VerificationState,
)
__all__ = [
    "Entity",
    "DiscrepancyJudgment",
    "ImageOnlyRuntimeCase",
    "PerceptionReport",
    "TextRegion",
    "VerificationState",
    "UnifiedReactState",
]


def __getattr__(name: str):
    """Load the active runtime state lazily to avoid tool import cycles."""

    if name == "UnifiedReactState":
        from src.orchestrator.react_runtime import UnifiedReactState

        return UnifiedReactState
    raise AttributeError(name)

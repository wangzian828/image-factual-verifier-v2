from src.orchestrator.investigation_models import (
    Finding,
    ImageOnlyInvestigationState,
    DiscrepancyJudgment,
    InvestigationBrief,
    ResearchTask,
    DiscrepancyVerdictBasis,
    VisualEntity,
    VisualFact,
)
from src.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    TextRegion,
    VerificationState,
)
__all__ = [
    "Entity",
    "Finding",
    "ImageOnlyInvestigationState",
    "DiscrepancyJudgment",
    "ImageOnlyRuntimeCase",
    "InvestigationBrief",
    "PerceptionReport",
    "ResearchTask",
    "TextRegion",
    "DiscrepancyVerdictBasis",
    "VerificationState",
    "VisualEntity",
    "VisualFact",
    "UnifiedReactState",
]


def __getattr__(name: str):
    """Load the active runtime state lazily to avoid tool import cycles."""

    if name == "UnifiedReactState":
        from src.orchestrator.react_runtime import UnifiedReactState

        return UnifiedReactState
    raise AttributeError(name)

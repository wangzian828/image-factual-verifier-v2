"""Generic schemas used only to exercise StageRunner protocol behavior."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolEvidence(StrictModel):
    function_call_id: str = ""
    source: str = ""
    summary: str = ""
    raw_excerpt: str = ""
    direction: Literal["supports", "refutes", "neutral"] = "neutral"
    quality: Literal["strong", "moderate", "weak"] = "moderate"
    tool_used: str = ""
    related_question: str = ""


class ToolStageOutput(StrictModel):
    evidence: List[ToolEvidence] = Field(default_factory=list)
    visual_anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    authenticity_assessment: Literal[
        "authentic",
        "likely_ai",
        "likely_manipulated",
        "uncertain",
    ] = "uncertain"
    key_findings: List[str] = Field(default_factory=list)
    source_findings: List[Dict[str, Any]] = Field(default_factory=list)
    visual_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    world_model: Dict[str, Any] = Field(default_factory=dict)
    question_resolutions: List[Dict[str, Any]] = Field(default_factory=list)
    coverage_complete: bool = False
    unresolved_priority_questions: List[str] = Field(default_factory=list)
    exhausted_priority_questions: List[str] = Field(default_factory=list)
    iteration_count: int = Field(default=0, ge=0)


class StructuredJudgmentOutput(StrictModel):
    verdict: Literal["real", "fake", "unverifiable"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_chain: str = ""
    key_evidence: List[str] = Field(default_factory=list)
    anomalies: List[str] = Field(default_factory=list)
    overall_assessment: str = ""


class TruthAptQuestion(StrictModel):
    question_id: str = ""
    question: str = ""
    claim_text: str = Field(min_length=1)


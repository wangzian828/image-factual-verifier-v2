# -*- coding: utf-8 -*-
"""Data contracts for the 4-stage orchestrator pipeline.

All inter-stage data is defined here as Pydantic models for:
- Type safety and validation
- Serialization (JSON export for trajectories)
- Clear API boundaries between stages
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Stage 1: PERCEPTION — output
# =============================================================================


class Entity(BaseModel):
    """A detected entity in the image."""

    name: str = ""  # e.g. "person in blue shirt", "Eiffel Tower"
    entity_type: str = ""  # "person"|"object"|"building"|"text"|"logo"|"animal"|"scene_element"
    bbox: List[float] = Field(default_factory=list)  # [x1, y1, x2, y2] normalized 0-1
    confidence: float = 1.0
    attributes: Dict[str, str] = Field(default_factory=dict)  # {"age": "~30", "color": "red"}


class TextRegion(BaseModel):
    """A detected text region with position."""

    text: str = ""
    bbox_quad: List[List[float]] = Field(default_factory=list)  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    confidence: float = 0.0
    language: str = "unknown"


class FaceDetection(BaseModel):
    """A detected face with embedding."""

    bbox: List[float] = Field(default_factory=list)  # [x1, y1, x2, y2]
    embedding: List[float] = Field(default_factory=list)  # 512-d vector
    age: int = 0
    gender: str = ""  # "male"|"female"
    confidence: float = 0.0


class PerceptionReport(BaseModel):
    """Stage 1 output: everything the system can observe in the image."""

    entities: List[Entity] = Field(default_factory=list)
    text_regions: List[TextRegion] = Field(default_factory=list)
    faces: List[FaceDetection] = Field(default_factory=list)
    scene_description: str = ""
    image_type: str = "photo"  # "photo"|"screenshot"|"document"|"illustration"|"meme"


# =============================================================================
# Stage 2: PLANNING — output
# =============================================================================


class InvestigationQuestion(BaseModel):
    """A question that needs to be investigated to verify the image."""

    question_id: str = ""  # "q0", "q1", ...
    question: str = ""  # "这个人是否是马斯克？"
    why: str = ""  # 为什么需要调查这个问题
    suggested_tools: List[str] = Field(default_factory=list)  # ["reverse_image_search", "text_search"]
    suggested_queries: List[str] = Field(default_factory=list)  # ["马斯克 抖音直播"]
    related_entities: List[str] = Field(default_factory=list)  # 关联的实体名
    priority: int = 1  # 1=必须, 2=建议, 3=可选


class VerificationPlan(BaseModel):
    """Stage 2 output: what to investigate and how."""

    questions: List[InvestigationQuestion] = Field(default_factory=list)
    image_intent: str = ""  # 图片试图传达什么信息（一句话）
    is_trying_to_be_real: bool = True  # 这张图是否试图让观众相信它是真实的
    risk_assessment: str = ""  # "可能是AI生成"|"可能是篡改"|"可能是真实但误导"|"看起来正常"


# =============================================================================
# Stage 3: VERIFICATION — output
# =============================================================================


class EvidenceItem(BaseModel):
    """A single piece of evidence collected during verification."""

    source: str = ""  # URL or tool name
    summary: str = ""  # 1-2 sentence summary of what was found
    raw_excerpt: str = ""  # 搜索结果/网页的原文关键句，逐字摘抄，不转写（供 judgment 看一手证据）
    direction: str = "neutral"  # "supports"|"refutes"|"neutral"
    quality: str = "moderate"  # "strong"|"moderate"|"weak"
    tool_used: str = ""  # which tool produced this evidence
    related_question: str = ""  # which question_id this answers


class VerificationResult(BaseModel):
    """Stage 3 output: all evidence collected + anomaly findings."""

    evidence: List[EvidenceItem] = Field(default_factory=list)
    visual_anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    authenticity_assessment: str = "uncertain"  # "authentic"|"likely_ai"|"likely_manipulated"|"uncertain"
    key_findings: List[str] = Field(default_factory=list)  # 关键发现的一句话总结


# =============================================================================
# Stage 4: JUDGMENT — output
# =============================================================================


class FinalJudgment(BaseModel):
    """Stage 4 output: the final verdict."""

    verdict: str = "unverifiable"  # "real"|"fake"|"unverifiable"
    confidence: float = 0.5
    reasoning_chain: str = ""  # 完整推理链
    key_evidence: List[str] = Field(default_factory=list)  # 支撑判定的关键证据摘要
    anomalies: List[str] = Field(default_factory=list)  # 发现的异常描述
    overall_assessment: str = ""  # 一段话总结


# =============================================================================
# Aggregate state across all stages
# =============================================================================


@dataclass
class VerificationState:
    """Full state across all 4 stages. Serializable for trajectory export."""

    image_path: str = ""
    image_id: str = ""

    # Stage outputs (filled progressively)
    perception: Optional[PerceptionReport] = None
    plan: Optional[VerificationPlan] = None
    verification: Optional[VerificationResult] = None
    judgment: Optional[FinalJudgment] = None

    # Trajectory (all steps from all stages)
    all_steps: List[Any] = field(default_factory=list)

    # Metadata
    stage_timings: Dict[str, float] = field(default_factory=dict)
    total_tool_calls: int = 0
    llm_api_calls: int = 0  # Total LLM API calls across all stages
    token_usage: Dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    termination: str = ""  # "success"|"timeout"|"error"|"fallback"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full state for trajectory export."""
        # Serialize all_steps (tool calls with results)
        steps_data = []
        for step in self.all_steps:
            step_dict = {
                "round": step.round,
                "action_type": step.action_type,
                "tool_name": step.tool_name,
                "tool_args": step.tool_args,
                "tool_result": step.tool_result[:2000] if step.tool_result else "",
                "tokens": step.tokens,
            }
            if step.thought:
                step_dict["thought"] = step.thought[:200]
            steps_data.append(step_dict)

        return {
            "image_path": self.image_path,
            "image_id": self.image_id,
            "perception": self.perception.model_dump() if self.perception else None,
            "plan": self.plan.model_dump() if self.plan else None,
            "verification": self.verification.model_dump() if self.verification else None,
            "judgment": self.judgment.model_dump() if self.judgment else None,
            "all_steps": steps_data,
            "stage_timings": self.stage_timings,
            "total_tool_calls": self.total_tool_calls,
            "llm_api_calls": self.llm_api_calls,
            "token_usage": self.token_usage,
            "termination": self.termination,
        }

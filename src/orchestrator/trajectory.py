# -*- coding: utf-8 -*-
"""Trajectory data structures for recording ReAct agent execution.

Stores structured per-round records with:
- thought: agent's reasoning
- tool_call: tool name + full arguments
- tool_result: what the tool returned
- answer: final structured answer (last round only)

Compatible with rLLM's format for RL training, and supports
HTML visualization and analysis.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Step:
    """A single round in the agent's reasoning trajectory.

    Each step is a complete (think → act → observe) cycle.
    """

    round: int
    thought: str = ""  # Agent's <think> content
    action_type: str = ""  # "tool_call" | "answer" | "format_error"

    # For tool_call steps
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_args_summary: str = ""  # Short summary for display
    tool_outcome: str = ""  # "success" | "empty" | "error" | "duplicate"
    tool_result: str = ""  # Full tool response text
    evidence_images: List[str] = field(default_factory=list)  # Image URLs from tool results

    # For answer steps
    answer: Optional[Dict[str, Any]] = None

    # Legacy fields (used by agent.py)
    action: Optional[Dict[str, Any]] = None
    observation: str = ""

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "round": self.round,
            "thought": self.thought,
            "action_type": self.action_type,
            "timestamp": self.timestamp,
        }
        if self.action_type == "tool_call":
            d["tool_name"] = self.tool_name
            # Filter out base64 image data from args
            d["tool_args"] = {
                k: ("<image_data>" if isinstance(v, str) and len(v) > 500 and v.startswith("data:") else v)
                for k, v in self.tool_args.items()
            }
            d["tool_args_summary"] = self.tool_args_summary
            d["tool_outcome"] = self.tool_outcome
            d["tool_result"] = self.tool_result[:3000]  # Truncate very long results
            if self.evidence_images:
                d["evidence_images"] = self.evidence_images
        elif self.action_type == "answer":
            d["answer"] = self.answer
        return d


@dataclass
class Trajectory:
    """Complete trajectory of an agent run on one image.

    Designed for three use cases:
    1. Analysis/visualization (HTML trace)
    2. Training data (SFT/RL)
    3. Debugging (full context)
    """

    image_id: str = ""
    image_path: str = ""
    steps: List[Step] = field(default_factory=list)
    final_answer: Optional[Dict[str, Any]] = None
    termination: str = ""  # "answer" | "max_rounds" | "error" | "consecutive_errors" | "api_instability"
    rounds: int = 0
    time_taken: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0})

    # Raw messages (for training format compatibility)
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def add_step(self, **kwargs) -> Step:
        """Add a step. Accepts any Step field as keyword argument."""
        step = Step(**kwargs)
        self.steps.append(step)
        return step

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON output. Includes full structured steps."""
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "termination": self.termination,
            "rounds": self.rounds,
            "time_taken": self.time_taken,
            "token_usage": self.token_usage,
            "num_tool_calls": sum(1 for s in self.steps if s.action_type == "tool_call"),
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
        }

    def to_training_format(self) -> Dict[str, Any]:
        """Convert to rLLM-compatible training format.

        Returns a dict with:
        - messages: assistant-only messages (for SFT)
        - steps: structured (thought, action, observation) tuples (for RL)
        """
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "messages": [
                {k: (v[:2000] if isinstance(v, str) else v) for k, v in m.items()}
                for m in self.messages
                if m.get("role") == "assistant"
            ],
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "termination": self.termination,
            "rounds": self.rounds,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

"""Versioned schemas for training examples derived from canonical traces."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PolicyExample(StrictModel):
    trajectory_version: Literal["ifv-policy-v1"] = "ifv-policy-v1"
    tokenizer_id: str = Field(min_length=1, max_length=200)
    episode_id: str = Field(min_length=1, max_length=200)
    step_id: str = Field(min_length=1, max_length=300)
    example_type: Literal["planning", "react", "reflection", "judgment"]
    runtime_observation_refs: List[str] = Field(default_factory=list, max_length=64)
    policy_input: Dict[str, Any]
    policy_action: Dict[str, Any]
    policy_input_token_ids: List[int] = Field(default_factory=list)
    policy_action_token_ids: List[int] = Field(default_factory=list)
    policy_action_loss_mask: List[int] = Field(default_factory=list)
    action_valid: bool
    terminated: bool
    fatal_boundary: bool = False

    @model_validator(mode="after")
    def validate_token_alignment(self) -> "PolicyExample":
        if len(self.policy_action_token_ids) != len(
            self.policy_action_loss_mask
        ):
            raise ValueError(
                "policy_action_token_ids and policy_action_loss_mask must align"
            )
        if any(value not in {0, 1} for value in self.policy_action_loss_mask):
            raise ValueError("policy_action_loss_mask values must be 0 or 1")
        return self

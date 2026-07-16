"""Deterministic ms-swift Gym smoke plugin.

This is not a trainer or rollout engine. It registers one tiny environment and
uses ms-swift's built-in Gym total_reward path.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from swift.infer_engine.protocol import RolloutInferRequest
from swift.rollout.gym_env import Env, envs
from swift.template import Messages


SYSTEM_PROMPT = """You are the policy model in a deterministic image fact-search
smoke environment. Output exactly one JSON object per turn.

Tool action:
{"type":"tool_call","name":"inspect|search|visit","arguments":{"key":"..."}}

Finish action:
{"type":"finish","verdict":"real|fake|unverifiable"}

Do not emit prose or hidden reasoning."""


def _parse_object(text: str) -> Optional[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class IFVMockSearchEnv(Env):
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.scenario_id = str(env_config.get("scenario_id", ""))
        self.routes = dict(env_config.get("routes", {}))
        self.required_keys = set(str(item) for item in env_config.get("required_keys", []))
        self.target_verdict = str(env_config.get("target_verdict", ""))
        self.visited: set[str] = set()
        self.steps = 0

    async def reset(
        self,
        config: RolloutInferRequest,
    ) -> Tuple[str, Dict[str, Any], str]:
        self.visited.clear()
        self.steps = 0
        observation = (
            "<image>\n"
            f"scenario={self.scenario_id}\n"
            "Investigate the image. Start with a useful tool action."
        )
        return observation, {"scenario_id": self.scenario_id}, SYSTEM_PROMPT

    async def step(
        self,
        action: Messages,
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        self.steps += 1
        content = str(action[-1].get("content", "")) if action else ""
        parsed = _parse_object(content)
        if parsed is None:
            return (
                "Protocol error: output one valid JSON object.",
                -0.25,
                False,
                {"status": "invalid_json", "step": self.steps},
            )
        action_type = str(parsed.get("type", ""))
        if action_type == "finish":
            verdict = str(parsed.get("verdict", ""))
            complete = self.required_keys.issubset(self.visited)
            correct = verdict == self.target_verdict
            reward = 1.0 if complete and correct else 0.0
            return (
                f"terminal complete={complete} correct={correct}",
                reward,
                True,
                {
                    "status": "finished",
                    "complete": complete,
                    "correct": correct,
                    "step": self.steps,
                },
            )
        if action_type != "tool_call":
            return (
                "Protocol error: expected type=tool_call or type=finish.",
                -0.25,
                False,
                {"status": "invalid_type", "step": self.steps},
            )
        name = str(parsed.get("name", ""))
        arguments = parsed.get("arguments")
        key = str(arguments.get("key", "")) if isinstance(arguments, dict) else ""
        route_id = f"{name}:{key}"
        if route_id in self.visited:
            return (
                f"Duplicate route rejected: {route_id}",
                -0.1,
                False,
                {"status": "duplicate", "route_id": route_id, "step": self.steps},
            )
        observation = self.routes.get(route_id)
        if observation is None:
            return (
                f"No result for route {route_id}. Choose another bounded route.",
                -0.05,
                False,
                {"status": "no_result", "route_id": route_id, "step": self.steps},
            )
        self.visited.add(route_id)
        remaining = sorted(self.required_keys - self.visited)
        return (
            f"Tool result for {route_id}: {observation}\nremaining={remaining}",
            0.05,
            False,
            {"status": "ok", "route_id": route_id, "step": self.steps},
        )

    async def close(self) -> None:
        return None


envs["ifv_mock_search"] = IFVMockSearchEnv

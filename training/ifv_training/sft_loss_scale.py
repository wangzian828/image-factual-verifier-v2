"""Loss weighting for full-trajectory IFV Agent SFT.

The teacher's native reasoning remains fully supervised at unit weight.  The
comparatively sparse executable tool calls and final ``<answer>`` contract are
weighted more strongly so their delimiters and payloads are not drowned out by
the much larger reasoning target.

This is registered as an ms-swift external plugin instead of relying on the
built-in ``qwen`` loss scale.  In ms-swift 4.4.2 that preset matches the legacy
``✿FUNCTION✿`` syntax, while Qwen3.5 renders native calls with
``<tool_call><function=...>``.
"""

from __future__ import annotations

from typing import Any


IFV_AGENT_LOSS_SCALE = "ifv_agent"
IFV_AGENT_LOSS_SCALE_SPEC = "ifv_agent+ignore_empty_think"
IFV_REASONING_WEIGHT = 1.0
IFV_TOOL_CALL_WEIGHT = 2.0
IFV_FINAL_ANSWER_WEIGHT = 2.0

# One-value entries are regular expressions in ms-swift's loss-scale engine.
# DOTALL matching is applied by ms-swift, so complete multiline blocks match.
IFV_AGENT_RESPONSE_WEIGHTS = {
    r"<tool_call>.+?</tool_call>": [IFV_TOOL_CALL_WEIGHT],
    r"<answer>.+?</answer>": [IFV_FINAL_ANSWER_WEIGHT],
}


def install_ms_swift_sft_loss_scale() -> None:
    """Register the IFV Agent loss scale in the active ms-swift process."""

    from swift.loss_scale.base import LossScale
    from swift.loss_scale.mapping import loss_scale_map
    from swift.loss_scale.utils import calculate_loss_scale

    class IfvAgentLossScale(LossScale):
        is_binary = False

        def get_loss_scale(
            self,
            context: Any,
            *,
            query: str | None = None,
            **_: Any,
        ) -> tuple[list[Any], list[float]]:
            if isinstance(context, str):
                return calculate_loss_scale(
                    query,
                    context,
                    IFV_AGENT_RESPONSE_WEIGHTS,
                )
            return super().get_loss_scale(context, query=query)

    existing = loss_scale_map.get(IFV_AGENT_LOSS_SCALE)
    if existing is not None:
        if (
            existing.__name__ != IfvAgentLossScale.__name__
            or existing.__module__ != IfvAgentLossScale.__module__
        ):
            raise RuntimeError(
                f"ms-swift loss scale {IFV_AGENT_LOSS_SCALE!r} is already "
                "registered by another implementation"
            )
        return
    loss_scale_map[IFV_AGENT_LOSS_SCALE] = IfvAgentLossScale

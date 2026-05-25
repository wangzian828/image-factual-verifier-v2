from __future__ import annotations

from typing import Any, Dict

from src.tools.base import BaseTool, CallableToolAdapter


class ToolRegistry:
    """Schema-aware tool registration and invocation layer."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def register_callable(
        self,
        name: str,
        fn,
        *,
        description: str = "",
        parameters: Dict[str, Any] | None = None,
    ) -> None:
        self.register_tool(
            CallableToolAdapter(
                name=name,
                fn=fn,
                description=description,
                parameters=parameters,
            )
        )

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def schemas(self) -> list[Dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]

    def invoke(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        tool = self.get(name)
        try:
            output = tool.call(dict(kwargs))
            return {
                "tool_name": name,
                "tool_input": dict(kwargs),
                "tool_output": output,
                "status": "success",
                "error": None,
            }
        except Exception as exc:
            return {
                "tool_name": name,
                "tool_input": dict(kwargs),
                "tool_output": None,
                "status": "error",
                "error": str(exc),
            }

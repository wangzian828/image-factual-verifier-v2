from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    @abstractmethod
    def call(self, params: Dict[str, Any]) -> Any:
        raise NotImplementedError

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class CallableToolAdapter(BaseTool):
    def __init__(self, *, name: str, fn, description: str = "", parameters: Dict[str, Any] | None = None):
        self.name = name
        self.description = description or f"Callable adapter for {name}."
        self.parameters = parameters or {
            "type": "object",
            "properties": {},
            "required": [],
        }
        self._fn = fn

    def call(self, params: Dict[str, Any]) -> Any:
        return self._fn(**params)

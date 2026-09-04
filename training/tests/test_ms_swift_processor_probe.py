from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe"
    / "verify_ms_swift_agent_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("verify_ms_swift_agent_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_verify_rendered_tool_calls_accepts_qwen_function_xml() -> None:
    messages = [
        {
            "role": "tool_call",
            "content": (
                '{"name":"text_search","arguments":'
                '"{\\"queries\\":\\"rail station\\",'
                '\\"investigation_progress\\":{\\"status\\":\\"investigating\\"}}"}'
            ),
        }
    ]
    decoded = """
    <tool_call>
    <function=text_search>
    <parameter=queries>
    rail station
    </parameter>
    <parameter=investigation_progress>
    {"status":"investigating"}
    </parameter>
    </function>
    </tool_call>
    """

    assert MODULE._verify_rendered_tool_calls(messages, decoded) == 1


def test_verify_rendered_tool_calls_rejects_missing_parameter() -> None:
    messages = [
        {
            "role": "tool_call",
            "content": (
                '{"name":"text_search","arguments":'
                '"{\\"queries\\":\\"rail station\\"}"}'
            ),
        }
    ]

    with pytest.raises(ValueError, match="text_search.queries"):
        MODULE._verify_rendered_tool_calls(
            messages,
            "<function=text_search></function>",
        )

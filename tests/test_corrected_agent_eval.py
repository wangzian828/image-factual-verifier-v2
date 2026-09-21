import json

from scripts.server.control_corrected_agent_eval import (
    extraction_protocol_report,
    normalized_command,
)


def test_normalized_command_uses_same_parser_for_every_weight(tmp_path):
    command = [
        "python",
        "vllm",
        "serve",
        "/old",
        "--served-model-name",
        "old-alias",
        "--tool-call-parser",
        "legacy",
        "--tool-parser-plugin",
        "/legacy.py",
    ]
    result = normalized_command(command, tmp_path / "model")
    assert result[3] == str(tmp_path / "model")
    assert result[result.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert "--tool-parser-plugin" not in result


def test_extract_protocol_requires_qwen_local_and_exact_model(tmp_path):
    run = tmp_path / "smoke"
    run.mkdir()
    trace = run / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "state": {
                    "tool_observations": [
                        {
                            "subcalls": [
                                {
                                    "kind": "page_extract",
                                    "provider": "qwen_local",
                                    "model": "policy-under-test",
                                    "status": "success",
                                    "thinking_enabled": False,
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    selected = {"a": ({"trace_path": trace.name}, run)}
    report = extraction_protocol_report(selected, ["a"], "policy-under-test")
    assert report["passed"] is True
    assert report["page_extracts"] == 1

    mismatch = extraction_protocol_report(selected, ["a"], "different-policy")
    assert mismatch["passed"] is False
    assert mismatch["mismatches"][0]["model"] == "policy-under-test"

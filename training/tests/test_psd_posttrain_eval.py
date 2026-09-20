import json

from scripts.server.control_psd_posttrain_eval import (
    MODEL_ALIAS,
    adapter_command,
    smoke_anomaly_report,
)


def test_adapter_command_binds_lora_and_uses_formal_tool_parser(tmp_path):
    adapter = tmp_path / "adapter"
    command = ["python", "vllm", "serve", "base-model", "--served-model-name", MODEL_ALIAS,
        "--tool-call-parser", "ifv_psd_qwen3_single", "--tool-parser-plugin", "old.py",
        "--logits-processors", "scripts.server.psd_qwen_thinking:PSDThinkingBudget"]
    result = adapter_command(command, adapter)
    assert result[result.index("--served-model-name") + 1] == MODEL_ALIAS + "-base"
    assert result[result.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert "--tool-parser-plugin" not in result
    assert result[result.index("--lora-modules") + 1] == f"{MODEL_ALIAS}={adapter}"
    assert "--enable-tower-connector-lora" in result
    assert "scripts.server.psd_qwen_thinking:PSDThinkingBudget" in result


def _write_smoke(tmp_path, *, report="Normal grounded report.", finish="stop", repeats=1):
    directory = tmp_path / "smoke"
    (directory / "traces").mkdir(parents=True)
    trace = {"state": {"all_steps": [
        {"action_type": "tool_call", "tool_name": "text_search", "tool_input": {"q": "x"},
         "metadata": {"finish_reason": finish}}
        for _ in range(repeats)
    ]}}
    (directory / "traces/case-1.json").write_text(json.dumps(trace), encoding="utf-8")
    row = {"case_id": "case-1", "status": "success", "termination": "success",
        "verdict": "real", "trace_path": "traces/case-1.json",
        "fact_check_report": report, "total_tool_calls": repeats}
    (directory / "run_results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return directory


def test_smoke_anomaly_report_accepts_normal_output(tmp_path):
    directory = _write_smoke(tmp_path)
    result = smoke_anomaly_report(directory, ["case-1"])
    assert result["passed"] is True
    assert result["scope"].startswith("engineering anomalies only")


def test_smoke_anomaly_report_rejects_length_and_tool_loop(tmp_path):
    directory = _write_smoke(tmp_path, finish="length", repeats=8)
    result = smoke_anomaly_report(directory, ["case-1"])
    assert result["passed"] is False
    assert set(result["records"][0]["issues"]) >= {
        "length_or_abort_finish", "repeated_identical_tool_loop"}

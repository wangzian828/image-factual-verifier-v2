from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_endpoint_probe_covers_all_qwen3vl_serving_gates() -> None:
    source = (ROOT / "scripts/probe/vllm_qwen3vl_endpoint.py").read_text(
        encoding="utf-8"
    )

    assert 'f"{origin}/health"' in source
    assert 'f"{base_url}/models"' in source
    assert 'f"{origin}/tokenizer_info"' in source
    assert 'message.get("reasoning") or message.get("reasoning_content")' in source
    assert '"type": "image_url"' in source
    assert '"type": "json_schema"' in source
    assert 'name="image_account_planning"' in source
    assert "_validate_planning_semantics" in source
    assert '"tool_choice": "required"' in source
    assert '"role": "tool"' in source
    assert '"tool_choice": "none"' in source
    assert 'default=131072' in source
    assert 'default=8' in source


def test_planning_probe_schema_has_runtime_contract_shape() -> None:
    schema = json.loads(
        (ROOT / "configs/serve/image-account-planning.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["required"] == [
        "account_summary",
        "image_claims",
        "search_hypotheses",
    ]
    assert schema["additionalProperties"] is False
    claim = schema["properties"]["image_claims"]
    hypothesis = schema["properties"]["search_hypotheses"]
    assert claim["minItems"] == 1 and claim["maxItems"] == 3
    assert hypothesis["minItems"] == 1 and hypothesis["maxItems"] == 6


def test_environment_verifier_pins_protocol_critical_packages() -> None:
    source = (ROOT / "scripts/probe/verify_vllm_environment.py").read_text(
        encoding="utf-8"
    )

    for pair in (
        '"vllm": "0.11.2"',
        '"torch": "2.9.0"',
        '"transformers": "4.57.6"',
        '"xformers": "0.0.33.post1"',
        '"flashinfer-python": "0.5.2"',
        '"xgrammar": "0.1.25"',
        '"llguidance": "1.3.0"',
    ):
        assert pair in source
    assert "Qwen3VLForConditionalGeneration" in source
    assert '"qwen3" not in ReasoningParserManager.list_registered()' in source
    assert '"qwen3_xml" not in ToolParserManager.list_registered()' in source


def test_nccl_probe_exercises_real_tensor_parallel_collective() -> None:
    source = (ROOT / "scripts/probe/verify_nccl_tensor_parallel.py").read_text(
        encoding="utf-8"
    )

    assert 'dist.init_process_group("nccl")' in source
    assert "torch.cuda.set_device(local_rank)" in source
    assert "dist.all_reduce(value)" in source
    assert "dist.all_gather_object" in source
    assert "dist.destroy_process_group()" in source


def test_chat_template_keeps_generated_think_start_for_vllm_parser() -> None:
    source = (ROOT / "scripts/serve/prepare_qwen3vl_chat_template.py").read_text(
        encoding="utf-8"
    )

    assert "GENERATION_PREFILL" in source
    assert "PARSER_COMPATIBLE_PREFILL" in source
    assert "template.count(GENERATION_PREFILL) != 1" in source
    assert "removed_generation_think_prefill" in source

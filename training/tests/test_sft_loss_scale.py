from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from ifv_training.sft_loss_scale import (
    IFV_AGENT_LOSS_SCALE_SPEC,
    IFV_AGENT_RESPONSE_WEIGHTS,
    IFV_FINAL_ANSWER_WEIGHT,
    IFV_REASONING_WEIGHT,
    IFV_TOOL_CALL_WEIGHT,
)


def test_ifv_agent_weighting_preserves_think_and_emphasizes_actions() -> None:
    assert IFV_AGENT_LOSS_SCALE_SPEC == "ifv_agent+ignore_empty_think"
    assert IFV_REASONING_WEIGHT == 1.0
    assert IFV_TOOL_CALL_WEIGHT == 2.0
    assert IFV_FINAL_ANSWER_WEIGHT == 2.0
    assert any(
        re.fullmatch(pattern, "<tool_call>\ncall\n</tool_call>", re.DOTALL)
        and weights == [2.0]
        for pattern, weights in IFV_AGENT_RESPONSE_WEIGHTS.items()
    )
    assert any(
        re.fullmatch(pattern, "<answer>\nreport\n</answer>", re.DOTALL)
        and weights == [2.0]
        for pattern, weights in IFV_AGENT_RESPONSE_WEIGHTS.items()
    )
    assert not any(
        re.fullmatch(pattern, "<think>reasoning</think>", re.DOTALL)
        for pattern in IFV_AGENT_RESPONSE_WEIGHTS
    )


def test_sft_launcher_loads_the_loss_scale_plugin_and_h20_profile() -> None:
    training = Path(__file__).resolve().parents[1]
    launcher = (training / "scripts/train/run_sft.sh").read_text(encoding="utf-8")
    profile = (
        training
        / "configs/sft/qwen3.5-full-4gpu-h20-fsdp2-sp4-flash-128k-agent-v2.env"
    ).read_text(encoding="utf-8")
    canary = (
        training
        / "configs/sft/qwen3.5-full-1step-4gpu-h20-fsdp2-sp4-flash-128k-agent-v2-canary.env"
    ).read_text(encoding="utf-8")

    assert "--external_plugins" in launcher
    assert '--model_type "$IFV_MODEL_FAMILY"' in launcher
    assert "IFV_PACKING_LENGTH requires IFV_PACKING=true" in launcher
    assert "trainer_truncation_strategy=delete" in launcher
    assert "--save_only_model" in launcher
    assert "IFV_LOSS_SCALE=ifv_agent+ignore_empty_think" in profile
    assert "ifv_sft_agent_plugin.py" in profile
    assert "IFV_SEQUENCE_PARALLEL_SIZE=4" in profile
    assert "IFV_PACKING_LENGTH=120000" in profile
    assert "IFV_GRADIENT_CHECKPOINTING=false" in profile
    assert "IFV_VIT_GRADIENT_CHECKPOINTING=false" in profile
    assert "IFV_USE_LIGER_KERNEL=true" in profile
    assert "IFV_SAVE_ONLY_MODEL=true" in profile
    assert "qwen3.5-full-4gpu-h20-fsdp2-sp4-flash-128k-agent-v2.env" in canary
    assert "IFV_MAX_STEPS=1" in canary
    assert "IFV_SAVE_STRATEGY=no" in canary


def test_formal_h20_command_replaces_historical_loss_with_verified_plugin(
    tmp_path: Path,
) -> None:
    training = Path(__file__).resolve().parents[1]
    planner_path = training / "scripts/h20/prepare_formal_sft.py"
    spec = importlib.util.spec_from_file_location("prepare_formal_sft", planner_path)
    assert spec is not None and spec.loader is not None
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)

    accepted = {
        "--model": "/models/qwen",
        "--model_type": "qwen3_5",
        "--tuner_type": "full",
        "--fsdp": "fsdp2",
        "--sequence_parallel_size": "4",
        "--max_length": "131072",
        "--packing": "true",
        "--packing_length": "120000",
        "--padding_free": "true",
        "--attn_impl": "flash_attn",
        "--freeze_llm": "false",
        "--freeze_vit": "false",
        "--freeze_aligner": "false",
        "--per_device_train_batch_size": "1",
        "--gradient_accumulation_steps": "1",
        "--truncation_strategy": "delete",
        "--loss_scale": "ignore_empty_think",
        "--enable_thinking": "false",
        "--add_non_thinking_prefix": "false",
        "--max_pixels": "262144",
        "--split_dataset_ratio": "0",
        "--torch_dtype": "bfloat16",
        "--bf16": "true",
        "--use_logits_to_keep": "false",
        "--lazy_tokenize": "false",
        "--learning_rate": "1e-5",
    }
    benchmark = [
        "swift",
        "sft",
        *(value for item in accepted.items() for value in item),
    ]
    command = planner.command_from_benchmark(
        benchmark,
        tmp_path / "train.jsonl",
        tmp_path / "output",
        save_only_model=True,
        save_steps=400,
        save_total_limit=3,
    )
    options = dict(zip(command[2::2], command[3::2]))

    assert options["--loss_scale"] == "ifv_agent+ignore_empty_think"
    assert options["--strict"] == "true"
    assert options["--external_plugins"] == str(planner.SFT_AGENT_PLUGIN)
    assert options["--save_only_model"] == "true"
    assert options["--save_steps"] == "400"
    assert options["--save_total_limit"] == "3"

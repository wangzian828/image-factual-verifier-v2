from __future__ import annotations

import importlib.util
from argparse import Namespace
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


def test_template_contract_matches_training_controls() -> None:
    args = Namespace(
        max_context=131072,
        truncation_strategy="raise",
        max_pixels=262144,
        padding_free=True,
        sequence_parallel_size=8,
        loss_scale="ifv_agent+ignore_empty_think",
        enable_thinking=False,
        add_non_thinking_prefix=False,
    )

    assert MODULE._template_kwargs(args) == {
        "max_length": 131072,
        "truncation_strategy": "raise",
        "max_pixels": 262144,
        "padding_free": True,
        "sequence_parallel_size": 8,
        "loss_scale": "ifv_agent+ignore_empty_think",
        "enable_thinking": False,
        "add_non_thinking_prefix": False,
    }


def test_file_record_binds_processor_report_to_exact_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"messages": []}\n', encoding="utf-8")

    record = MODULE._file_record(dataset)

    assert record["path"] == str(dataset.resolve())
    assert record["size"] == dataset.stat().st_size
    assert len(record["sha256"]) == 64


def test_boundary_summary_counts_rows_and_selects_nearest_candidates() -> None:
    rows = [
        {"row_index": 0, "input_tokens": 30_000},
        {"row_index": 1, "input_tokens": 65_000},
        {"row_index": 2, "input_tokens": 121_000},
    ]

    summary = MODULE._boundary_summary(rows, boundaries=(32_768, 120_000))

    assert summary[0]["rows_at_or_above"] == 2
    assert summary[0]["closest_at_or_below"]["row_index"] == 0
    assert summary[1]["rows_at_or_above"] == 1
    assert summary[1]["closest_at_or_above"]["row_index"] == 2


def test_count_labeled_subsequences_distinguishes_masks() -> None:
    input_ids = [1, 2, 3, 1, 2, 3]
    labels = [1, 2, 3, -100, -100, -100]

    assert MODULE._count_labeled_subsequences(input_ids, labels, [1, 2, 3]) == (
        1,
        1,
    )


def test_subsequence_search_skips_false_prefixes_and_overlaps() -> None:
    sequence = [1, 9, 1, 2, 1, 2, 3, 1, 2, 3]

    assert list(MODULE._subsequence_starts(sequence, [1, 2, 3])) == [4, 7]
    assert MODULE._contains_subsequence(sequence, [1, 2, 3]) is True
    assert MODULE._contains_subsequence(sequence, [2, 3, 4]) is False


def test_labeled_subsequence_search_checks_every_real_match() -> None:
    input_ids = [7, 8, 7, 8]
    labels = [-100, -100, 7, 8]

    assert MODULE._contains_supervised_subsequence(input_ids, labels, [7, 8])
    assert MODULE._contains_masked_subsequence(input_ids, labels, [7, 8])
    assert MODULE._count_labeled_subsequences(input_ids, labels, [7, 8]) == (
        1,
        1,
    )


def test_thought_loss_contract_counts_distinct_markers() -> None:
    class Tokenizer:
        @staticmethod
        def encode(_text: str, add_special_tokens: bool = False) -> list[int]:
            return [7]

    messages = [
        {"role": "assistant", "content": "<think>a</think>"},
        {"role": "assistant", "content": "<think>b</think>", "loss": False},
    ]

    assert MODULE._verify_thought_loss_contract(
        messages,
        [7, 7],
        [7, -100],
        [1.0, 0.0],
        Tokenizer(),
        supervised_weight=1.0,
    ) == (1, 1)


def test_tag_block_contract_rejects_silent_unweighted_tool_calls() -> None:
    class Tokenizer:
        TOKENS = {"<tool_call>": [1], "</tool_call>": [4]}

        @classmethod
        def encode(cls, text: str, add_special_tokens: bool = False) -> list[int]:
            return cls.TOKENS[text]

    with pytest.raises(ValueError, match="uniformly weighted 2.0"):
        MODULE._verify_tag_block_loss_contract(
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1.0, 1.0, 1.0, 1.0],
            Tokenizer(),
            start_tag="<tool_call>",
            end_tag="</tool_call>",
            expected_supervised=1,
            expected_masked_minimum=0,
            supervised_weight=2.0,
        )


def test_tag_block_contract_accepts_weighted_and_masked_blocks() -> None:
    class Tokenizer:
        TOKENS = {"<answer>": [1], "</answer>": [4]}

        @classmethod
        def encode(cls, text: str, add_special_tokens: bool = False) -> list[int]:
            return cls.TOKENS[text]

    assert MODULE._verify_tag_block_loss_contract(
        [1, 2, 4, 1, 3, 4],
        [1, 2, 4, -100, -100, -100],
        [2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
        Tokenizer(),
        start_tag="<answer>",
        end_tag="</answer>",
        expected_supervised=1,
        expected_masked_minimum=1,
        supervised_weight=2.0,
    ) == (1, 1)

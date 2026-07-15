from __future__ import annotations

import argparse

from scripts import run_real_canary


def test_explicit_canary_cases_are_forwarded_in_order() -> None:
    args = argparse.Namespace(
        benchmark="release/runtime_input/cases.jsonl",
        output_dir="run",
        model="gemini-3.5-flash",
        limit=2,
        case_id=["case_refuted", "case_supported"],
        source_access_policy=None,
    )

    command = run_real_canary._command(args)

    assert command[command.index("--limit") + 1] == "2"
    assert command[-4:] == [
        "--case-id",
        "case_refuted",
        "--case-id",
        "case_supported",
    ]

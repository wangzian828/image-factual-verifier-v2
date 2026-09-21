import asyncio
import json

from scripts.server.stream_agent_judges import (
    LedgerTail,
    StreamingJudge,
    compact_record,
    terminal_judge,
)


def test_ledger_tail_reads_only_new_complete_lines(tmp_path):
    path = tmp_path / "run_results.jsonl"
    path.write_bytes(b'{"case_id":"a"}\n{"case_id":"b"')
    tail = LedgerTail()
    assert tail.read_new(path) == [{"case_id": "a"}]
    path.write_bytes(path.read_bytes() + b'}\n')
    assert tail.read_new(path) == [{"case_id": "b"}]
    assert tail.read_new(path) == []


def test_compact_receipt_drops_reconstructable_payloads(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    record = {
        "case_id": "a",
        "status": "completed",
        "candidate_output": {"large": "payload"},
        "candidate_answer": {"verdict": "real"},
        "private_gold": {"secret": True},
        "judge_output_text": "raw",
        "judge_native_thought": "hidden",
        "quality_bucket": "strong",
    }
    compact = compact_record(record, source_trace=trace)
    assert "candidate_output" not in compact
    assert "private_gold" not in compact
    assert compact["source_trace"]["size"] == 2
    assert compact["large_payload_hashing"] is False


def test_terminal_judge_requires_valid_structured_fields():
    good = {
        "status": "completed",
        "judge_json_parse_error": None,
        "quality_bucket": "strong",
        "fact_alignment": "same_fact",
        "reason_quality": "decisive_and_grounded",
    }
    assert terminal_judge(good) is True
    assert terminal_judge({**good, "quality_bucket": None}) is False


def test_existing_failed_response_is_not_replayed(tmp_path):
    judge = StreamingJudge.__new__(StreamingJudge)
    judge.output = tmp_path
    judge.index = {"a": 1}
    directory = judge.case_dir("a")
    directory.mkdir(parents=True)
    (directory / "attempt-01.intent.json").write_text("{}", encoding="utf-8")
    (directory / "attempt-01.result.json").write_text("{}", encoding="utf-8")
    (directory / "failed.json").write_text("{}", encoding="utf-8")

    class Client:
        async def create(self, **_kwargs):
            raise AssertionError("a completed provider attempt must not be replayed")

    asyncio.run(
        judge.judge_one(
            Client(),
            "a",
            {"source_trace_path": str(tmp_path / "unused.json")},
            tmp_path,
        )
    )

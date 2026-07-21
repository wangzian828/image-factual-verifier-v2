import asyncio
import json
from pathlib import Path

from src.orchestrator.runtime_events import (
    CANONICAL_TRACE_SCHEMA_VERSION,
    CaseRuntimeStore,
    atomic_write_json,
    bind_case_runtime_store,
    current_case_runtime_store,
    reset_case_runtime_store,
)


def test_atomic_json_replaces_complete_document(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    atomic_write_json(path, {"schema_version": CANONICAL_TRACE_SCHEMA_VERSION, "value": 1})
    atomic_write_json(path, {"schema_version": CANONICAL_TRACE_SCHEMA_VERSION, "value": 2})

    assert json.loads(path.read_text(encoding="utf-8"))["value"] == 2
    assert not list(tmp_path.glob(".tmp-*"))


def test_context_manifest_reconstructs_exact_image_request(tmp_path: Path) -> None:
    store = CaseRuntimeStore(tmp_path, case_id="case/with/a/long/name" * 4)
    # Keep this large enough that externalizing media is meaningfully smaller
    # than serializing its base64 representation as ordinary request text.
    image_bytes = b"not-a-real-png-but-stable" * 256
    import base64

    encoded = base64.b64encode(image_bytes).decode("ascii")
    request_id = store.context_ledger.begin_request(
        stage="planning",
        lifecycle_kind="standalone_request",
        system_instruction="short prompt",
        input_payload=[
            {"type": "text", "text": "inspect"},
            {"type": "image", "mime_type": "image/png", "data": encoded},
        ],
        tools=[],
        response_format={"type": "text"},
        max_output_tokens=8192,
        model="controlled",
        prompt_version="planning-v1",
    )
    manifest = store.context_ledger.complete_request(
        request_id,
        usage={"total_input_tokens": 20, "total_output_tokens": 3},
        interaction_id="interaction-1",
    )

    reconstructed = store.context_ledger.reconstruct_request(request_id)
    assert reconstructed["system_prompt"] == "short prompt"
    assert reconstructed["input_payload"][1]["data"] == encoded
    assert manifest["provider_input_tokens"] == 20
    assert manifest["image_count"] == 1
    assert manifest["media_bytes"] == len(image_bytes)
    assert manifest["serialized_input_chars"] > manifest["explicit_input_chars"]
    assert manifest["parent_interaction_id"] is None
    assert manifest["max_output_tokens"] == 8192


def test_event_reader_ignores_torn_last_line(tmp_path: Path) -> None:
    store = CaseRuntimeStore(tmp_path, case_id="case")
    store.append_event("valid", {"value": 1})
    with store.events_path.open("ab") as handle:
        handle.write(b'{"torn":')

    events = store.read_events()
    assert [event["event_type"] for event in events][-1] == "valid"


def test_case_runtime_context_is_task_local(tmp_path: Path) -> None:
    first = CaseRuntimeStore(tmp_path / "first", case_id="first")
    second = CaseRuntimeStore(tmp_path / "second", case_id="second")

    async def observe(store: CaseRuntimeStore) -> str:
        token = bind_case_runtime_store(store)
        try:
            await asyncio.sleep(0)
            current = current_case_runtime_store()
            return current.case_id if current is not None else ""
        finally:
            reset_case_runtime_store(token)

    async def run_both() -> list[str]:
        return list(await asyncio.gather(observe(first), observe(second)))

    assert asyncio.run(run_both()) == [
        "first",
        "second",
    ]


def test_archive_recall_then_exact_read_preserves_raw_result(tmp_path: Path) -> None:
    store = CaseRuntimeStore(tmp_path, case_id="archive-case")
    descriptor = store.archive_tool_result(
        stage="verification",
        action_index=3,
        tool_name="visit",
        tool_args={"url": "https://example.org/event", "__question_id": "task-1"},
        tool_result=json.dumps(
            {
                "status": "success",
                "evidence": "The official account identifies the ceremonial coach.",
            }
        ),
    )
    store.bind_archive_lineage(
        descriptor["memory_id"],
        {
            "task_id": "task-1",
            "created_evidence_ids": ["evidence-1"],
        },
    )

    recalled = store.recall_archive(
        query="ceremonial coach",
        filters={"task_id": "task-1", "tool": "visit"},
        top_k=3,
    )
    assert recalled["candidate_count"] == 1
    assert recalled["candidates"][0]["memory_id"] == descriptor["memory_id"]

    read = store.read_archive_item(
        descriptor["memory_id"],
        offset=0,
        length=30,
    )
    assert read["status"] == "success"
    assert read["content"] == json.dumps(
        {
            "status": "success",
            "evidence": "The official account identifies the ceremonial coach.",
        }
    )[:30]
    assert read["artifact"]["sha256"] == descriptor["sha256"]


def test_archive_read_rejects_unknown_memory_id(tmp_path: Path) -> None:
    store = CaseRuntimeStore(tmp_path, case_id="archive-case")
    assert store.read_archive_item("memory-missing")["status"] == "error"

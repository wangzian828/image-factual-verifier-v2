from scripts.monitor_runtime_events import _new_summaries, summarize_event


def test_monitor_summarizes_request_budget_tokens_and_error() -> None:
    started = {
        "sequence": 4,
        "event_type": "context_request_started",
        "payload": {
            "request_id": "req-000001",
            "stage": "verification",
            "explicit_input_tokens_estimate": 32000,
            "max_output_tokens": 8192,
        },
    }
    completed = {
        "sequence": 5,
        "event_type": "context_request_completed",
        "payload": {
            "request_id": "req-000001",
            "status": "error",
            "provider_input_tokens": None,
            "provider_output_tokens": None,
            "error": "ReadTimeout",
        },
    }

    assert "max_output=8192" in str(summarize_event(started))
    assert "error=ReadTimeout" in str(summarize_event(completed))
    latest, lines = _new_summaries([started, completed], after=4)
    assert latest == 5
    assert len(lines) == 1

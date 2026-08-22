"""Standalone contract tests for the Gemini Interactions REST client."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

import httpx
import pytest

from src.integrations.gemini import (
    DEFAULT_INTERACTIONS_URL,
    RETRYABLE_HTTP_STATUSES,
    GeminiInteractionsClient,
    GeminiInteractionsError,
    GeminiInteractionsHTTPError,
    GeminiInteractionsResponseError,
    extract_function_calls,
    extract_images,
    extract_text,
    extract_videos,
)
from src.orchestrator.llm_backend import APIBackend


TEST_API_KEY = "contract-test-key"


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def set_test_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", TEST_API_KEY)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_INTERACTIONS_URL", raising=False)


def test_client_accepts_credentials_from_environment_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "api_key" not in inspect.signature(GeminiInteractionsClient).parameters
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(GeminiInteractionsError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
        GeminiInteractionsClient()

    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    monkeypatch.setenv("GOOGLE_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("GEMINI_INTERACTIONS_URL", "   ")
    client = GeminiInteractionsClient()
    assert client.base_url == DEFAULT_INTERACTIONS_URL
    run(client.aclose())


def test_create_posts_generic_interaction_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_test_key(monkeypatch)
    captured: list[httpx.Request] = []

    async def scenario() -> dict[str, Any]:
        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                200,
                request=request,
                json={"id": "interaction-1", "status": "completed"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = GeminiInteractionsClient(client=http)
            return await client.create(
                model="gemini-test-model",
                input=[{"type": "text", "text": "verify this image"}],
                system_instruction="Use evidence.",
                tools=[
                    {
                        "type": "function",
                        "name": "search",
                        "parameters": {"type": "object"},
                    }
                ],
                previous_interaction_id="interaction-0",
                generation_config={"temperature": 0.0},
                user_metadata={"stage": "verification"},
            )

    result = run(scenario())

    assert result == {"id": "interaction-1", "status": "completed"}
    assert len(captured) == 1
    request = captured[0]
    assert str(request.url) == DEFAULT_INTERACTIONS_URL
    assert request.method == "POST"
    assert request.headers["x-goog-api-key"] == TEST_API_KEY
    assert request.headers["content-type"] == "application/json"
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {
        "model": "gemini-test-model",
        "input": [{"type": "text", "text": "verify this image"}],
        "stream": False,
        "system_instruction": "Use evidence.",
        "tools": [
            {
                "type": "function",
                "name": "search",
                "parameters": {"type": "object"},
            }
        ],
        "previous_interaction_id": "interaction-0",
        "generation_config": {"temperature": 0.0},
        "background": False,
        "store": True,
        "user_metadata": {"stage": "verification"},
    }


def test_create_rejects_unknown_fields_and_mixed_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_test_key(monkeypatch)
    client = GeminiInteractionsClient()
    with pytest.raises(ValueError, match="Unsupported Gemini Interactions"):
        run(client.create(model="model", input="prompt", metadata={"bad": True}))
    with pytest.raises(ValueError, match="cannot mix Step and Content"):
        run(
            client.create(
                model="model",
                input=[
                    {
                        "type": "function_result",
                        "name": "search",
                        "call_id": "call-1",
                        "result": [{"type": "text", "text": "ok"}],
                    },
                    {"type": "text", "text": "finish now"},
                ],
            )
        )
    run(client.aclose())


def test_create_accepts_function_result_followed_by_user_input_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_test_key(monkeypatch)
    captured: list[dict[str, Any]] = []

    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                request=request,
                json={"id": "interaction-2", "status": "completed"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = GeminiInteractionsClient(client=http)
            await client.create(
                model="gemini-test-model",
                previous_interaction_id="interaction-1",
                input=[
                    {
                        "type": "function_result",
                        "name": "search",
                        "call_id": "call-1",
                        "result": [{"type": "text", "text": "tool result"}],
                    },
                    {
                        "type": "user_input",
                        "content": [
                            {"type": "text", "text": "Review current evidence."}
                        ],
                    },
                ],
            )

    run(scenario())

    assert captured[0]["input"][1] == {
        "type": "user_input",
        "content": [{"type": "text", "text": "Review current evidence."}],
    }


def test_image_and_video_helpers_build_media_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_test_key(monkeypatch)
    requests: list[dict[str, Any]] = []

    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                request=request,
                json={
                    "id": f"interaction-{len(requests)}",
                    "status": "completed",
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = GeminiInteractionsClient(client=http)
            await client.generate_image(
                model="gemini-image-model",
                prompt="A source photograph",
                aspect_ratio="4:3",
                image_size="2K",
            )
            await client.generate_video(
                model="gemini-video-model",
                prompt="A short camera pan",
                aspect_ratio="9:16",
                task="image_to_video",
                generation_config={
                    "temperature": 0.2,
                    "video_config": {"duration_seconds": 5},
                },
            )

    run(scenario())

    assert requests[0]["response_format"] == {
        "type": "image",
        "aspect_ratio": "4:3",
        "image_size": "2K",
    }
    assert requests[1]["response_format"] == {
        "type": "video",
        "delivery": "uri",
        "aspect_ratio": "9:16",
    }
    assert requests[1]["generation_config"] == {
        "temperature": 0.2,
        "video_config": {
            "duration_seconds": 5,
            "task": "image_to_video",
        },
    }
    assert requests[1]["background"] is False
    assert requests[1]["store"] is True
    assert requests[1]["stream"] is False


def test_retries_only_documented_http_statuses_with_delay_and_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_test_key(monkeypatch)
    statuses = sorted(RETRYABLE_HTTP_STATUSES)
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario() -> dict[str, Any]:
        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            status = statuses[attempts] if attempts < len(statuses) else 200
            attempts += 1
            return httpx.Response(
                status,
                request=request,
                json={
                    "id": "interaction-retried",
                    "status": "completed",
                    "attempt": attempts,
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = GeminiInteractionsClient(
                client=http,
                max_retries=len(statuses),
                retry_delay=3.0,
                retry_jitter=2.0,
                sleep=fake_sleep,
                random_uniform=lambda _start, _end: 0.75,
            )
            return await client.create(model="model", input="prompt")

    assert run(scenario()) == {
        "id": "interaction-retried",
        "status": "completed",
        "attempt": len(statuses) + 1,
    }
    assert attempts == len(statuses) + 1
    assert delays == [3.75, 6.75, 12.75, 24.75, 48.75]


@pytest.mark.parametrize("status_code", [400, 408, 501])
def test_does_not_retry_other_http_statuses(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    set_test_key(monkeypatch)
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(
                status_code,
                request=request,
                text='{"error":"request rejected"}',
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = GeminiInteractionsClient(
                client=http,
                max_retries=8,
                sleep=fake_sleep,
            )
            await client.create(model="model", input="prompt")

    with pytest.raises(GeminiInteractionsHTTPError) as error:
        run(scenario())

    assert attempts == 1
    assert delays == []
    assert error.value.status_code == status_code
    assert error.value.response_body == '{"error":"request rejected"}'
    assert "request rejected" in str(error.value)


@pytest.mark.parametrize(
    "error_code",
    ["invalid_request", "malformed_tool_call"],
)
def test_retries_provider_replayable_bad_request_once(
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
) -> None:
    set_test_key(monkeypatch)
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario() -> dict[str, Any]:
        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                message = (
                    "Request contains an invalid argument."
                    if error_code == "invalid_request"
                    else "Model generated invalid JSON syntax."
                )
                return httpx.Response(
                    400,
                    request=request,
                    json={
                        "error": {
                            "message": message,
                            "code": error_code,
                        }
                    },
                )
            return httpx.Response(
                200,
                request=request,
                json={
                    "id": "interaction-replayed",
                    "status": "completed",
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = GeminiInteractionsClient(
                client=http,
                max_retries=8,
                retry_delay=1.0,
                retry_jitter=0.0,
                sleep=fake_sleep,
            )
            return await client.create(model="model", input="prompt")

    assert run(scenario())["id"] == "interaction-replayed"
    assert attempts == 2
    assert delays == [1.0]


def test_persistent_replayable_bad_request_is_retried_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_test_key(monkeypatch)
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(
                400,
                request=request,
                json={
                    "error": {
                        "message": "Request contains an invalid argument.",
                        "code": "invalid_request",
                    }
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = GeminiInteractionsClient(
                client=http,
                max_retries=8,
                retry_delay=1.0,
                retry_jitter=0.0,
                sleep=fake_sleep,
            )
            await client.create(model="model", input="prompt")

    with pytest.raises(GeminiInteractionsHTTPError) as error:
        run(scenario())

    assert attempts == 2
    assert delays == [1.0]
    assert error.value.retry_attempts == 1


def test_retries_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    set_test_key(monkeypatch)
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario() -> dict[str, Any]:
        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("connection failed", request=request)
            return httpx.Response(
                200,
                request=request,
                json={"id": "interaction-transport", "status": "completed"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = GeminiInteractionsClient(
                client=http,
                max_retries=1,
                retry_delay=1.0,
                retry_jitter=0.5,
                sleep=fake_sleep,
                random_uniform=lambda _start, _end: 0.25,
            )
            return await client.create(model="model", input="prompt")

    assert run(scenario()) == {
        "id": "interaction-transport",
        "status": "completed",
    }
    assert attempts == 2
    assert delays == [1.25]


def test_api_backend_rebuilds_gemini_transport_after_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", TEST_API_KEY)
    clients: list[Any] = []

    class FakeAsyncClient:
        def __init__(self, *, fail: bool) -> None:
            self.fail = fail
            self.closed = False

        async def post(self, url: str, *, headers: dict[str, str], json: Any):
            if self.fail:
                self.fail = False
                raise httpx.RemoteProtocolError("stale keep-alive connection")
            request = httpx.Request("POST", url, headers=headers)
            return httpx.Response(
                200,
                request=request,
                json={"id": "fresh-transport", "status": "completed"},
            )

        async def aclose(self) -> None:
            self.closed = True

    def make_client(**_kwargs: Any) -> FakeAsyncClient:
        client = FakeAsyncClient(fail=not clients)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", make_client)

    async def scenario() -> httpx.Response:
        backend = APIBackend(provider="gemini", max_retries=0)
        try:
            with pytest.raises(httpx.RemoteProtocolError):
                await backend._post_gemini_request(
                    "https://example.test/interactions",
                    headers={"content-type": "application/json"},
                    json={"model": "model"},
                )
            return await backend._post_gemini_request(
                "https://example.test/interactions",
                headers={"content-type": "application/json"},
                json={"model": "model"},
            )
        finally:
            await backend.aclose()

    response = run(scenario())

    assert response.json()["id"] == "fresh-transport"
    assert len(clients) == 2
    assert clients[0].closed
    assert clients[1].closed


def test_exhausted_http_retry_preserves_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_test_key(monkeypatch)

    async def no_sleep(_delay: float) -> None:
        return None

    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                503,
                request=request,
                text="upstream temporarily unavailable: diagnostic-id-42",
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = GeminiInteractionsClient(
                client=http,
                max_retries=1,
                sleep=no_sleep,
            )
            await client.create(model="model", input="prompt")

    with pytest.raises(GeminiInteractionsHTTPError) as error:
        run(scenario())

    assert error.value.status_code == 503
    assert error.value.response_body == (
        "upstream temporarily unavailable: diagnostic-id-42"
    )
    assert error.value.response_body in str(error.value)
    assert DEFAULT_INTERACTIONS_URL in str(error.value)
    assert error.value.retry_attempts == 1
    assert error.value.retry_delays
    assert "after 1 retry" in str(error.value)


def test_429_honors_retry_after_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_test_key(monkeypatch)
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario() -> dict[str, Any]:
        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(
                    429,
                    request=request,
                    headers={"Retry-After": "17"},
                    json={"error": {"code": 429, "message": "rate limited"}},
                )
            return httpx.Response(
                200,
                request=request,
                json={"id": "interaction-recovered", "status": "completed"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = GeminiInteractionsClient(
                client=http,
                max_retries=1,
                retry_delay=1.0,
                retry_jitter=0.0,
                sleep=fake_sleep,
            )
            return await client.create(model="model", input="prompt")

    assert run(scenario())["id"] == "interaction-recovered"
    assert delays == [17.0]


def test_429_honors_google_retry_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_test_key(monkeypatch)
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario() -> dict[str, Any]:
        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(
                    429,
                    request=request,
                    json={
                        "error": {
                            "code": 429,
                            "message": "resource exhausted",
                            "details": [
                                {
                                    "@type": (
                                        "type.googleapis.com/google.rpc.RetryInfo"
                                    ),
                                    "retryDelay": "42s",
                                }
                            ],
                        }
                    },
                )
            return httpx.Response(
                200,
                request=request,
                json={"id": "interaction-recovered", "status": "completed"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = GeminiInteractionsClient(
                client=http,
                max_retries=1,
                retry_delay=1.0,
                retry_jitter=0.0,
                sleep=fake_sleep,
            )
            return await client.create(model="model", input="prompt")

    assert run(scenario())["id"] == "interaction-recovered"
    assert delays == [42.0]


def test_successful_invalid_json_is_a_response_error_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_test_key(monkeypatch)
    attempts = 0

    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(200, request=request, text="not-json")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = GeminiInteractionsClient(client=http, max_retries=8)
            await client.create(model="model", input="prompt")

    with pytest.raises(GeminiInteractionsResponseError, match="not-json"):
        run(scenario())
    assert attempts == 1


def test_extractors_support_direct_and_nested_interaction_content() -> None:
    payload = {
        "steps": [
            {"type": "text", "text": "first finding"},
            {
                "type": "function_call",
                "id": "call-1",
                "name": "search",
                "arguments": {"query": "source"},
            },
            {
                "type": "message",
                "content": [
                    {"type": "text", "text": "second finding"},
                    {
                        "type": "image",
                        "mime_type": "image/png",
                        "data": "aW1hZ2U=",
                    },
                ],
            },
        ],
        "output": [
            {"type": "video", "uri": "https://example.test/video.mp4"},
            {
                "content": [
                    {
                        "type": "function-call",
                        "id": "call-2",
                        "name": "inspect",
                        "arguments": "{}",
                    },
                    {
                        "type": "image",
                        "uri": "https://example.test/image.png",
                    },
                ]
            },
        ],
    }

    assert extract_text(payload) == "first finding\nsecond finding"
    assert extract_images(payload) == [
        {"type": "image", "mime_type": "image/png", "data": "aW1hZ2U="},
        {"type": "image", "uri": "https://example.test/image.png"},
    ]
    assert extract_videos(payload) == [
        {"type": "video", "uri": "https://example.test/video.mp4"}
    ]
    assert [call["id"] for call in extract_function_calls(payload)] == [
        "call-1",
        "call-2",
    ]
    assert GeminiInteractionsClient.extract_text(payload) == extract_text(payload)
    assert GeminiInteractionsClient.extract_images(payload) == extract_images(payload)
    assert GeminiInteractionsClient.extract_videos(payload) == extract_videos(payload)
    assert GeminiInteractionsClient.extract_function_calls(payload) == (
        extract_function_calls(payload)
    )


def test_output_text_takes_precedence() -> None:
    payload = {
        "output_text": "canonical output",
        "steps": [{"content": [{"type": "text", "text": "fallback"}]}],
    }
    assert extract_text(payload) == "canonical output"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"status": "completed"}, "interaction id"),
        ({"id": "i1"}, "did not include a status"),
        ({"id": "i1", "status": "incomplete"}, "status=incomplete"),
        (
            {
                "id": "i1",
                "status": "completed",
                "steps": [
                    {
                        "id": "call-1",
                        "type": "function_call",
                        "name": "search",
                        "arguments": {},
                    }
                ],
            },
            "expected status=requires_action",
        ),
        (
            {"id": "i1", "status": "requires_action", "steps": []},
            "did not return a function_call",
        ),
    ],
)
def test_invalid_interaction_envelopes_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    message: str,
) -> None:
    set_test_key(monkeypatch)

    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, request=request, json=payload)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = GeminiInteractionsClient(client=http)
            await client.create(model="model", input="prompt")

    with pytest.raises(GeminiInteractionsResponseError, match=message):
        run(scenario())

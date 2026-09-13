from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import httpx


def _load_gateway(monkeypatch):
    monkeypatch.setenv(
        "QWEN_REPLICA_BACKENDS",
        "http://replica-0,http://replica-1,http://replica-2,http://replica-3",
    )
    path = Path(__file__).parent / "scripts/server/qwen_replica_gateway.py"
    spec = importlib.util.spec_from_file_location("qwen_replica_gateway_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qwen_gateway_balances_concurrent_heavy_tail_requests(monkeypatch) -> None:
    gateway = _load_gateway(monkeypatch)

    reservations = [gateway._acquire_replica() for _ in range(32)]

    assert gateway._replica_loads() == [8, 8, 8, 8]
    for index, _ in reservations:
        gateway._release_replica(index)
    assert gateway._replica_loads() == [0, 0, 0, 0]


def test_qwen_gateway_sends_new_work_to_the_replica_that_drains(monkeypatch) -> None:
    gateway = _load_gateway(monkeypatch)
    reservations = [gateway._acquire_replica() for _ in range(8)]
    assert gateway._replica_loads() == [2, 2, 2, 2]

    gateway._release_replica(2)
    replacement = gateway._acquire_replica()

    assert replacement[0] == 2
    assert gateway._replica_loads() == [2, 2, 2, 2]
    skipped_released_reservation = False
    for index, _ in reservations:
        if index == 2 and not skipped_released_reservation:
            skipped_released_reservation = True
            continue
        gateway._release_replica(index)
    gateway._release_replica(replacement[0])
    assert gateway._replica_loads() == [0, 0, 0, 0]


def test_qwen_gateway_transport_retry_releases_failed_reservation(
    monkeypatch,
) -> None:
    gateway = _load_gateway(monkeypatch)

    class FakeHttp:
        def __init__(self) -> None:
            self.urls: list[str] = []

        async def request(self, method, url, *, content, headers):
            self.urls.append(url)
            if url.startswith("http://replica-0/"):
                raise httpx.ConnectError("replica unavailable")
            return httpx.Response(200, content=b"{}")

    class FakeRequest:
        method = "POST"
        headers = {"content-type": "application/json"}

        def __init__(self) -> None:
            self.app = SimpleNamespace(state=SimpleNamespace(http=FakeHttp()))

        async def body(self) -> bytes:
            return b'{"model":"test"}'

    request = FakeRequest()
    response = asyncio.run(
        gateway._request_replica(
            request,
            "v1/chat/completions",
            retry_transport_once=True,
        )
    )

    assert response.status_code == 200
    assert request.app.state.http.urls == [
        "http://replica-0/v1/chat/completions",
        "http://replica-1/v1/chat/completions",
    ]
    assert response.headers["x-ifv-qwen-replica"] == "http://replica-1"
    assert gateway._replica_loads() == [0, 0, 0, 0]

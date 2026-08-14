"""Small persistent asyncio runtime for synchronous tool adapters."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, Optional


class PersistentAsyncRuntime:
    """Run concurrent coroutines on one long-lived asyncio event loop.

    Synchronous tools call :meth:`run` from worker threads.  A single event loop
    keeps one async HTTP transport reusable while ``run_coroutine_threadsafe``
    still allows independent requests to overlap.
    """

    def __init__(self, *, thread_name: str) -> None:
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._closed = False
        self._startup_error: Optional[BaseException] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=thread_name,
        )
        self._thread = self._executor.submit(self._run_loop)
        self._ready.wait()
        if self._startup_error is not None:
            self.close()
            raise RuntimeError("Persistent asyncio runtime failed to start.") from (
                self._startup_error
            )

    def run(self, coroutine: Awaitable[Any]) -> Any:
        with self._lock:
            if self._closed:
                raise RuntimeError("Persistent asyncio runtime is closed.")
            loop = self._loop
        if loop is None:
            raise RuntimeError("Persistent asyncio runtime is not ready.")
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result()

    def close(
        self,
        cleanup_factory: Optional[Callable[[], Awaitable[Any]]] = None,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop

        if loop is not None and not loop.is_closed():
            if cleanup_factory is not None:
                cleanup = cleanup_factory()
                asyncio.run_coroutine_threadsafe(cleanup, loop).result(timeout=10)
            loop.call_soon_threadsafe(loop.stop)
        self._thread.result(timeout=10)
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._lock:
            self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        except BaseException as exc:
            self._startup_error = exc
            raise
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()


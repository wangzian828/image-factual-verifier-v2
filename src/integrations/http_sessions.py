"""Lifecycle helpers for thread-local requests sessions.

Synchronous tools run provider calls in worker threads.  A plain
``threading.local`` keeps each ``requests.Session`` alive until the worker
thread exits, which is too late for isolated rollout children and can leave
proxy sockets in ``CLOSE-WAIT``.  These helpers retain weakly-scoped owners'
sessions so their orchestrator can close them at the end of each case.
"""

from __future__ import annotations

import threading
from typing import Any

import requests
from requests.adapters import HTTPAdapter


def _new_session() -> requests.Session:
    """Create a requests session that cannot retain proxy keep-alives.

    The gpu-13 egress proxy may half-close idle keep-alive sockets while a
    rollout is still active.  urllib3 can then keep those sockets in its pool
    until the whole case ends, producing unbounded CLOSE-WAIT growth.  A
    ``Connection: close`` request header makes every provider response
    one-shot, while the adapter bounds any transient pool bookkeeping.
    """

    session = requests.Session()
    session.headers.update({"Connection": "close"})
    adapter = HTTPAdapter(
        pool_connections=1,
        pool_maxsize=1,
        max_retries=0,
        pool_block=True,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def init_tracked_sessions(owner: Any) -> None:
    """Initialize the session registry on a client/tool owner."""

    owner._session_registry_lock = threading.Lock()
    owner._tracked_sessions: list[requests.Session] = []
    owner._sessions_closed = False


def get_tracked_session(
    owner: Any,
    thread_local: threading.local,
    *,
    attribute: str = "session",
) -> requests.Session:
    """Return/create a thread-local session and register newly-created ones."""

    session = getattr(thread_local, attribute, None)
    if session is None:
        session = _new_session()
        with owner._session_registry_lock:
            if owner._sessions_closed:
                session.close()
                raise RuntimeError("HTTP session owner is already closed.")
            setattr(thread_local, attribute, session)
            owner._tracked_sessions.append(session)
    return session


def close_tracked_sessions(owner: Any) -> None:
    """Close every session created by an owner, once."""

    lock = getattr(owner, "_session_registry_lock", None)
    if lock is None:
        return
    with lock:
        owner._sessions_closed = True
        sessions = list(getattr(owner, "_tracked_sessions", ()))
        owner._tracked_sessions.clear()
    for session in sessions:
        try:
            session.close()
        except Exception:
            continue


def close_response(response: Any) -> None:
    """Release one requests response independently of its session owner."""

    responses = [response]
    history = getattr(response, "history", ()) or ()
    if isinstance(history, (list, tuple)):
        responses.extend(history)
    for item in responses:
        # ``requests.Response.close()`` only closes ``raw`` when the body has
        # not already been consumed.  With a proxy that half-closes an idle
        # keep-alive connection, a consumed response can otherwise leave the
        # underlying socket in CLOSE-WAIT inside urllib3's pool.  Close the
        # raw transport first, then perform the normal release_conn cleanup.
        raw = getattr(item, "raw", None)
        raw_close = getattr(raw, "close", None)
        if callable(raw_close):
            try:
                raw_close()
            except Exception:
                pass
        close = getattr(item, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                continue

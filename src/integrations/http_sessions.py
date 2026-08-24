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
        session = requests.Session()
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

#!/usr/bin/env python3
"""Execute Python or shell code through a tunneled Jupyter REST/WebSocket API."""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import requests
import websocket


DEFAULT_BASE = os.environ.get("JUPYTER_REMOTE_BASE", "http://127.0.0.1:8333")
DEFAULT_PASSWORD = os.environ.get("JUPYTER_REMOTE_PASSWORD")


def normalize_base(base: str) -> str:
    value = base.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Jupyter base must be an absolute http(s) URL")
    return value


def websocket_url(base: str, path: str) -> str:
    parsed = urlsplit(base)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    base_path = parsed.path.rstrip("/")
    return urlunsplit(
        (scheme, parsed.netloc, f"{base_path}/{path.lstrip('/')}", "", "")
    )


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def login(
    session: requests.Session,
    base: str,
    password: str,
    request_timeout: float,
) -> None:
    response = session.get(f"{base}/login", timeout=request_timeout)
    response.raise_for_status()
    xsrf = session.cookies.get("_xsrf")
    if not xsrf:
        raise RuntimeError("no _xsrf cookie returned by /login")
    response = session.post(
        f"{base}/login",
        data={"_xsrf": xsrf, "password": password},
        allow_redirects=False,
        timeout=request_timeout,
    )
    if response.status_code not in (200, 302):
        raise RuntimeError(f"login failed: HTTP {response.status_code}")


def start_kernel(
    session: requests.Session,
    base: str,
    request_timeout: float,
    kernel_name: str,
) -> str:
    response = session.post(
        f"{base}/api/kernels",
        headers={"X-XSRFToken": session.cookies.get("_xsrf", "")},
        json={"name": kernel_name},
        timeout=request_timeout,
    )
    response.raise_for_status()
    return str(response.json()["id"])


def stop_kernel(
    session: requests.Session,
    base: str,
    kernel_id: str,
    request_timeout: float,
) -> None:
    response = session.delete(
        f"{base}/api/kernels/{kernel_id}",
        headers={"X-XSRFToken": session.cookies.get("_xsrf", "")},
        timeout=request_timeout,
    )
    if response.status_code not in (204, 404):
        response.raise_for_status()


def list_kernels(
    session: requests.Session,
    base: str,
    request_timeout: float,
) -> list[dict[str, object]]:
    response = session.get(f"{base}/api/kernels", timeout=request_timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise TypeError("Jupyter kernel list must be a JSON list")
    return payload


def _contents_url(base: str, remote_path: str) -> str:
    normalized = remote_path.replace("\\", "/").strip("/")
    if not normalized or any(
        part in {"", ".", ".."} for part in normalized.split("/")
    ):
        raise ValueError("remote upload path must be a normalized file path")
    return f"{base}/api/contents/{quote(normalized, safe='/')}"


def _remote_file_size(
    session: requests.Session,
    url: str,
    request_timeout: float,
) -> int | None:
    response = session.get(
        url,
        params={"content": 0},
        timeout=request_timeout,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    if payload.get("type") != "file":
        raise RuntimeError("remote upload target exists and is not a file")
    return int(payload.get("size", 0) or 0)


def upload_file(
    session: requests.Session,
    base: str,
    local_path: Path,
    remote_path: str,
    request_timeout: float,
    chunk_size: int,
) -> None:
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    if chunk_size < 1024 * 1024:
        raise ValueError("upload chunk size must be at least 1 MiB")

    total_size = local_path.stat().st_size
    url = _contents_url(base, remote_path)
    remote_size = _remote_file_size(session, url, request_timeout)
    if remote_size is not None and remote_size > total_size:
        raise RuntimeError(
            f"remote file is larger than local file: {remote_size} > {total_size}"
        )
    if remote_size == total_size:
        print(f"upload already complete: {remote_path} ({total_size} bytes)")
        return

    offset = int(remote_size or 0)
    headers = {"X-XSRFToken": session.cookies.get("_xsrf", "")}
    with local_path.open("rb") as handle:
        handle.seek(offset)
        chunk_index = 2 if offset else 1
        while offset < total_size:
            content = handle.read(min(chunk_size, total_size - offset))
            if not content:
                raise IOError("local file ended before the recorded size")
            next_offset = offset + len(content)
            is_last = next_offset == total_size
            if offset == 0 and is_last:
                chunk_marker = None
            else:
                chunk_marker = -1 if is_last else chunk_index
            payload = {
                "type": "file",
                "format": "base64",
                "content": base64.b64encode(content).decode("ascii"),
            }
            if chunk_marker is not None:
                payload["chunk"] = chunk_marker
            response = session.put(
                url,
                headers=headers,
                json=payload,
                timeout=max(request_timeout, 120.0),
            )
            response.raise_for_status()
            offset = next_offset
            chunk_index += 1
            print(
                f"uploaded {offset}/{total_size} bytes "
                f"({offset / max(total_size, 1):.1%})",
                flush=True,
            )

    confirmed_size = _remote_file_size(session, url, request_timeout)
    if confirmed_size != total_size:
        raise RuntimeError(
            f"upload size verification failed: remote={confirmed_size}, local={total_size}"
        )


def execute_code(
    session: requests.Session,
    base: str,
    kernel_id: str,
    code: str,
    timeout: int,
) -> int:
    cookies = "; ".join(
        f"{key}={value}" for key, value in session.cookies.get_dict().items()
    )
    connection = websocket.create_connection(
        websocket_url(base, f"api/kernels/{kernel_id}/channels"),
        header=[f"Cookie: {cookies}"],
        timeout=timeout,
    )
    message_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    connection.send(
        json.dumps(
            {
                "header": {
                    "msg_id": message_id,
                    "username": "remote",
                    "session": session_id,
                    "msg_type": "execute_request",
                    "version": "5.3",
                    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                "parent_header": {},
                "metadata": {},
                "content": {
                    "code": code,
                    "silent": False,
                    "store_history": False,
                    "user_expressions": {},
                    "allow_stdin": False,
                    "stop_on_error": False,
                },
                "channel": "shell",
                "buffers": [],
            }
        )
    )

    deadline = time.time() + timeout
    exit_code = 0
    saw_reply = False
    saw_idle = False
    try:
        while time.time() < deadline:
            connection.settimeout(max(0.5, deadline - time.time()))
            try:
                raw = connection.recv()
            except websocket.WebSocketTimeoutException:
                break
            try:
                message = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if (message.get("parent_header") or {}).get("msg_id") != message_id:
                continue
            message_type = message.get("msg_type") or message.get("header", {}).get(
                "msg_type"
            )
            content = message.get("content", {})
            if message_type == "stream":
                target = sys.stdout if content.get("name") == "stdout" else sys.stderr
                target.write(content.get("text", ""))
            elif message_type == "execute_result":
                rendered = content.get("data", {}).get("text/plain")
                if rendered is not None:
                    sys.stdout.write(f"{rendered}\n")
            elif message_type == "error":
                sys.stderr.write("\n".join(content.get("traceback", [])) + "\n")
                exit_code = 1
            elif message_type == "execute_reply":
                saw_reply = True
                if content.get("status") == "error":
                    exit_code = 1
            elif message_type == "status" and content.get("execution_state") == "idle":
                saw_idle = True
            if saw_reply and saw_idle:
                return exit_code
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        connection.close()
    raise TimeoutError(f"kernel execution did not finish within {timeout} seconds")


def wrap_shell(command: str) -> str:
    return (
        "import os as _os, subprocess as _sp, sys as _sys\n"
        "_env = _os.environ.copy()\n"
        "_env['PYTHONIOENCODING'] = 'utf-8'\n"
        "_env['LANG'] = _env.get('LANG') or 'C.UTF-8'\n"
        f"_r = _sp.run({command!r}, shell=True, executable='/bin/bash', "
        "capture_output=True, text=True, encoding='utf-8', errors='replace', env=_env)\n"
        "_sys.stdout.write(_r.stdout)\n"
        "_sys.stderr.write(_r.stderr)\n"
        "_sys.stdout.flush(); _sys.stderr.flush()\n"
        "if _r.returncode:\n"
        "    raise SystemExit(_r.returncode)\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument(
        "--kernel-name",
        default=os.environ.get("JUPYTER_REMOTE_KERNEL", "python3"),
        help=(
            "Jupyter kernelspec to start "
            "(default: JUPYTER_REMOTE_KERNEL or python3)"
        ),
    )
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--shell", "-s", action="store_true")
    parser.add_argument("--timeout", "-t", type=int, default=120)
    parser.add_argument("--kernel", "-k")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--list-kernels", action="store_true")
    parser.add_argument("--stop-kernel")
    parser.add_argument(
        "--upload",
        nargs=2,
        metavar=("LOCAL_PATH", "REMOTE_PATH"),
        help="upload one file through the Jupyter Contents API with resume support",
    )
    parser.add_argument("--upload-chunk-mib", type=int, default=8)
    parser.add_argument("--stdin", action="store_true")
    parser.add_argument("code", nargs="?", default="")
    return parser.parse_args()


def main() -> int:
    configure_stdio()
    args = parse_args()
    base = normalize_base(args.base)
    password = args.password
    if not password:
        if args.stdin or not sys.stdin.isatty():
            raise SystemExit(
                "set JUPYTER_REMOTE_PASSWORD or pass --password in an interactive shell"
            )
        password = getpass.getpass("Jupyter password: ")

    session = requests.Session()
    login(session, base, password, args.request_timeout)

    if args.list_kernels:
        for kernel in list_kernels(session, base, args.request_timeout):
            print(
                kernel.get("id"),
                kernel.get("name"),
                kernel.get("execution_state"),
                kernel.get("last_activity"),
            )
        return 0
    if args.stop_kernel:
        stop_kernel(session, base, args.stop_kernel, args.request_timeout)
        print(f"stopped {args.stop_kernel}")
        return 0
    if args.upload:
        upload_file(
            session,
            base,
            Path(args.upload[0]).expanduser().resolve(),
            args.upload[1],
            args.request_timeout,
            max(1, int(args.upload_chunk_mib)) * 1024 * 1024,
        )
        return 0

    code = sys.stdin.read() if args.stdin else args.code
    code = code.lstrip("\ufeff")
    if not code:
        raise SystemExit("no code given")
    if args.shell:
        code = wrap_shell(code)

    kernel_id = args.kernel or start_kernel(
        session,
        base,
        args.request_timeout,
        args.kernel_name,
    )
    started_here = args.kernel is None
    try:
        return execute_code(session, base, kernel_id, code, args.timeout)
    finally:
        if started_here and not args.keep:
            stop_kernel(session, base, kernel_id, args.request_timeout)
        elif started_here:
            print(f"[kernel-id: {kernel_id}]", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
